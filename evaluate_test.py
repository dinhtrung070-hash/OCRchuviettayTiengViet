"""Compare pretrained and fine-tuned VietOCR on valid or held-out test crops."""

from __future__ import annotations

import argparse
import csv
import json
import hashlib
from pathlib import Path

import torch
from PIL import Image

from ocr_review_app import BASE_DIR, get_field_predictor
from train_extended1 import read_annotations
from field_inference import normalize_value, predict_field
from handwriting_preprocess import PREPROCESS_VERSION


def edit_distance(left: str, right: str) -> int:
    previous = list(range(len(right) + 1))
    for index, first in enumerate(left, 1):
        current = [index]
        for column, second in enumerate(right, 1):
            current.append(min(current[-1] + 1, previous[column] + 1,
                               previous[column - 1] + (first != second)))
        previous = current
    return previous[-1]


def summarize(rows: list[dict]) -> dict:
    return {
        "correct": sum(row["exact"] for row in rows),
        "total": len(rows),
        "cer": sum(row["edit_distance"] for row in rows) /
               max(1, sum(len(row["expected"]) for row in rows)),
        "formatted_correct": sum(row.get("formatted_exact", row["exact"]) for row in rows),
        "structured_correct": sum(row.get("structured_exact", row.get("formatted_exact", row["exact"])) for row in rows),
    }


def run(weights: Path | None, rows: list[tuple[Path, str]], metadata: dict[str, dict], decoder: str = "greedy", structured: bool = False) -> dict:
    results = []
    for image_path, label in rows:
        info = metadata[image_path.relative_to(metadata["_root"]).as_posix()]
        predictor = get_field_predictor(str(weights.resolve()) if weights else None, info["field"])
        with Image.open(image_path) as image:
            detail = predict_field(predictor, image, info["field"], decoder, structured)
            prediction = detail["raw"]
        results.append({"form": info["form"], "field": info["field"],
                        "ink": "blue" if info["blue_ink_detected"] == "True" else "dark",
                        "expected": label, "predicted": prediction,
                        "exact": prediction == label,
                        "formatted": normalize_value(info["field"], prediction),
                        "formatted_exact": normalize_value(info["field"], prediction).casefold() == normalize_value(info["field"], label).casefold(),
                        "structured_value": detail["value"], "structured_proposal": detail["structured_proposal"],
                        "structured_exact": detail["value"].casefold() == normalize_value(info["field"], label).casefold(),
                        "model_score": detail["model_score"],
                        "model_metadata": predictor.form_metadata,
                        "edit_distance": edit_distance(label, prediction)})
    summary = summarize(results)
    summary["by_field"] = {field: summarize([row for row in results if row["field"] == field])
                           for field in sorted({row["field"] for row in results})}
    summary["by_ink"] = {ink: summarize([row for row in results if row["ink"] == ink])
                         for ink in ("blue", "dark")}
    summary["rows"] = results
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=BASE_DIR / "prepared")
    parser.add_argument("--weights", type=Path, default=BASE_DIR / "models" / "extended1_finetuned.pth")
    parser.add_argument("--output", type=Path, default=BASE_DIR / "models" / "extended1_test.json")
    parser.add_argument("--split", choices=["train", "valid", "test"], default="test")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--decoder", choices=["greedy", "beam"], default="greedy")
    parser.add_argument("--structured", action="store_true", help="Also evaluate model-based field/date proposals separately from raw OCR")
    parser.add_argument("--previous-weights", type=Path, help="Optional previous checkpoint, evaluated on the same crops")
    parser.add_argument("--previous-label", default="Fine-tune v2", help="Display label for the comparison checkpoint")
    args = parser.parse_args()
    if not args.weights.is_file():
        parser.error(f"Missing fine-tuned weights: {args.weights}")
    torch.set_num_threads(args.threads)
    rows = read_annotations(args.data_dir, f"{args.split}_annotation.txt")
    with (args.data_dir / "to_label.csv").open(newline="", encoding="utf-8-sig") as handle:
        metadata = {row["image"]: row for row in csv.DictReader(handle)}
    metadata["_root"] = args.data_dir.resolve()
    baseline = run(None, rows, metadata, args.decoder, args.structured)
    tuned = run(args.weights, rows, metadata, args.decoder, args.structured)
    split_path = args.data_dir / "split_manifest.json"
    split_info = json.loads(split_path.read_text(encoding="utf-8")) if split_path.is_file() else {}
    profile = args.data_dir / "preprocess.json"
    version = json.loads(profile.read_text(encoding="utf-8"))["version"] if profile.is_file() else PREPROCESS_VERSION
    report = {"split": args.split, "group_type": split_info.get("group_type", "unknown"),
              "preprocess_version": version, "weights": str(args.weights.resolve()),
              "decoder": args.decoder,
              "structured_decoding": args.structured,
              "structured_metric": "Case-insensitive field values after image-conditioned date grammar/vocabulary proposals; reported separately from raw OCR.",
              "weights_sha256": hashlib.sha256(args.weights.read_bytes()).hexdigest(),
              "annotation_sha256": hashlib.sha256((args.data_dir / f"{args.split}_annotation.txt").read_bytes()).hexdigest(),
              "formatted_metric": "Ignore letter case; standardize valid numeric dates. No fuzzy replacements or suggestions are counted as correct.",
              "limits": "Writer IDs unknown; reused test supports exploratory evaluation only.",
              "baseline": baseline, "fine_tuned": tuned}
    if args.previous_weights:
        report["previous_v2"] = run(args.previous_weights, rows, metadata, args.decoder, args.structured)
        report["previous_model_label"] = args.previous_label
        report["previous_weights_sha256"] = hashlib.sha256(args.previous_weights.read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"baseline": {k: baseline[k] for k in ("correct", "total")},
                      "fine_tuned": {k: tuned[k] for k in ("correct", "total")}}), flush=True)
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
