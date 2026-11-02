"""Compare pretrained and fine-tuned VietOCR on valid or held-out test crops."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
from PIL import Image
from vietocr.tool.predictor import Predictor

from ocr_review_app import BASE_DIR, get_model_config
from train_extended1 import read_annotations


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
    }


def run(weights: Path | None, rows: list[tuple[Path, str]], metadata: dict[str, dict]) -> dict:
    config = get_model_config(str(weights.resolve()) if weights else None)
    predictor = Predictor(config)
    results = []
    for image_path, label in rows:
        with Image.open(image_path) as image:
            prediction = " ".join(predictor.predict(image).split())
        info = metadata[image_path.relative_to(metadata["_root"]).as_posix()]
        results.append({"form": info["form"], "field": info["field"],
                        "ink": "blue" if info["blue_ink_detected"] == "True" else "dark",
                        "expected": label, "predicted": prediction,
                        "exact": prediction == label,
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
    parser.add_argument("--split", choices=["valid", "test"], default="test")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if not args.weights.is_file():
        parser.error(f"Missing fine-tuned weights: {args.weights}")
    torch.set_num_threads(args.threads)
    rows = read_annotations(args.data_dir, f"{args.split}_annotation.txt")
    with (args.data_dir / "to_label.csv").open(newline="", encoding="utf-8-sig") as handle:
        metadata = {row["image"]: row for row in csv.DictReader(handle)}
    metadata["_root"] = args.data_dir.resolve()
    baseline = run(None, rows, metadata)
    tuned = run(args.weights, rows, metadata)
    split_path = args.data_dir / "split_manifest.json"
    split_info = json.loads(split_path.read_text(encoding="utf-8")) if split_path.is_file() else {}
    report = {"split": args.split, "group_type": split_info.get("group_type", "unknown"),
              "baseline": baseline, "fine_tuned": tuned}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"baseline": {k: baseline[k] for k in ("correct", "total")},
                      "fine_tuned": {k: tuned[k] for k in ("correct", "total")}}), flush=True)
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
