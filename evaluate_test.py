"""Evaluate pretrained and fine-tuned VietOCR on the untouched test split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from vietocr.tool.predictor import Predictor

from ocr_review_app import BASE_DIR, get_model_config
from train_extended1 import read_annotations


def run(weights: Path | None, rows: list[tuple[Path, str]]) -> dict:
    config = get_model_config(str(weights.resolve()) if weights else None)
    predictor = Predictor(config)
    results = []
    for image_path, label in rows:
        with Image.open(image_path) as image:
            prediction = " ".join(predictor.predict(image).split())
        results.append({"image": image_path.name, "expected": label, "predicted": prediction, "exact": prediction == label})
    return {"correct": sum(row["exact"] for row in results), "total": len(results), "rows": results}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=BASE_DIR / "prepared")
    parser.add_argument("--weights", type=Path, default=BASE_DIR / "models" / "extended1_finetuned.pth")
    parser.add_argument("--output", type=Path, default=BASE_DIR / "models" / "extended1_test.json")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if not args.weights.is_file():
        parser.error(f"Missing fine-tuned weights: {args.weights}")
    torch.set_num_threads(args.threads)
    rows = read_annotations(args.data_dir, "test_annotation.txt")
    baseline = run(None, rows)
    tuned = run(args.weights, rows)
    report = {"split": "test", "baseline": baseline, "fine_tuned": tuned}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"baseline": {k: baseline[k] for k in ("correct", "total")},
                      "fine_tuned": {k: tuned[k] for k in ("correct", "total")}}), flush=True)
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
