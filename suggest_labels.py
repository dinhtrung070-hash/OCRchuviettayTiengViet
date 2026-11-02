"""Generate VietOCR suggestions for manual transcription of form crops.

Suggestions are never treated as ground truth. Check each against the source
form before copying it into to_label.csv.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from PIL import Image
from vietocr.tool.predictor import Predictor

from ocr_review_app import get_model_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    manifest = args.data_dir / "to_label.csv"
    with manifest.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    predictor = Predictor(get_model_config(None))
    output = args.data_dir / "suggestions.csv"
    with output.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=["form", "field", "image", "prediction", "label"])
        writer.writeheader()
        for index, row in enumerate(rows, 1):
            with Image.open(args.data_dir / row["image"]) as image:
                prediction = " ".join(predictor.predict(image).split())
            writer.writerow({"form": row["form"], "field": row["field"],
                             "image": row["image"], "prediction": prediction,
                             "label": row["label"]})
            if index % 20 == 0:
                print(f"{index}/{len(rows)} crops", flush=True)
    print(f"Saved suggestions: {output}", flush=True)


if __name__ == "__main__":
    main()
