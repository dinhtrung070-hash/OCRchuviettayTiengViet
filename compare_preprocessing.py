"""Ablate form boxes and ink filtering with the original VietOCR weights.

Uses the two old demo forms as a regression check and the existing validation
forms for exploratory selection. Never uses the test split to select a mode.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from vietocr.tool.predictor import Predictor

from evaluate_test import edit_distance
from ocr_review_app import BASE_DIR, get_model_config
from prepare_dataset import (FIELDS, SCALE, blue_handwriting_only,
                             dark_handwriting_only, estimate_alignment, read_image,
                             write_image)


LEGACY_FIELDS = {
    "ho_ten": (185, 105, 820, 157),
    "ngay_sinh": (185, 157, 365, 212),
    "dan_toc": (401, 157, 590, 212),
    "ton_giao": (635, 157, 825, 212),
}
EXTENDED_FIELDS = {
    "ho_ten": FIELDS["ho_ten"],
    "ngay_sinh": (195, 145, 365, 212),
    "dan_toc": (401, 145, 590, 212),
    "ton_giao": (635, 145, 825, 212),
}


def legacy_blue(region: np.ndarray) -> tuple[np.ndarray, bool]:
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([90, 45, 25]), np.array([140, 255, 255]))
    if cv2.countNonZero(mask) < 40:
        return region, False
    mask = cv2.dilate(mask, np.ones((2, 2), np.uint8), iterations=1)
    ys, xs = np.where(mask > 0)
    pad = 12
    x1, x2 = max(0, int(xs.min()) - pad), min(region.shape[1], int(xs.max()) + pad + 1)
    y1, y2 = max(0, int(ys.min()) - pad), min(region.shape[0], int(ys.max()) + pad + 1)
    cleaned = np.full_like(region, 255)
    cleaned[mask > 0] = region[mask > 0]
    return cleaned[y1:y2, x1:x2], True


def aligned_pair(template: np.ndarray, photo: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    transform, _ = estimate_alignment(template, photo)
    height, width = template.shape[:2]
    aligned = cv2.warpPerspective(photo, np.diag([SCALE, SCALE, 1.0]) @ np.linalg.inv(transform),
                                  (width * SCALE, height * SCALE), flags=cv2.INTER_CUBIC,
                                  borderValue=(255, 255, 255))
    blank = cv2.resize(template, (width * SCALE, height * SCALE), interpolation=cv2.INTER_CUBIC)
    return aligned, blank


def crop(aligned: np.ndarray, blank: np.ndarray, field: str, mode: str) -> tuple[np.ndarray, str]:
    boxes = (LEGACY_FIELDS if mode.startswith("legacy_box") else
             EXTENDED_FIELDS if mode.startswith("extended_box") else FIELDS)
    a, b, c, d = [v * SCALE for v in boxes[field]]
    region, blank_region = aligned[b:d, a:c], blank[b:d, a:c]
    if mode.endswith("legacy_blue"):
        cleaned, blue = legacy_blue(region)
    elif mode.endswith("component_blue"):
        cleaned, blue = blue_handwriting_only(region)
    else:
        cleaned, blue = blue_handwriting_only(region, blank_region)
    if blue:
        return cleaned, "blue"
    cleaned, dark = dark_handwriting_only(region, blank_region)
    return cleaned, "dark" if dark else "raw"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=BASE_DIR / "prepared" / "preprocess_ablation")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    valid_root = BASE_DIR / "prepared" / "dataset_20261102"
    source_root = BASE_DIR / "forms" / "dataset_20261102"
    examples = []
    with (valid_root / "valid_annotation.txt").open(encoding="utf-8-sig") as handle:
        for line in handle:
            relative, label = line.rstrip("\n").split("\t", 1)
            form_stem, field_name = Path(relative).parts[0], Path(relative).stem
            examples.append(("valid", source_root / f"{form_stem}.jpg", field_name, label))
    with (BASE_DIR / "pilot_labels.csv").open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            for field_name in FIELDS:
                examples.append(("pilot", BASE_DIR / row["form"], field_name, row[field_name]))
    template = read_image(BASE_DIR / "phieu_mau.png")
    predictor = Predictor(get_model_config(None))
    modes = ("legacy_box_component_blue", "tight_box_component_blue")
    cache = {}
    results = []
    for split, path, field_name, label in examples:
        if path not in cache:
            cache[path] = aligned_pair(template, read_image(path))
        aligned, blank = cache[path]
        for mode in modes:
            image, ink = crop(aligned, blank, field_name, mode)
            output = args.output_dir / mode / path.stem / f"{field_name}.png"
            output.parent.mkdir(parents=True, exist_ok=True)
            write_image(output, image)
            raw_prediction, confidence = predictor.predict(Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), return_prob=True)
            prediction = " ".join(raw_prediction.split())
            results.append({"split": split, "form": path.name, "field": field_name,
                            "mode": mode, "ink": ink, "expected": label,
                            "predicted": prediction, "exact": prediction == label,
                            "distance": edit_distance(label, prediction), "confidence": float(confidence),
                            "image": str(output)})
        print(f"{split} {path.name} {field_name}", flush=True)
    summary = {}
    for mode in modes:
        summary[mode] = {}
        for split in ("pilot", "valid"):
            subset = [row for row in results if row["mode"] == mode and row["split"] == split]
            summary[mode][split] = {"exact": sum(row["exact"] for row in subset), "total": len(subset),
                                    "cer": sum(row["distance"] for row in subset) /
                                           sum(len(row["expected"]) for row in subset)}
    report = {"summary": summary, "rows": results,
              "note": "Valid is exploratory: preprocessing choices were previously informed by this dataset; pilot is in-sample."}
    (args.output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
