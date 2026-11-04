"""Validation-only, label-independent preprocessing experiments.

Writes variants under prepared/, never changes active crops or the model bundle.
The default preparation step performs no model inference.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from handwriting_preprocess import blue_signal, clean_handwriting, normalize_paper, remove_isolated_dots, trim_ink
from prepare_dataset import FIELDS, SCALE, read_image, write_image

ROOT = Path(__file__).resolve().parent
VARIANTS = ("baseline", "chroma", "median3", "clahe", "periodic_gaps", "chroma_mask")


def chroma_mask(region: np.ndarray, blank: np.ndarray) -> np.ndarray:
    """Use chroma to locate blue strokes, retain their original gray intensity."""
    signal, blue = blue_signal(region)
    if not blue:
        return clean_handwriting(region, blank, mode="dots")[0]
    gray = normalize_paper(region)
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    colored = np.uint8((signal > 10) & (hsv[:, :, 0] >= 85) & (hsv[:, :, 0] <= 145) & (hsv[:, :, 1] > 25))
    # Dilate the chroma support so faded/anti-aliased stroke edges survive.
    support = cv2.dilate(colored, np.ones((5, 5), np.uint8))
    gray[support == 0] = 255
    gray, foreground = remove_isolated_dots(gray, blank)
    return cv2.cvtColor(trim_ink(gray, foreground), cv2.COLOR_GRAY2BGR)


def periodic_gaps(region: np.ndarray, blank: np.ndarray) -> np.ndarray:
    """Suppress detached baseline dots only in whitespace gaps between letters.

Whole stroke components and marks directly above/below letters are retained.
This is experimental; it cannot distinguish every black accent from print.
"""
    gray, foreground = remove_isolated_dots(normalize_paper(region), blank)
    b = cv2.cvtColor(blank, cv2.COLOR_BGR2GRAY)
    n, _, boxes, centers = cv2.connectedComponentsWithStats(np.uint8(b < 190))
    dots = [i for i in range(1, n) if 3 <= boxes[i, 4] <= 160 and 2 <= boxes[i, 2] <= 16 and 2 <= boxes[i, 3] <= 16]
    if len(dots) < 4:
        return cv2.cvtColor(trim_ink(gray, foreground), cv2.COLOR_GRAY2BGR)
    baseline = float(np.median(centers[dots, 1]))
    dots = [i for i in dots if abs(centers[i, 1]-baseline) <= 3]
    pitch = float(np.median(np.diff(sorted(centers[dots, 0]))))
    count, ids, stats, centroids = cv2.connectedComponentsWithStats(foreground)
    major = [i for i in range(1, count) if stats[i, 4] >= 30 and stats[i, 3] >= max(12, gray.shape[0]*.14)]
    for i in range(1, count):
        x, y, w, h, area = stats[i]
        cx, cy = centroids[i]
        if not (3 <= area <= 160 and w <= 16 and h <= 16 and .5 <= w/max(1, h) <= 2 and abs(cy-baseline) <= 6):
            continue
        # Accents are normally horizontally near a substantial letter stroke.
        # Retain these even if they resemble the printed baseline dots.
        nearby = any(stats[j, 0]-pitch*.6 <= cx <= stats[j, 0]+stats[j, 2]+pitch*.6 for j in major)
        if nearby:
            continue
        mask = cv2.dilate(np.uint8(ids == i), np.ones((3, 3), np.uint8))
        gray[mask > 0] = 255
        foreground[mask > 0] = 0
    return cv2.cvtColor(trim_ink(gray, foreground), cv2.COLOR_GRAY2BGR)


def variants(region: np.ndarray, blank: np.ndarray, baseline: np.ndarray) -> dict[str, np.ndarray]:
    gray = cv2.cvtColor(baseline, cv2.COLOR_BGR2GRAY)
    return {"baseline": baseline,
            "chroma": clean_handwriting(region, blank, mode="chroma")[0],
            "median3": cv2.cvtColor(cv2.medianBlur(gray, 3), cv2.COLOR_GRAY2BGR),
            "clahe": cv2.cvtColor(cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8)).apply(gray), cv2.COLOR_GRAY2BGR),
            "periodic_gaps": periodic_gaps(region, blank),
            "chroma_mask": chroma_mask(region, blank)}


def prepare() -> list[dict]:
    source = ROOT / "prepared" / "extended1_v2"
    target = ROOT / "prepared" / "v4_preprocess"
    blank = read_image(ROOT / "phieu_mau.png")
    blank = cv2.resize(blank, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_CUBIC)
    records = []
    for line in (source / "valid_annotation.txt").read_text(encoding="utf-8").splitlines():
        rel, label = line.split("\t", 1)
        path = source / rel
        field = path.stem
        a, b, c, d = [v*SCALE for v in FIELDS[field]]
        images = variants(read_image(path.with_name(field+"_raw.png")), blank[b:d, a:c], read_image(path))
        outputs = {}
        for name, picture in images.items():
            output = target / name / rel
            output.parent.mkdir(parents=True, exist_ok=True)
            write_image(output, picture)
            outputs[name] = str(output)
        records.append({"image": rel, "field": field, "expected": label, "variants": outputs})
    (target / "manifest.json").write_text(json.dumps({"split": "valid", "records": records}, ensure_ascii=False, indent=2), encoding="utf-8")
    font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 16)
    for page in range(4):
        sheet = Image.new("RGB", (2280, 1080), "#eeeeee")
        draw = ImageDraw.Draw(sheet)
        for j, record in enumerate(records[page*7:page*7+7]):
            for col, name in enumerate(VARIANTS):
                x, y = col*380, j*150
                draw.text((x+5, y+3), f"{record['image']} / {name}", font=font, fill="black")
                image = Image.open(record["variants"][name]).convert("RGB")
                image.thumbnail((370, 110))
                sheet.paste(image, (x+5, y+30))
        sheet.save(target / f"review_{page+1}.png")
    return records


def evaluate(records: list[dict]) -> None:
    import torch
    from field_inference import predict_field, normalize_value
    from ocr_review_app import get_field_predictor
    from train_extended1 import edit_distance
    torch.set_num_threads(2)
    results = {}
    for name in VARIANTS:
        rows = []
        for record in records:
            predictor = get_field_predictor(str(ROOT / "models" / "extended1_bundle.json"), record["field"])
            torch.set_num_threads(2)
            with Image.open(record["variants"][name]) as image:
                detail = predict_field(predictor, image, record["field"])
            label = record["expected"]
            rows.append({"image": record["image"], "field": record["field"], "expected": label,
                         "raw": detail["raw"], "value": detail["value"], "model_score": detail["model_score"],
                         "distance": edit_distance(label, detail["raw"]), "exact": label == detail["raw"],
                         "structured_exact": detail["value"].casefold() == normalize_value(record["field"], label).casefold()})
        def summary(rows):
            return {"correct": sum(r["exact"] for r in rows), "total": len(rows), "cer": sum(r["distance"] for r in rows)/sum(len(r["expected"]) for r in rows), "structured_correct": sum(r["structured_exact"] for r in rows)}
        results[name] = {**summary(rows), "by_field": {field: summary([r for r in rows if r["field"] == field]) for field in FIELDS}, "rows": rows}
        print(name, {k: results[name][k] for k in ("correct", "total", "cer", "structured_correct")}, flush=True)
        (ROOT / "models" / "v4_preprocess_valid.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluate", action="store_true")
    args = parser.parse_args()
    records = prepare()
    print("Prepared", len(records), "validation fields in", len(VARIANTS), "variants", flush=True)
    if args.evaluate:
        evaluate(records)
