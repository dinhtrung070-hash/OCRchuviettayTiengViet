"""Fixed input trials for the v4 bundle, using TRAIN/VALID only.

No active crops, model weights, bundle, or test files are changed/read. Geometry
rules are independent of labels; transcripts are used only for validation.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path

import cv2
from PIL import Image, ImageDraw, ImageFont, ImageOps

from handwriting_preprocess import clean_name_line
from prepare_dataset import FIELDS, SCALE, read_image
from train_extended1 import read_annotations
from v4_name_geometry import major_cluster
from v4_preprocess_experiments import chroma_mask, periodic_gaps

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "prepared" / "extended1_v2"
OUTPUT = ROOT / "prepared" / "v5_input_trials"
NAMES = ("baseline", "major_cluster", "name_focus", "chroma_mask", "periodic_gaps",
         "aspect_093", "aspect_107", "pad_y005")


def from_bgr(image):
    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


def variants(path: Path, blank) -> dict[str, Image.Image]:
    field = path.stem
    image = Image.open(path).convert("RGB")
    raw = read_image(path.with_name(field + "_raw.png"))
    a, b, c, d = [value*SCALE for value in FIELDS[field]]
    background = blank[b:d, a:c]
    return {"baseline": image,
            "major_cluster": major_cluster(image) if field == "ho_ten" else image.copy(),
            "name_focus": from_bgr(clean_name_line(raw, background)) if field == "ho_ten" else image.copy(),
            "chroma_mask": from_bgr(chroma_mask(raw, background)),
            "periodic_gaps": from_bgr(periodic_gaps(raw, background)),
            "aspect_093": image.resize((max(1, round(image.width*.93)), image.height), Image.Resampling.LANCZOS),
            "aspect_107": image.resize((max(1, round(image.width*1.07)), image.height), Image.Resampling.LANCZOS),
            "pad_y005": ImageOps.expand(image, border=(0, max(1, round(image.height*.05)), 0, max(1, round(image.height*.05))), fill="white")}


def review(rows, title, prefix):
    font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 16)
    for offset in (0, 4):
        names = NAMES[offset:offset+4]
        canvas = Image.new("RGB", (1680, len(rows)*150), "#eeeeee")
        draw = ImageDraw.Draw(canvas)
        for i, row in enumerate(rows):
            for col, name in enumerate(names):
                x, y = col*420, i*150
                draw.text((x+5, y+3), f"{title} {row['image']} / {name}", font=font, fill="black")
                with Image.open(row["variants"][name]) as picture:
                    picture.thumbnail((410, 115))
                    canvas.paste(picture, (x+5, y+30))
        canvas.save(OUTPUT / f"{prefix}_{offset//4+1}.png")


def prepare():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    blank = read_image(ROOT / "phieu_mau.png")
    blank = cv2.resize(blank, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_CUBIC)
    rows = []
    valid = read_annotations(DATA, "valid_annotation.txt")
    train = [(path, label) for path, label in read_annotations(DATA, "train_annotation.txt")
             if path.stem == "ho_ten" and path.parent.name in {"31", "32", "33"}]
    for split, samples in (("valid", valid), ("train_review", train)):
        for path, label in samples:
            record = {"image": path.relative_to(DATA.resolve()).as_posix(), "field": path.stem,
                      "expected": label, "variants": {}}
            for name, picture in variants(path, blank).items():
                target = OUTPUT / split / name / record["image"]
                target.parent.mkdir(parents=True, exist_ok=True)
                picture.save(target)
                record["variants"][name] = str(target)
            if split == "valid":
                rows.append(record)
            else:
                rows.append(dict(record, review_only=True))
    for field in FIELDS:
        review([row for row in rows if row["field"] == field and not row.get("review_only")], "valid", field)
    review([row for row in rows if row.get("review_only")], "train", "train_names")
    (OUTPUT / "manifest.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    return [row for row in rows if not row.get("review_only")]


def summary(rows):
    return {"correct": sum(row["exact"] for row in rows), "total": len(rows),
            "cer": sum(row["distance"] for row in rows)/sum(len(row["expected"]) for row in rows),
            "formatted_correct": sum(row["formatted_exact"] for row in rows),
            "structured_correct": sum(row["structured_exact"] for row in rows)}


def evaluate(records):
    import torch
    from field_inference import predict_field, normalize_value, text_distance
    from ocr_review_app import get_field_predictor, get_predictor
    from field_bundle import load_bundle
    bundle = ROOT / "models" / "extended1_bundle.json"
    checksum = hashlib.sha256(bundle.read_bytes()).hexdigest()
    result = {"split": "valid", "test_accessed": False,
              "weights_sha256": checksum,
              "annotation_sha256": hashlib.sha256((DATA / "valid_annotation.txt").read_bytes()).hexdigest(),
              "preprocess_reference": "stroke_preserving_v2", "results": {name: {"rows": []} for name in NAMES}}
    for field in FIELDS:
        predictor = get_field_predictor(str(bundle.resolve()), field)
        torch.set_num_threads(2)
        for name in NAMES:
            for record in [row for row in records if row["field"] == field]:
                with Image.open(record["variants"][name]) as picture:
                    detail = predict_field(predictor, picture.convert("RGB"), field)
                expected = record["expected"]
                result["results"][name]["rows"].append({"image": record["image"], "field": field,
                    "expected": expected, "raw": detail["raw"], "value": detail["value"],
                    "model_score": detail["model_score"], "warnings": detail["warnings"],
                    "distance": text_distance(expected, detail["raw"]), "exact": expected == detail["raw"],
                    "formatted_exact": normalize_value(field, expected).casefold() == normalize_value(field, detail["raw"]).casefold(),
                    "structured_exact": normalize_value(field, expected).casefold() == detail["value"].casefold()})
        del predictor
        get_predictor.cache_clear()
        load_bundle.cache_clear()
        gc.collect()
        print("completed", field, flush=True)
    for name, item in result["results"].items():
        rows = item["rows"]
        item.update(summary(rows))
        item["by_field"] = {field: summary([row for row in rows if row["field"] == field]) for field in FIELDS}
        print(name, {key: item[key] for key in ("correct", "cer", "formatted_correct", "structured_correct")}, flush=True)
    if hashlib.sha256(bundle.read_bytes()).hexdigest() != checksum:
        raise ValueError("Active bundle changed during the controlled experiment")
    (ROOT / "models" / "v5_input_valid.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluate", action="store_true")
    args = parser.parse_args()
    rows = prepare()
    print("prepared", len(rows), "validation crops x", len(NAMES), "variants", flush=True)
    if args.evaluate:
        evaluate(rows)
