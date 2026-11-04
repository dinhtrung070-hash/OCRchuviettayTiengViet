"""Validation-only name geometry experiments with generic, fixed rules.

No label-dependent cropping, dictionary, test labels, or active model changes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from field_inference import text_distance
from ocr_review_app import BASE_DIR
from train_extended1 import read_annotations

NAMES = ("baseline", "trim_y", "pad_y", "aspect_085", "aspect_115", "major_cluster")


def major_cluster(image: Image.Image) -> Image.Image:
    """Reject distant tiny fragments by horizontal ink mass, keeping accents.

    Only the horizontal bounds change: every vertical pixel and accent within
    the main cluster survives. Core components are joined across normal word
    gaps; a crop is applied only if separated groups exist.
    """
    gray = np.asarray(image.convert("L"))
    h, w = gray.shape
    count, _, stats, _ = cv2.connectedComponentsWithStats(np.uint8(gray < 190))
    cores = sorted([tuple(map(int, row)) for row in stats[1:]
                    if row[4] >= 20 and row[3] >= max(10, h*.14)], key=lambda row: row[0])
    groups = []
    for x, y, cw, ch, area in cores:
        if not groups or x-groups[-1][1] > h*2.5:
            groups.append([x, x+cw, area])
        else:
            groups[-1][1] = max(groups[-1][1], x+cw)
            groups[-1][2] += area
    if len(groups) <= 1:
        return image.copy()
    left, right, _ = max(groups, key=lambda row: row[2])
    pad = max(12, round(h*.35))
    return image.crop((max(0, left-pad), 0, min(w, right+pad), h))


def variants(image: Image.Image) -> dict[str, Image.Image]:
    image = image.convert("RGB")
    gray = np.asarray(image.convert("L"))
    ys, xs = np.where(gray < 190)
    tight = image.copy()
    if len(ys):
        pad = max(2, round((int(ys.max())-int(ys.min())+1)*.04))
        tight = image.crop((0, max(0, int(ys.min())-pad), image.width, min(image.height, int(ys.max())+pad+1)))
    return {"baseline": image,
            "trim_y": tight,
            "pad_y": ImageOps.expand(image, border=(0, round(image.height*.15), 0, round(image.height*.15)), fill="white"),
            "aspect_085": image.resize((max(1, round(image.width*.85)), image.height), Image.Resampling.LANCZOS),
            "aspect_115": image.resize((max(1, round(image.width*1.15)), image.height), Image.Resampling.LANCZOS),
            "major_cluster": major_cluster(image)}


def prepare():
    source = BASE_DIR / "prepared" / "extended1_v2"
    target = BASE_DIR / "prepared" / "v4_name_geometry"
    target.mkdir(parents=True, exist_ok=True)
    rows = []
    for path, expected in read_annotations(source, "valid_annotation.txt"):
        if path.stem != "ho_ten":
            continue
        with Image.open(path) as image:
            images = variants(image)
        outputs = {}
        for name, image in images.items():
            output = target / name / path.parent.name / "ho_ten.png"
            output.parent.mkdir(parents=True, exist_ok=True)
            image.save(output)
            outputs[name] = str(output)
        rows.append({"form": path.parent.name, "expected": expected, "variants": outputs})
    font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 16)
    sheet = Image.new("RGB", (2400, 7*160), "#eeeeee")
    draw = ImageDraw.Draw(sheet)
    for index, record in enumerate(rows):
        for col, name in enumerate(NAMES):
            x, y = col*400, index*160
            draw.text((x+4, y+3), f"{record['form']} / {name}", fill="black", font=font)
            with Image.open(record["variants"][name]) as image:
                image.thumbnail((390, 120))
                sheet.paste(image, (x+4, y+30))
    sheet.save(target / "review.png")
    for number in (31, 32, 33):
        image = Image.open(source / str(number) / "ho_ten.png").convert("RGB")
        cropped = major_cluster(image)
        cropped.save(target / f"train_{number}_major_cluster.png")
    return rows


def evaluate(records):
    import torch
    from field_inference import predict_field
    from ocr_review_app import get_field_predictor
    predictor = get_field_predictor(str(BASE_DIR / "models" / "extended1_bundle.json"), "ho_ten")
    torch.set_num_threads(2)
    results = {}
    for name in NAMES:
        rows = []
        for record in records:
            with Image.open(record["variants"][name]) as image:
                detail = predict_field(predictor, image, "ho_ten")
            rows.append({"form": record["form"], "expected": record["expected"], "predicted": detail["raw"],
                         "exact": detail["raw"] == record["expected"], "distance": text_distance(record["expected"], detail["raw"])})
        results[name] = {"exact": sum(row["exact"] for row in rows), "total": len(rows),
                         "cer": sum(row["distance"] for row in rows)/sum(len(row["expected"]) for row in rows), "rows": rows}
        print(name, {key: results[name][key] for key in ("exact", "total", "cer")}, flush=True)
    (BASE_DIR / "models" / "v4_name_geometry_valid.json").write_text(json.dumps({"split": "valid", "results": results,
        "note": "Fixed geometric variants only. No test read, no active crop changes."}, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluate", action="store_true")
    args = parser.parse_args()
    records = prepare()
    if args.evaluate:
        evaluate(records)
