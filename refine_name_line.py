"""Experimental name-line bounds: reject distant fragments from the next row."""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import cv2
from PIL import Image, ImageDraw, ImageFont

from handwriting_preprocess import PREPROCESS_VERSION, clean_name_line
from prepare_dataset import FIELDS, SCALE, read_image, write_image

BASE_DIR = Path(__file__).resolve().parent


def main() -> None:
    source = BASE_DIR / "prepared" / "extended1_v2"
    target = BASE_DIR / "prepared" / "extended1_v3"
    target.mkdir(parents=True, exist_ok=True)
    blank = read_image(BASE_DIR / "phieu_mau.png")
    blank = cv2.resize(blank, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_CUBIC)
    a, b, c, d = [x*SCALE for x in FIELDS["ho_ten"]]
    with (source / "to_label.csv").open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        original = source / row["image"]
        output = target / row["image"]
        output.parent.mkdir(parents=True, exist_ok=True)
        if row["field"] == "ho_ten":
            raw = read_image(original.with_name("ho_ten_raw.png"))
            write_image(output, clean_name_line(raw, blank[b:d, a:c]))
        else:
            shutil.copyfile(original, output)
        for suffix in (f"{row['field']}_raw.png", "aligned.png", "debug_fields.png"):
            extra = original.parent / suffix
            if extra.is_file():
                shutil.copyfile(extra, output.parent / suffix)
    for filename in ("train_annotation.txt", "valid_annotation.txt", "test_annotation.txt", "to_label.csv", "split_manifest.json"):
        shutil.copyfile(source / filename, target / filename)
    shutil.copytree(source / "pilot", target / "pilot", dirs_exist_ok=True)
    (target / "preprocess.json").write_text(json.dumps({"version": "stroke_preserving_v3_experimental"}, indent=2), encoding="utf-8")
    font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 17)
    for offset in range(0, 50, 10):
        canvas = Image.new("RGB", (1400, 1300), "#eeeeee")
        draw = ImageDraw.Draw(canvas)
        for index, number in enumerate(range(offset+1, offset+11)):
            y = index*130
            for column, root in enumerate((source, target)):
                image = Image.open(root / str(number) / "ho_ten.png").convert("RGB")
                image.thumbnail((680, 95))
                x = column*700
                draw.text((x+5, y+3), f"{number} / {'v2' if column == 0 else 'v3'}", font=font, fill="black")
                canvas.paste(image, (x+5, y+28))
        canvas.save(target / f"name_bounds_review_{offset//10+1}.png")
    print("Created", target)


if __name__ == "__main__":
    main()
