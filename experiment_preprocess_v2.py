"""Compare conservative dot removal on validation crops, without reading test labels."""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import cv2
import torch
from PIL import Image, ImageDraw, ImageFont
from vietocr.tool.predictor import Predictor

from handwriting_preprocess import clean_handwriting
from prepare_dataset import FIELDS, LEGACY_FIELDS, SCALE, read_image, write_image
from ocr_review_app import BASE_DIR, get_model_config
from train_extended1 import edit_distance


def main() -> None:
    torch.set_num_threads(4)
    old_root = BASE_DIR / "prepared" / "dataset_20261103_component"
    output = BASE_DIR / "prepared" / "stroke_v2"
    output.mkdir(parents=True, exist_ok=True)
    blank = read_image(BASE_DIR / "phieu_mau.png")
    blank = cv2.resize(blank, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_CUBIC)
    with (old_root / "to_label.csv").open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    with (old_root / "pilot" / "to_label.csv").open(newline="", encoding="utf-8-sig") as handle:
        pilot_rows = list(csv.DictReader(handle))
    manifests = {mode: [] for mode in ("raw", "dots", "chroma")}
    raw_for_sheet = {}
    aligned_cache = {}
    for row in rows + pilot_rows:
        pilot = row["form"].endswith(".png")
        source_root = old_root / "pilot" if pilot else old_root
        stem = Path(row["form"]).stem
        if stem not in aligned_cache:
            aligned_cache[stem] = read_image(source_root / stem / "aligned.png")
        aligned = aligned_cache[stem]
        box = (LEGACY_FIELDS if pilot else FIELDS)[row["field"]]
        a, b, c, d = [v * SCALE for v in box]
        raw = aligned[b:d, a:c]
        raw_for_sheet[(row["form"], row["field"])] = Image.fromarray(cv2.cvtColor(raw, cv2.COLOR_BGR2RGB))
        for mode in manifests:
            root = output / mode
            dest = root / stem / f"{row['field']}.png"
            dest.parent.mkdir(parents=True, exist_ok=True)
            cleaned, ink = clean_handwriting(raw, None if pilot else blank[b:d, a:c], mode)
            write_image(dest, cleaned)
            record = dict(row)
            record["image"] = f"{stem}/{row['field']}.png"
            record["blue_ink_detected"] = str(ink == "blue")
            manifests[mode].append(record)
    for mode, manifest in manifests.items():
        root = output / mode
        with (root / "to_label.csv").open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(manifest[0]))
            writer.writeheader()
            writer.writerows(manifest)
        for split in ("train", "valid", "test"):
            shutil.copyfile(old_root / f"{split}_annotation.txt", root / f"{split}_annotation.txt")
        shutil.copyfile(old_root / "split_manifest.json", root / "split_manifest.json")
        pilot_lines = [f"{row['image']}\t{row['label']}" for row in manifest if row["form"].endswith(".png")]
        (root / "replay_annotation.txt").write_text("\n".join(pilot_lines) + "\n", encoding="utf-8")
    valid_paths = {line.split("\t")[0] for line in (old_root / "valid_annotation.txt").read_text(encoding="utf-8").splitlines()}
    predictor = Predictor(get_model_config(None))
    results = []
    for mode, manifest in manifests.items():
        for row in manifest:
            pilot = row["form"].endswith(".png")
            if not pilot and row["image"] not in valid_paths:
                continue
            with Image.open(output / mode / row["image"]) as image:
                text, probability = predictor.predict(image, return_prob=True)
            expected = row["label"]
            text = " ".join(text.split())
            results.append({"mode": mode, "split": "pilot" if pilot else "valid",
                            "form": row["form"], "field": row["field"],
                            "ink": "blue" if row["blue_ink_detected"] == "True" else "dark",
                            "expected": expected, "predicted": text, "exact": text == expected,
                            "distance": edit_distance(text, expected), "probability": float(probability)})
        print(f"finished {mode}", flush=True)
    summary = {}
    for mode in manifests:
        summary[mode] = {}
        for split in ("valid", "pilot"):
            subset = [row for row in results if row["mode"] == mode and row["split"] == split]
            summary[mode][split] = {"exact": sum(row["exact"] for row in subset), "total": len(subset),
                                    "cer": sum(row["distance"] for row in subset) / sum(len(row["expected"]) for row in subset)}
    (output / "preprocess_report.json").write_text(json.dumps({"summary": summary, "rows": results}, ensure_ascii=False, indent=2), encoding="utf-8")
    font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 18)
    for form in ("1.jpg", "20.jpg", "23.jpg", "39.jpg", "phieudaviet1.png"):
        sheet = Image.new("RGB", (1600, 620), "#eeeeee")
        draw = ImageDraw.Draw(sheet)
        for column, mode in enumerate(("original", "raw", "dots", "chroma")):
            draw.text((column * 400 + 8, 6), f"{form} / {mode}", font=font, fill="black")
            for index, field in enumerate(FIELDS):
                im = (raw_for_sheet[(form, field)].copy() if mode == "original" else
                      Image.open(output / mode / Path(form).stem / f"{field}.png").convert("RGB"))
                im.thumbnail((380, 105))
                x, y = column * 400 + 8, 35 + index * 140
                draw.text((x, y), field, font=font, fill="black")
                sheet.paste(im, (x, y+24))
        sheet.save(output / f"audit_{Path(form).stem}.png")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
