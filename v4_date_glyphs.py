"""Build reviewable TRAIN-only date glyphs and synthetic date-line augmentation.

No valid/test annotations are read. Upper-body whitespace is used only to
propose boundaries; original full-height pixels are retained in each glyph.
Only lines with ten separated upper-body groups are proposed, and a contact
sheet must be visually checked before using the supplemental annotation.
"""

from __future__ import annotations

import argparse
import datetime
import json
import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from handwriting_preprocess import clean_handwriting
from prepare_dataset import FIELDS, SCALE, read_image, write_image

ROOT = Path(__file__).resolve().parent


def propose_spans(raw: np.ndarray) -> tuple[list[tuple[int, int]], int] | None:
    gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    # Fractions correspond to upper character body, above the print baseline.
    # Trying these does not alter full-height crop pixels or use label text.
    for proportion in (.50, .535, .565):
        cutoff = round(height*proportion)
        foreground = np.uint8(gray < 150)
        foreground[cutoff:] = 0
        count, ids, boxes, _ = cv2.connectedComponentsWithStats(foreground)
        core = np.zeros_like(gray, dtype=bool)
        for i in range(1, count):
            if boxes[i, cv2.CC_STAT_HEIGHT] >= height*.12 and boxes[i, cv2.CC_STAT_AREA] >= 20:
                core[ids == i] = True
        columns = core.any(axis=0)
        runs = []
        start = None
        for x in range(width+1):
            occupied = columns[x] if x < width else False
            if occupied and start is None:
                start = x
            elif not occupied and start is not None:
                runs.append((start, x))
                start = None
        if len(runs) != 10:
            continue
        if any(b-a < 4 for a, b in runs) or any(runs[i+1][0]-runs[i][1] < 1 for i in range(9)):
            continue
        boundaries = [0] + [(runs[i][1]+runs[i+1][0])//2 for i in range(9)] + [width]
        return list(zip(boundaries[:-1], boundaries[1:])), cutoff
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--exclude-forms", nargs="*", default=[])
    parser.add_argument("--exclude-glyphs", nargs="*", default=[], help="IDs such as 46_6 rejected after visual review")
    args = parser.parse_args()
    if args.count < 1:
        parser.error("count must be positive")
    source = ROOT / "prepared" / "extended1_v2"
    target = source / "glyph_augmentation"
    target.mkdir(parents=True, exist_ok=True)
    annotations = [line.split("\t", 1) for line in (source / "train_annotation.txt").read_text(encoding="utf-8").splitlines()]
    train_groups = {Path(rel).parts[0] for rel, _ in annotations}
    blank = read_image(ROOT / "phieu_mau.png")
    blank = cv2.resize(blank, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_CUBIC)
    a, b, c, d = [v*SCALE for v in FIELDS["ngay_sinh"]]
    blank = blank[b:d, a:c]
    font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 19)
    glyphs = []
    candidates = []
    skipped = []
    original_labels = {label for rel, label in annotations if Path(rel).stem == "ngay_sinh"}
    for relative, label in annotations:
        path = source / relative
        if path.stem != "ngay_sinh":
            continue
        if len(label) != 10 or label[2] != "/" or label[5] != "/":
            skipped.append({"form": path.parent.name, "reason": "not ten characters DD/MM/YYYY"})
            continue
        raw = read_image(path.with_name("ngay_sinh_raw.png"))
        proposed = propose_spans(raw)
        if proposed is None:
            skipped.append({"form": path.parent.name, "reason": "not ten unambiguous upper-body groups"})
            continue
        spans, cutoff = proposed
        candidates.append({"form": path.parent.name, "label": label, "cutoff": cutoff, "spans": spans})
        if path.parent.name in args.exclude_forms:
            continue
        for index, ((left, right), character) in enumerate(zip(spans, label)):
            identifier = f"{path.parent.name}_{index}"
            if identifier in args.exclude_glyphs:
                continue
            cleaned, ink = clean_handwriting(raw[:, left:right], blank[:, left:right], mode="dots")
            output = target / "glyphs" / f"{identifier}.png"
            output.parent.mkdir(parents=True, exist_ok=True)
            write_image(output, cleaned)
            glyphs.append({"id": identifier, "character": character, "image": output.relative_to(source).as_posix(),
                           "source": relative, "source_form": path.parent.name, "source_label": label,
                           "cutoff_for_detection_only": cutoff, "left": left, "right": right, "ink": ink})
    pools = {}
    for record in glyphs:
        pools.setdefault(record["character"], []).append(record)
    if "/" not in pools:
        raise SystemExit("No usable separators; review candidate segmentation first")
    sheet = Image.new("RGB", (1500, max(180, ((len(glyphs)+9)//10)*170)), "#eeeeee")
    draw = ImageDraw.Draw(sheet)
    for i, record in enumerate(glyphs):
        x, y = i%10*150, i//10*170
        draw.text((x+5, y+4), f"{record['id']}: {record['character']}", font=font, fill="black")
        image = Image.open(source / record["image"]).convert("RGB")
        image.thumbnail((140, 120))
        sheet.paste(image, (x+5, y+35))
    sheet.save(target / "glyph_review.png")
    rng = random.Random(20261104)
    records = []
    seen = set(original_labels)
    first = datetime.date(1930, 1, 1)
    span = (datetime.date(2020, 12, 31)-first).days
    for attempt in range(30000):
        label = (first+datetime.timedelta(days=rng.randrange(span))).strftime("%d/%m/%Y")
        if label in seen or any(char not in pools for char in label):
            continue
        seen.add(label)
        chosen = [rng.choice(pools[character]) for character in label]
        height = rng.randint(73, 90)
        images = []
        for record in chosen:
            original = Image.open(source / record["image"]).convert("RGB")
            image = original.resize((max(7, round(original.width*height/original.height)), height), Image.Resampling.BICUBIC)
            images.append(image)
        gaps = [rng.randint(0, 3) for _ in range(9)]
        canvas = Image.new("RGB", (sum(image.width for image in images)+sum(gaps)+12, height+12), "white")
        x = 6
        for i, image in enumerate(images):
            canvas.paste(image, (x, 6))
            x += image.width + (gaps[i] if i < 9 else 0)
        index = len(records)
        output = target / "synthetic_dates" / f"{index:03d}" / "ngay_sinh.png"
        output.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(output)
        records.append({"image": output.relative_to(source).as_posix(), "label": label,
                        "source_glyph_ids": [r["id"] for r in chosen],
                        "source_training_forms": sorted({r["source_form"] for r in chosen})})
        if len(records) == args.count:
            break
    for page in range((len(records)+23)//24):
        sheet = Image.new("RGB", (1600, 960), "#eeeeee")
        draw = ImageDraw.Draw(sheet)
        for i, record in enumerate(records[page*24:page*24+24]):
            x, y = i%4*400, i//4*160
            draw.text((x+5, y+3), record["label"], font=font, fill="black")
            image = Image.open(source / record["image"]).convert("RGB")
            image.thumbnail((390, 120))
            sheet.paste(image, (x+5, y+31))
        sheet.save(target / f"synthetic_review_{page+1}.png")
    annotation = source / "glyph_synthetic_annotation.txt"
    annotation.write_text("".join(f"{r['image']}\t{r['label']}\n" for r in records), encoding="utf-8")
    audit = {"train_groups": sorted(train_groups), "read_annotation_files": ["train_annotation.txt"],
             "excluded_forms": args.exclude_forms, "excluded_glyphs": args.exclude_glyphs,
             "characters": sorted(pools), "candidate_segmentations": candidates, "skipped": skipped,
             "glyphs": glyphs, "records": records,
             "note": "Visual review required before training. Synthetic lines are TRAIN augmentations, not new real writers or test forms. Segmentation proposals can be rejected with --exclude-glyphs."}
    (target / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"candidate_forms": [r["form"] for r in candidates], "glyph_count": len(glyphs),
                      "characters": sorted(pools), "synthetic_count": len(records),
                      "annotation": str(annotation), "review": str(target / "glyph_review.png")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
