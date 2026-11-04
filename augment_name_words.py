"""Create syllable crops from training names only, with a visual review sheet.

Labels come from existing human-read transcriptions. Segmentation only accepts
clear whitespace gaps; contact sheets must be reviewed before using annotations.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ocr_review_app import BASE_DIR
from prepare_dataset import read_image, write_image
from handwriting_preprocess import clean_handwriting
from train_extended1 import read_annotations


def segments(image: np.ndarray, count: int) -> list[tuple[int, int]]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    n, components, stats, _ = cv2.connectedComponentsWithStats(np.uint8(gray < 175))
    mask = np.zeros_like(gray, dtype=bool)
    for i in range(1, n):
        top, h = stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_HEIGHT]
        if h >= height*.14 and stats[i, cv2.CC_STAT_AREA] >= 25 and top < height*.65 and top+h > height*.35:
            mask[components == i] = True
    columns = mask.any(axis=0)
    occupied = np.flatnonzero(columns)
    if not len(occupied):
        return []
    left, right = int(occupied[0]), int(occupied[-1])+1
    gaps = []
    start = None
    for x in range(left, right):
        if not columns[x] and start is None:
            start = x
        elif columns[x] and start is not None:
            if x-start >= max(8, height*.07):
                gaps.append((start, x))
            start = None
    gaps.sort(key=lambda pair: pair[1]-pair[0], reverse=True)
    if len(gaps) < count-1:
        return []
    selected = gaps[:count-1]
    smallest = min(b-a for a, b in selected)
    next_gap = gaps[count-1][1]-gaps[count-1][0] if len(gaps) >= count else 0
    if smallest < height*.18 and smallest < next_gap*1.4:
        return []
    boundaries = [max(0, left-10)] + sorted((a+b)//2 for a, b in selected) + [min(width, right+10)]
    spans = list(zip(boundaries[:-1], boundaries[1:]))
    if any(b-a < height*.3 for a, b in spans):
        return []
    return spans


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=BASE_DIR / "prepared" / "extended1_v2")
    parser.add_argument("--exclude", nargs="*", default=[], help="Reviewed bad crop IDs, e.g. 3_0")
    args = parser.parse_args()
    root = args.data_dir
    sources = [(path, label) for path, label in read_annotations(root, "train_annotation.txt") if path.stem == "ho_ten"]
    records = []
    skipped = []
    for path, label in sources:
        picture = read_image(path)
        words = label.split()
        spans = segments(picture, len(words))
        if len(spans) != len(words):
            skipped.append(path.parent.name)
            continue
        for index, ((left, right), word) in enumerate(zip(spans, words)):
            crop_id = f"{path.parent.name}_{index}"
            if crop_id in args.exclude:
                continue
            output = root / "word_crops" / crop_id / "ho_ten.png"
            output.parent.mkdir(parents=True, exist_ok=True)
            cleaned, _ = clean_handwriting(picture[:, left:right], mode="dots")
            write_image(output, cleaned)
            records.append({"id": crop_id, "image": output.relative_to(root).as_posix(),
                            "label": word, "source": path.relative_to(root.resolve()).as_posix(),
                            "left": left, "right": right, "split": "train"})
    (root / "word_annotation.txt").write_text("".join(f"{r['image']}\t{r['label']}\n" for r in records), encoding="utf-8")
    (root / "word_crop_audit.json").write_text(json.dumps({"records": records, "skipped_forms": skipped,
                                                         "excluded_crop_ids": args.exclude,
                                                         "warning": "Review contact sheets before training; word crops are augmentations of existing train forms."}, ensure_ascii=False, indent=2), encoding="utf-8")
    font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 18)
    for offset in range(0, len(records), 30):
        canvas = Image.new("RGB", (1250, 900), "#eeeeee")
        draw = ImageDraw.Draw(canvas)
        for index, row in enumerate(records[offset:offset+30]):
            x, y = (index % 5)*250, (index//5)*150
            draw.text((x+7, y+5), f"{row['id']}: {row['label']}", font=font, fill="black")
            image = Image.open(root / row["image"]).convert("RGB")
            image.thumbnail((235, 110))
            canvas.paste(image, (x+7, y+32))
        canvas.save(root / f"word_review_{offset//30+1}.png")
    print(f"Accepted {len(records)} word crops from {len(set(r['source'] for r in records))}/{len(sources)} training names. Skipped {skipped}")


if __name__ == "__main__":
    main()
