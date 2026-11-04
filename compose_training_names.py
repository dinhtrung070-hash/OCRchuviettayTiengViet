"""Compose line augmentations exclusively from reviewed training word crops."""

from __future__ import annotations

import json
import argparse
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=180)
    args = parser.parse_args()
    base = Path(__file__).resolve().parent
    source = base / "prepared" / "extended1_v2"
    target = base / "prepared" / "extended1_v3"
    audit = json.loads((source / "word_crop_audit.json").read_text(encoding="utf-8"))
    labels = {line.split("\t")[0]: line.split("\t")[1] for line in (source / "train_annotation.txt").read_text(encoding="utf-8").splitlines()}
    target_groups = {Path(line.split("\t")[0]).parts[0] for line in (target / "train_annotation.txt").read_text(encoding="utf-8").splitlines()}
    pools = {"first": [], "middle": [], "last": []}
    for record in audit["records"]:
        assert Path(record["source"]).parts[0] in target_groups
        word_count = len(labels[record["source"]].split())
        index = int(record["id"].rsplit("_", 1)[1])
        role = "first" if index == 0 else "last" if index == word_count-1 else "middle"
        pools[role].append(record)
    rng = random.Random(20261104)
    records = []
    seen = set(labels.values())
    for attempt in range(2000):
        chosen = [rng.choice(pools["first"]), rng.choice(pools["middle"]), rng.choice(pools["last"])]
        if rng.random() < .15:
            chosen.insert(2, rng.choice(pools["middle"]))
        label = " ".join(row["label"] for row in chosen)
        if label in seen:
            continue
        seen.add(label)
        images = []
        for row in chosen:
            image = Image.open(source / row["image"]).convert("RGB")
            height = rng.randint(87, 110)
            image = image.resize((max(32, round(image.width*height/image.height)), height), Image.Resampling.BICUBIC)
            mask = np.asarray(image.convert("L")) < 170
            ys, _ = np.where(mask)
            baseline = int(np.quantile(ys, .85)) if len(ys) else height-15
            images.append((image, 110-baseline))
        gaps = [rng.randint(16, 28) for _ in chosen[:-1]]
        canvas = Image.new("RGB", (24+sum(image.width for image, _ in images)+sum(gaps), max(150, max(y+image.height for image, y in images)+12)), "white")
        x = 12
        for index, (image, y) in enumerate(images):
            canvas.paste(image, (x, max(0, y)))
            x += image.width + (gaps[index] if index < len(gaps) else 0)
        index = len(records)
        path = target / "synthetic_names" / f"{index:03d}" / "ho_ten.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(path)
        records.append({"image": path.relative_to(target).as_posix(), "label": label,
                        "source_word_ids": [row["id"] for row in chosen],
                        "source_training_forms": [Path(row["source"]).parts[0] for row in chosen]})
        if len(records) >= args.count:
            break
    (target / "synthetic_name_annotation.txt").write_text("".join(f"{r['image']}\t{r['label']}\n" for r in records), encoding="utf-8")
    (target / "synthetic_name_audit.json").write_text(json.dumps({"records": records, "note": "Train-only image compositions. These are not additional real forms or writers."}, ensure_ascii=False, indent=2), encoding="utf-8")
    font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 17)
    for page in range(3):
        sheet = Image.new("RGB", (1400, 1080), "#eeeeee")
        draw = ImageDraw.Draw(sheet)
        for i, row in enumerate(records[page*16:page*16+16]):
            x, y = (i%2)*700, (i//2)*135
            draw.text((x+7, y+4), row["label"], font=font, fill="black")
            image = Image.open(target / row["image"]).convert("RGB")
            image.thumbnail((680, 100))
            sheet.paste(image, (x+7, y+29))
        sheet.save(target / f"synthetic_name_review_{page+1}.png")
    print("Created", len(records), "synthetic training lines from reviewed training words only.")


if __name__ == "__main__":
    main()
