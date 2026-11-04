"""Render train-only date augmentations with installed handwriting-style fonts.

No validation/test labels are read. Font files are not distributed with the repo.
"""

from __future__ import annotations

import datetime
import json
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageFilter


def main() -> None:
    root = Path(__file__).resolve().parent / "prepared" / "extended1_v3"
    fonts = [Path("C:/Windows/Fonts") / name for name in ("segoepr.ttf", "segoeprb.ttf", "segoesc.ttf", "comic.ttf", "Inkfree.ttf", "mvboli.ttf")]
    fonts = [path for path in fonts if path.is_file()]
    if not fonts:
        raise SystemExit("No supported handwriting-style fonts found")
    rng = random.Random(20261105)
    records = []
    start = datetime.date(1920, 1, 1)
    days = (datetime.date(2020, 12, 31)-start).days
    for index in range(120):
        date = start + datetime.timedelta(days=rng.randrange(days))
        text = date.strftime("%d/%m/%Y")
        font_path = rng.choice(fonts)
        font = ImageFont.truetype(str(font_path), rng.randint(66, 88))
        box = font.getbbox(text)
        canvas = Image.new("RGB", (box[2]-box[0]+28, box[3]-box[1]+30), "white")
        draw = ImageDraw.Draw(canvas)
        draw.text((14-box[0], 12-box[1]), text, font=font, fill=(rng.randint(5, 45),)*3)
        if rng.random() < .45:
            y = int(canvas.height*.75)
            for x in range(5, canvas.width-5, rng.randint(9, 14)):
                draw.ellipse((x-1, y-1, x+1, y+1), fill=(90, 90, 90))
        canvas = canvas.resize((round(canvas.width*rng.uniform(.92, 1.12)), canvas.height), Image.Resampling.BICUBIC)
        canvas = canvas.rotate(rng.uniform(-1.2, 1.2), Image.Resampling.BICUBIC, expand=True, fillcolor="white")
        if rng.random() < .25:
            canvas = canvas.filter(ImageFilter.GaussianBlur(.3))
        output = root / "synthetic_dates" / f"{index:03d}" / "ngay_sinh.png"
        output.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(output)
        records.append({"image": output.relative_to(root).as_posix(), "label": text, "font": font_path.name})
    (root / "synthetic_date_annotation.txt").write_text("".join(f"{r['image']}\t{r['label']}\n" for r in records), encoding="utf-8")
    (root / "synthetic_date_audit.json").write_text(json.dumps({"records": records, "note": "Procedural train augmentation, not additional real handwriting or new writers. No evaluation labels used."}, ensure_ascii=False, indent=2), encoding="utf-8")
    label_font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 17)
    for page in range(2):
        sheet = Image.new("RGB", (1440, 960), "#eeeeee")
        draw = ImageDraw.Draw(sheet)
        for i, row in enumerate(records[page*24:page*24+24]):
            x, y = i%4*360, i//4*160
            draw.text((x+6, y+3), f"{row['label']} / {row['font']}", font=label_font, fill="black")
            image = Image.open(root / row["image"]).convert("RGB")
            image.thumbnail((345, 118))
            sheet.paste(image, (x+6, y+30))
        sheet.save(root / f"synthetic_date_review_{page+1}.png")
    print("Created", len(records), "synthetic training dates")


if __name__ == "__main__":
    main()
