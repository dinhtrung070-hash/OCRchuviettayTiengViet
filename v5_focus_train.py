"""Prepare train-only focused name augmentations in an isolated experiment.

Original training images remain present. Validation pixels/labels are copied
unchanged. Focused images add views of existing training writers, not new data.
Run finetune_v2.py separately with --extra-train focus_annotation.txt.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import cv2

from handwriting_preprocess import clean_name_line
from prepare_dataset import FIELDS, SCALE, read_image, write_image
from train_extended1 import read_annotations

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "prepared" / "extended1_v2"
TARGET = ROOT / "prepared" / "extended1_v5_focus"


def main():
    if not TARGET.resolve().is_relative_to((ROOT / "prepared").resolve()):
        raise ValueError("Experiment target must remain inside prepared")
    TARGET.mkdir(parents=True, exist_ok=True)
    rows = {}
    for split in ("train", "valid"):
        rows[split] = read_annotations(SOURCE, f"{split}_annotation.txt")
        for path, _ in rows[split]:
            output = TARGET / path.relative_to(SOURCE.resolve())
            output.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, output)
        shutil.copyfile(SOURCE / f"{split}_annotation.txt", TARGET / f"{split}_annotation.txt")
    if {path for path, _ in rows["train"]} & {path for path, _ in rows["valid"]}:
        raise ValueError("Train/valid overlap")
    shutil.copytree(SOURCE / "pilot", TARGET / "pilot", dirs_exist_ok=True)
    for name in ("preprocess.json", "split_manifest.json"):
        shutil.copyfile(SOURCE / name, TARGET / name)
    blank = read_image(ROOT / "phieu_mau.png")
    blank = cv2.resize(blank, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_CUBIC)
    a, b, c, d = [value*SCALE for value in FIELDS["ho_ten"]]
    records = []
    for path, label in rows["train"]:
        if path.stem != "ho_ten":
            continue
        raw_path = path.with_name("ho_ten_raw.png")
        picture = clean_name_line(read_image(raw_path), blank[b:d, a:c])
        output = TARGET / "focus_train" / path.parent.name / "ho_ten.png"
        output.parent.mkdir(parents=True, exist_ok=True)
        write_image(output, picture)
        records.append({"image": output.relative_to(TARGET).as_posix(), "label": label,
                        "source": path.relative_to(SOURCE.resolve()).as_posix(),
                        "source_raw_sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
                        "source_split": "train"})
    (TARGET / "focus_annotation.txt").write_text("".join(f"{row['image']}\t{row['label']}\n" for row in records), encoding="utf-8")
    valid_identical = all(path.read_bytes() == (TARGET / path.relative_to(SOURCE.resolve())).read_bytes()
                          for path, _ in rows["valid"])
    if not valid_identical:
        raise ValueError("Validation pixels changed")
    audit = {"records": records, "original_train_crops": len(rows["train"]),
             "focused_train_augmentations": len(records), "validation_crops": len(rows["valid"]),
             "valid_copied_byte_identically": valid_identical,
             "note": "Train-only focused views; not additional real forms or writers. No test samples used for training or validation selection."}
    (TARGET / "focus_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in audit.items() if key != "records"}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
