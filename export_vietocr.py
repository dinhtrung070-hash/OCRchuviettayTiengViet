"""Validate labels and split by writer for VietOCR fine-tuning.

Fill writer_id and label in prepared/to_label.csv, then run:
    python Demo/export_vietocr.py --data-dir Demo/prepared
"""

from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).parent / "prepared")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    data_dir = args.data_dir.resolve()
    with (data_dir / "to_label.csv").open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit("No crops found. Run prepare_dataset.py first.")

    errors = []
    by_writer = defaultdict(list)
    for index, row in enumerate(rows, start=2):
        writer = row.get("writer_id", "").strip()
        label = row.get("label", "").strip()
        relative = Path(row["image"])
        if not writer:
            errors.append(f"row {index}: missing writer_id")
        if not label or "\t" in label or "\n" in label:
            errors.append(f"row {index}: missing/invalid label")
        if relative.is_absolute() or ".." in relative.parts or not (data_dir / relative).is_file():
            errors.append(f"row {index}: missing/unsafe image path {relative}")
        if writer:
            by_writer[writer].append((row["image"].replace("\\", "/"), label))
    if errors:
        raise SystemExit("Fix to_label.csv:\n" + "\n".join(errors[:20]))
    if len(by_writer) < 3:
        raise SystemExit("Need at least 3 distinct writer_id values for train/valid/test split.")

    writers = sorted(by_writer)
    random.Random(args.seed).shuffle(writers)
    count = len(writers)
    valid_count = max(1, round(count * 0.15))
    test_count = max(1, round(count * 0.15))
    if valid_count + test_count >= count:
        valid_count = test_count = 1
    writer_splits = {
        "valid": writers[:valid_count],
        "test": writers[valid_count:valid_count + test_count],
        "train": writers[valid_count + test_count:],
    }
    for split, people in writer_splits.items():
        lines = [f"{image}\t{label}" for person in people for image, label in by_writer[person]]
        (data_dir / f"{split}_annotation.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"{split}: {len(people)} writers, {len(lines)} crops")


if __name__ == "__main__":
    main()
