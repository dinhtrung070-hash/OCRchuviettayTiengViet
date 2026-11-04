"""Validate labels and split by writer (or provisional form group) for VietOCR.

Fill writer_id and label in prepared/to_label.csv, then run:
    python Demo/export_vietocr.py --data-dir Demo/prepared
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).parent / "prepared")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reuse-split", type=Path,
                        help="Preserve previous valid/test groups; add newly labeled groups only to train")
    parser.add_argument("--skip-unlabeled", action="store_true",
                        help="Omit crops with an empty transcription")
    parser.add_argument("--allow-provisional-form-groups", action="store_true",
                        help="Split by form ID when real writer IDs are unavailable; this is not a writer-independent test")
    args = parser.parse_args()
    data_dir = args.data_dir.resolve()
    with (data_dir / "to_label.csv").open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit("No crops found. Run prepare_dataset.py first.")

    errors = []
    by_writer = defaultdict(list)
    skipped = 0
    for index, row in enumerate(rows, start=2):
        writer = row.get("writer_id", "").strip()
        label = row.get("label", "").strip()
        if args.skip_unlabeled and not label:
            skipped += 1
            continue
        relative = Path(row["image"])
        if not writer:
            errors.append(f"row {index}: missing writer_id")
        if not label or "\t" in label or "\n" in label:
            errors.append(f"row {index}: missing/invalid label")
        if relative.is_absolute() or ".." in relative.parts or not (data_dir / relative).is_file():
            errors.append(f"row {index}: missing/unsafe image path {relative}")
        if writer:
            if writer.startswith("form_") and not args.allow_provisional_form_groups:
                errors.append(f"row {index}: provisional form ID needs --allow-provisional-form-groups")
            by_writer[writer].append((row["image"].replace("\\", "/"), label))
    if errors:
        raise SystemExit("Fix to_label.csv:\n" + "\n".join(errors[:20]))
    if len(by_writer) < 3:
        raise SystemExit("Need at least 3 distinct writer_id values for train/valid/test split.")
    print(f"Labeled crops: {sum(map(len, by_writer.values()))}; skipped unlabeled: {skipped}")
    if args.allow_provisional_form_groups:
        print("WARNING: provisional form groups; writer independence is unknown")

    writers = sorted(by_writer)
    if args.reuse_split:
        original = json.loads(args.reuse_split.read_text(encoding="utf-8"))
        writer_splits = {key: list(original["groups"][key]) for key in ("valid", "test", "train")}
        assigned = [person for people in writer_splits.values() for person in people]
        if len(set(assigned)) != len(assigned) or not set(assigned) <= set(writers):
            raise SystemExit("Reused split contains duplicate or missing groups")
        writer_splits["train"].extend(person for person in writers if person not in assigned)
    else:
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
        unit = "form groups" if args.allow_provisional_form_groups else "writers"
        print(f"{split}: {len(people)} {unit}, {len(lines)} crops")
    (data_dir / "split_manifest.json").write_text(json.dumps({
        "seed": args.seed,
        "group_type": "provisional_form" if args.allow_provisional_form_groups else "writer",
        "skipped_unlabeled_crops": skipped,
        "groups": writer_splits,
    }, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
