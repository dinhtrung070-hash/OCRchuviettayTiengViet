"""Promote the checkpoint selected by validation to the local demo filename."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

from handwriting_preprocess import PREPROCESS_VERSION


def main() -> None:
    root = Path(__file__).resolve().parent
    candidates = []
    for path in (root / "models").glob("extended1_v2*.json"):
        report = json.loads(path.read_text(encoding="utf-8"))
        if report.get("selected_epoch") and path.with_suffix(".pth").is_file():
            selected = next(row for row in report["history"] if row["epoch"] == report["selected_epoch"])
            candidates.append((selected["valid"]["cer"], -selected["valid"]["exact"], path, selected))
    if not candidates:
        raise SystemExit("No fine-tuned checkpoint passed validation")
    _, _, report_path, selected = min(candidates, key=lambda row: row[:2])
    source = report_path.with_suffix(".pth")
    fingerprint = hashlib.sha256(source.read_bytes()).hexdigest()
    target = root / "models" / "extended1_selected.pth"
    temporary = target.with_suffix(".tmp")
    shutil.copyfile(source, temporary)
    os.replace(temporary, target)
    metadata = {"name": "VietOCR Extended 1 v2", "source_checkpoint": source.name,
                "selected_epoch": selected["epoch"], "sha256": fingerprint,
                "preprocess_version": PREPROCESS_VERSION, "form_decoder": "greedy", "form_structured": True,
                "validation": {key: selected["valid"][key] for key in ("exact", "total", "cer")},
                "replay_exact": selected["replay"]["exact"],
                "note": "Selected by validation before test; reused test is exploratory and writer IDs are unknown."}
    for split in ("valid", "test"):
        path = root / "models" / f"extended1_v2_{split}_structured.json"
        if path.is_file():
            report = json.loads(path.read_text(encoding="utf-8"))
            if report["weights_sha256"] == fingerprint:
                metadata[split + "_comparison"] = {
                    name: {key: report[name][key] for key in ("correct", "total", "cer", "formatted_correct", "structured_correct")}
                    for name in ("baseline", "fine_tuned")}
    target.with_suffix(".json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
