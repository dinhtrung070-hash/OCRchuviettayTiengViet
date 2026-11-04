"""Choose field experts by matched validation rows, never by test scores."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from handwriting_preprocess import PREPROCESS_VERSION


def summary(rows: list[dict]) -> dict:
    return {"exact": sum(row["exact"] for row in rows), "total": len(rows),
            "cer": sum(row.get("edit_distance", row.get("distance", 0)) for row in rows) / sum(len(row["expected"]) for row in rows)}


def entry(path: Path) -> dict:
    return {"weights": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "models" / "extended1_bundle_candidate.json")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    base = root / "models" / "extended1_v2_continued.pth"
    reference_report = json.loads((root / "models" / "extended1_v2_valid_structured.json").read_text(encoding="utf-8"))
    reference = reference_report["fine_tuned"]
    base_entry = entry(base)
    old_metadata = json.loads((root / "models" / "extended1_selected.json").read_text(encoding="utf-8"))
    if base_entry["sha256"] != old_metadata["sha256"] or base_entry["sha256"] != reference_report["weights_sha256"]:
        raise ValueError("Validation reference is for a different baseline checkpoint")
    fields = sorted({row["field"] for row in reference["rows"]})
    selected = {}
    decisions = {}
    for field in fields:
        baseline = [row for row in reference["rows"] if row["field"] == field]
        expected = {row["form"]: row["expected"] for row in baseline}
        best = summary(baseline)
        decision = {"baseline": best, "selected": best, "source": base.name}
        for path in (root / "models").glob("extended1_v3*.json"):
            report = json.loads(path.read_text(encoding="utf-8"))
            if not report.get("selected_epoch") or field not in (report.get("fields") or []):
                continue
            checkpoint = path.with_suffix(".pth")
            if not checkpoint.is_file():
                continue
            epoch = next(row for row in report["history"] if row["epoch"] == report["selected_epoch"])
            rows = [row for row in epoch["valid"]["rows"] if Path(row["image"]).stem == field]
            actual = {Path(row["image"]).parent.name + ".jpg": row["expected"] for row in rows}
            if actual != expected:
                raise ValueError(f"Expert validation cohort differs: {path.name}")
            same_pixels = all(Path(row["image"]).read_bytes() ==
                              (root / "prepared" / "extended1_v2" / Path(row["image"]).parent.name / f"{field}.png").read_bytes()
                              for row in rows)
            if not same_pixels:
                continue
            candidate = summary(rows)
            if candidate["exact"] >= best["exact"] and (candidate["cer"], -candidate["exact"]) < (best["cer"], -best["exact"]):
                best = candidate
                selected[field] = entry(checkpoint)
                decision = {"baseline": summary(baseline), "selected": candidate,
                            "source": checkpoint.name, "training_report": path.name,
                            "epoch": epoch["epoch"]}
        decisions[field] = decision
    data = {"format": "vietocr_field_bundle_v1", "name": "VietOCR Extended 1 v3",
            "base": base_entry, "fields": selected, "preprocess_version": PREPROCESS_VERSION,
            "form_decoder": "greedy", "form_structured": True, "validation_decisions": decisions,
            "note": "Experts selected only by matched validation fields. Test is reused/exploratory; writer IDs unknown."}
    args.output.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(data, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
