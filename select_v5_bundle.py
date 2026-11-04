"""Freeze v5 from train/validation evidence; never read the historical test."""

from __future__ import annotations

import copy
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"
DATA = ROOT / "prepared" / "extended1_v2"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def field_annotations(split, field):
    rows = {}
    for line in (DATA / f"{split}_annotation.txt").read_text(encoding="utf-8").splitlines():
        image, label = line.split("\t", 1)
        if Path(image).stem == field:
            rows[Path(image).parent.name] = label
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keep-v4-ethnicity", action="store_true",
                        help="Keep the v4 ethnicity checkpoint and apply its TRAIN-calibrated category policy")
    args = parser.parse_args()
    previous = MODELS / "extended1_v4_previous_bundle.json"
    old = json.loads(previous.read_text(encoding="utf-8"))
    reference = json.loads((MODELS / "extended1_v4_final_valid.json").read_text(encoding="utf-8"))
    if reference["weights_sha256"] != digest(previous):
        raise ValueError("V4 reference and snapshot differ")
    bundle = copy.deepcopy(old)
    bundle["name"] = "VietOCR Extended 1 v5"
    bundle["previous_bundle_sha256"] = digest(previous)
    decisions = {}
    # All v5 neural trials are recorded, including rejected ones. Only strict
    # validation improvements with two correct replay fields are eligible.
    for field in ("ho_ten", "ngay_sinh", "dan_toc", "ton_giao"):
        base = reference["fine_tuned"]["by_field"][field]
        best = base["cer"], -base["correct"]
        decisions[field] = {"source": "v4", "valid": base}
        for path in sorted(MODELS.glob("extended1_v5_*.json")):
            report = json.loads(path.read_text(encoding="utf-8"))
            options = []
            if report.get("selected_epoch"):
                options.append((path.with_suffix(".pth"), report["selected_epoch"]))
            specific = report.get("field_checkpoints", {}).get(field)
            if specific:
                options.append((MODELS / specific["weights"], specific["epoch"]))
            for weights, number in options:
                epoch = next(item for item in report["history"] if item["epoch"] == number)
                rows = [row for row in epoch["valid"]["rows"] if Path(row["image"]).stem == field]
                if not rows:
                    continue
                if {Path(r["image"]).parent.name: r["expected"] for r in rows} != field_annotations("valid", field):
                    raise ValueError("Validation labels/forms differ")
                if not all(Path(r["image"]).read_bytes() == (DATA / Path(r["image"]).parent.name / f"{field}.png").read_bytes() for r in rows):
                    raise ValueError("Validation pixels differ")
                replay = [r for r in epoch["replay"]["rows"] if Path(r["image"]).stem == field]
                exact = sum(r["exact"] for r in rows)
                cer = sum(r["distance"] for r in rows) / sum(len(r["expected"]) for r in rows)
                score = cer, -exact
                if len(replay) == 2 and all(r["exact"] for r in replay) and exact >= base["correct"] and score < best:
                    bundle["fields"][field] = {"weights": weights.name, "sha256": digest(weights)}
                    best = score
                    decisions[field] = {"source": weights.name, "epoch": number, "report": path.name,
                                        "valid": {"correct": exact, "total": len(rows), "cer": cer}}
    # Prespecified tie experiment: same held-out field metrics, choose the head
    # fitting TRAIN best. This is not evidence of better generalization.
    if not args.keep_v4_ethnicity and decisions["dan_toc"]["source"] == "v4":
        heads = json.loads((MODELS / "v5_ethnicity_heads.json").read_text(encoding="utf-8"))
        eligible = []
        baseline = reference["fine_tuned"]["by_field"]["dan_toc"]
        for filename, evidence in heads["results"].items():
            weights = MODELS / filename
            if evidence["weights_sha256"] != digest(weights):
                raise ValueError("Ethnicity evidence checkpoint changed")
            for split in ("train", "valid"):
                rows = evidence["splits"][split]["rows"]
                if {Path(r["image"]).parent.name: r["expected"] for r in rows} != field_annotations(split, "dan_toc"):
                    raise ValueError("Ethnicity evidence cohort differs")
            valid, train, replay = (evidence["splits"][key] for key in ("valid", "train", "replay_train"))
            if (valid["correct"] == baseline["correct"] and abs(valid["cer"] - baseline["cer"]) < 1e-12
                and valid["structured_correct"] >= baseline["structured_correct"]
                and replay["total"] == replay["correct"] == 2):
                eligible.append(((train["cer"], -train["correct"]), filename, evidence))
        if not eligible:
            raise ValueError("No eligible tied ethnicity heads")
        _, filename, evidence = min(eligible, key=lambda item: item[0])
        bundle["base"] = {"weights": filename, "sha256": evidence["weights_sha256"]}
        bundle["fields"].pop("dan_toc", None)
        decisions["dan_toc"] = {"source": filename, "valid": evidence["splits"]["valid"],
                                "train_fit": {key: evidence["splits"]["train"][key] for key in ("correct", "total", "cer")},
                                "selection": "Validation tie; lower TRAIN CER, then higher TRAIN exact; test never read."}
        decisions["dan_toc"]["valid"] = {key: decisions["dan_toc"]["valid"][key] for key in ("correct", "total", "cer", "structured_correct")}
    policy_report = json.loads((MODELS / "v5_category_valid.json").read_text(encoding="utf-8"))
    if policy_report["weights_sha256"] != digest(previous):
        raise ValueError("Policy calibrated with a different source bundle")
    if any(policy_report["annotation_sha256"][split] != digest(DATA / f"{split}_annotation.txt") for split in ("train", "valid")):
        raise ValueError("Policy calibration labels changed")
    chosen = policy_report["selected_from_train"]
    if chosen["support"] < 4 or chosen["precision"] < .95:
        raise ValueError("Insufficient TRAIN support for the candidate policy")
    fields = ("dan_toc", "ton_giao") if args.keep_v4_ethnicity else ("ton_giao",)
    if args.keep_v4_ethnicity:
        bundle["base"] = dict(old["base"])
        if "dan_toc" in old["fields"]:
            bundle["fields"]["dan_toc"] = dict(old["fields"]["dan_toc"])
        else:
            bundle["fields"].pop("dan_toc", None)
        decisions["dan_toc"] = {"source": "v4", "valid": reference["fine_tuned"]["by_field"]["dan_toc"],
                                "selection": "Retained after the tied-head experiment failed the exploratory regression check."}
    bundle["category_policies"] = {field: {"threshold": chosen["threshold"], "odds": chosen["odds"]} for field in fields}
    for field in fields:
        if bundle["fields"].get(field, bundle["base"]) != old["fields"].get(field, old["base"]):
            raise ValueError("Recalibrate the category policy for the changed checkpoint first")
    bundle["policy_selection"] = {"report": "v5_category_valid.json", "support": chosen["support"],
                                  "note": "Small TRAIN calibration support; proposals require human review. Policies only apply to calibration checkpoint identities."}
    bundle["validation_decisions"] = decisions
    bundle["note"] = ("V5 retains all v4 neural weights and opts in to selective finite-field proposals. Reused test informed deployment rollback; it is exploratory."
                      if args.keep_v4_ethnicity else "V5 tests a validation-tied ethnicity head using train fit and selective religion proposals. Test is reused exploratory data.")
    (MODELS / "extended1_v5_bundle_candidate.json").write_text(json.dumps(bundle, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({field: {k:v for k,v in row.items() if k != "valid"} for field, row in decisions.items()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
