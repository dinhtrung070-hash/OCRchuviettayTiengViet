"""Select locally adapted field checkpoints using matched validation and replay.

The historical test set is never read here. External initializations alone are
not eligible: candidates must come from this project's recorded training runs.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from handwriting_preprocess import PREPROCESS_VERSION

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"
DATA = ROOT / "prepared" / "extended1_v2"


def metrics(rows):
    return {"exact": sum(row["exact"] for row in rows), "total": len(rows),
            "cer": sum(row["distance"] for row in rows) / max(1, sum(len(row["expected"]) for row in rows))}


def tensor_digest(path):
    # torch.save includes the archive filename; equal tensors can have different
    # file checksums. Share a predictor only after comparing all tensor contents.
    import torch
    digest = hashlib.sha256()
    state = torch.load(path, map_location="cpu", weights_only=True)
    for key, tensor in sorted(state.items()):
        digest.update(key.encode("utf-8"))
        digest.update(str((tensor.dtype, tuple(tensor.shape))).encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def main():
    old_path = MODELS / "extended1_v3_previous_bundle.json"
    old = json.loads(old_path.read_text(encoding="utf-8"))
    reference = json.loads((MODELS / "extended1_v3_final_valid.json").read_text(encoding="utf-8"))
    if hashlib.sha256(old_path.read_bytes()).hexdigest() != reference["weights_sha256"]:
        raise ValueError("Validation reference is for a different v3 bundle")
    bundle = {"format": old["format"], "name": "VietOCR Extended 1 v4",
              "base": old["base"], "fields": dict(old["fields"]),
              "preprocess_version": PREPROCESS_VERSION, "form_decoder": "greedy", "form_structured": True,
              "note": "Local adaptation from attributed external handwriting initializations. Validation selects fields; reused test is exploratory."}
    decisions = {}
    for field in ("ho_ten", "ngay_sinh", "dan_toc", "ton_giao"):
        ref = [row for row in reference["fine_tuned"]["rows"] if row["field"] == field]
        best = {"exact": sum(r["exact"] for r in ref), "total": len(ref),
                "cer": sum(r["edit_distance"] for r in ref) / sum(len(r["expected"]) for r in ref)}
        baseline_exact = best["exact"]
        expected = {Path(row["form"]).stem: row["expected"] for row in ref}
        decision = {"baseline": dict(best), "selected": dict(best), "source": "v3"}
        for report_path in sorted(MODELS.glob("extended1_v4_*adapt.json")):
            report = json.loads(report_path.read_text(encoding="utf-8"))
            options = []
            if report.get("selected_epoch"):
                options.append((report_path.with_suffix(".pth"), report["selected_epoch"]))
            specific = report.get("field_checkpoints", {}).get(field)
            if specific:
                options.append((MODELS / specific["weights"], specific["epoch"]))
            for checkpoint, number in options:
                epoch = next(item for item in report["history"] if item["epoch"] == number)
                rows = [row for row in epoch["valid"]["rows"] if Path(row["image"]).stem == field]
                actual = {Path(row["image"]).parent.name: row["expected"] for row in rows}
                if actual != expected or len(rows) != len(ref):
                    raise ValueError("Candidate has different validation labels/forms")
                if not all(Path(row["image"]).read_bytes() == (DATA / Path(row["image"]).parent.name / f"{field}.png").read_bytes() for row in rows):
                    raise ValueError("Candidate has different validation crop pixels")
                replay = [row for row in epoch["replay"]["rows"] if Path(row["image"]).stem == field]
                if len(replay) != 2 or not all(row["exact"] for row in replay):
                    continue
                score = metrics(rows)
                if score["exact"] >= baseline_exact and (score["cer"], -score["exact"]) < (best["cer"], -best["exact"]):
                    best = score
                    bundle["fields"][field] = {"weights": checkpoint.name,
                        "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest()}
                    decision = {"baseline": decision["baseline"], "selected": dict(score),
                                "source": checkpoint.name, "epoch": number, "training_report": report_path.name,
                                "initial_weights": Path(report["initial_weights"]).name, "replay_exact": len(replay)}
        decisions[field] = decision
    bundle["validation_decisions"] = decisions
    canonical = {}
    fingerprints = {}
    for field, entry in bundle["fields"].items():
        filename = entry["weights"]
        if filename not in fingerprints:
            fingerprints[filename] = tensor_digest(MODELS / filename)
        fingerprint = fingerprints[filename]
        if fingerprint in canonical:
            bundle["fields"][field] = dict(canonical[fingerprint])
            decisions[field]["routed_weights"] = canonical[fingerprint]["weights"]
        else:
            canonical[fingerprint] = dict(entry)
    initializations = {item["initial_weights"] for item in decisions.values() if "initial_weights" in item}
    bundle["initializations"] = {}
    for path in MODELS.glob("external_*_provenance.json"):
        source = json.loads(path.read_text(encoding="utf-8"))
        if source["local_weights"] in initializations:
            bundle["initializations"][source["local_weights"]] = {key: source[key] for key in
                ("repo", "revision", "filename", "sha256", "license_from_model_card", "model_card_url")}
    output = MODELS / "extended1_v4_bundle_candidate.json"
    output.write_text(json.dumps(bundle, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(decisions, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
