"""Promote v4 only after matched v3 comparisons and live-crop replay pass.

Test is reused exploratory data, not an independent performance estimate.
The two pilot forms are replay training data and are never counted as test.
Use --check-only to run every gate and save evidence without replacing active
models/extended1_bundle.json.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FIELDS = ("ho_ten", "ngay_sinh", "dan_toc", "ton_giao")
METRICS = ("correct", "total", "cer", "formatted_correct", "structured_correct")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def cohort(rows: list[dict]) -> dict[tuple[str, str], str]:
    result = {(row["form"], row["field"]): row["expected"] for row in rows}
    if len(result) != len(rows) or not rows:
        raise ValueError("Evaluation contains duplicate or empty rows")
    return result


def assert_no_regression(new: dict, old: dict, context: str) -> None:
    if new["total"] != old["total"] or not old["total"]:
        raise ValueError(f"Different evaluation totals for {context}")
    if not all(math.isfinite(float(result[key])) for result in (new, old) for key in METRICS):
        raise ValueError(f"Nonfinite metrics for {context}")
    if (new["correct"] < old["correct"] or new["cer"] > old["cer"] + 1e-12
            or new["structured_correct"] < old["structured_correct"]):
        raise ValueError(f"Candidate regresses on {context}; deployment rejected")


def assert_same_metrics(actual: dict, reference: dict, context: str) -> None:
    for key in METRICS:
        if not math.isclose(actual[key], reference[key], rel_tol=0, abs_tol=1e-12):
            raise ValueError(f"Comparison is not the recorded v3 result: {context}/{key}")


def verify_checkpoints(models: Path, bundle: dict) -> dict[str, str]:
    if bundle.get("format") != "vietocr_field_bundle_v1" or not set(bundle.get("fields", {})) <= set(FIELDS):
        raise ValueError("Invalid field bundle format or unknown field")
    hashes = {}
    for entry in [bundle["base"], *bundle["fields"].values()]:
        relative = Path(entry["weights"])
        path = (models / relative).resolve()
        if relative.is_absolute() or not path.is_relative_to(models.resolve()) or not path.is_file():
            raise ValueError("Bundle checkpoint must exist inside the local models directory")
        if str(path) not in hashes:
            hashes[str(path)] = sha256(path)
        checksum = hashes[str(path)]
        if checksum != entry["sha256"]:
            raise ValueError(f"Checkpoint checksum changed: {relative}")
    return hashes


def verify_reports(models: Path, data: Path, bundle: dict, checksum: str, previous_checksum: str,
                   version: int = 4, previous_version: int = 3) -> dict:
    comparisons = {}
    for split in ("valid", "test"):
        report = json.loads((models / f"extended1_v{version}_final_{split}.json").read_text(encoding="utf-8"))
        reference = json.loads((models / f"extended1_v{previous_version}_final_{split}.json").read_text(encoding="utf-8"))
        annotation_checksum = sha256(data / f"{split}_annotation.txt")
        if (report["split"] != split or reference["split"] != split
                or report["weights_sha256"] != checksum
                or report.get("previous_weights_sha256") != previous_checksum
                or reference["weights_sha256"] != previous_checksum):
            raise ValueError(f"Evaluation does not match candidate/v3 snapshot for {split}")
        if not (report["annotation_sha256"] == reference["annotation_sha256"] == annotation_checksum):
            raise ValueError(f"Annotations differ from the recorded v3 cohort on {split}")
        if not (report["preprocess_version"] == reference["preprocess_version"] == bundle["preprocess_version"]):
            raise ValueError(f"Evaluation preprocessing changed on {split}")
        if report["decoder"] != bundle["form_decoder"] or report["structured_decoding"] != bundle["form_structured"]:
            raise ValueError(f"Evaluation decoding differs from the candidate on {split}")
        new, old, recorded = report["fine_tuned"], report["previous_v2"], reference["fine_tuned"]
        if not (cohort(new["rows"]) == cohort(old["rows"]) == cohort(recorded["rows"])):
            raise ValueError(f"Evaluation rows differ on {split}")
        if not (set(new["by_field"]) == set(old["by_field"]) == set(recorded["by_field"]) == set(FIELDS)):
            raise ValueError(f"Missing field comparison on {split}")
        assert_same_metrics(old, recorded, split)
        assert_no_regression(new, old, split)
        field_comparisons = {}
        for field in FIELDS:
            assert_same_metrics(old["by_field"][field], recorded["by_field"][field], f"{split}/{field}")
            assert_no_regression(new["by_field"][field], old["by_field"][field], f"{split}/{field}")
            field_comparisons[field] = {f"v{previous_version}": {key: old["by_field"][field][key] for key in METRICS},
                                        f"v{version}": {key: new["by_field"][field][key] for key in METRICS}}
        comparisons[split] = {"annotation_sha256": annotation_checksum,
                              f"v{previous_version}": {key: old[key] for key in METRICS},
                              f"v{version}": {key: new[key] for key in METRICS}, "by_field": field_comparisons}
    return comparisons


def replay_candidate(candidate: Path, bundle: dict, checksum: str) -> dict:
    # Delay torch/model imports until every cheap comparison and checksum passes.
    import torch
    from field_inference import normalize_value
    from ocr_review_app import get_field_predictor, prepare_form, recognize_detailed

    labels_path = ROOT / "pilot_labels.csv"
    with labels_path.open(newline="", encoding="utf-8-sig") as source:
        labels = list(csv.DictReader(source))
    if len(labels) != 2 or len({row["form"] for row in labels}) != 2:
        raise ValueError("Replay must contain exactly the two labeled pilot forms")
    for field in FIELDS:
        get_field_predictor(str(candidate.resolve()), field)
    torch.set_num_threads(2)
    rows = []
    for label in labels:
        source = (ROOT / label["form"]).resolve()
        if not source.is_relative_to(ROOT) or not source.is_file() or not all(label[field] for field in FIELDS):
            raise ValueError("Missing/unsafe pilot form or incomplete replay labels")
        prepared = prepare_form(str(source))
        if prepared["preprocess_version"] != bundle["preprocess_version"]:
            raise ValueError("Live replay preprocessing differs from the evaluated bundle")
        details = recognize_detailed(prepared, str(candidate.resolve()))
        for field in FIELDS:
            detail = details[field]
            entry = bundle["fields"].get(field, bundle["base"])
            if detail["model_metadata"]["sha256"] != entry["sha256"]:
                raise ValueError("Replay used a different checkpoint than the candidate")
            expected = normalize_value(field, label[field])
            rows.append({"form": label["form"], "field": field, "expected": label[field],
                         "raw": detail["raw"], "predicted": detail["value"],
                         "raw_exact": detail["raw"] == label[field],
                         "structured_exact": detail["value"].casefold() == expected.casefold(),
                         "recognition": detail})
    return {"bundle_sha256": checksum, "preprocess_version": bundle["preprocess_version"],
            "pilot_labels_sha256": sha256(labels_path), "total": len(rows),
            "raw_correct": sum(row["raw_exact"] for row in rows),
            "structured_correct": sum(row["structured_exact"] for row in rows), "rows": rows,
            "role": "Replay training data only; this is a regression check, not held-out accuracy."}


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true", help="Run every gate, including live-crop replay, but do not replace the active bundle")
    parser.add_argument("--version", type=int, default=4)
    parser.add_argument("--previous-version", type=int, default=3)
    args = parser.parse_args()
    if not 3 <= args.previous_version < args.version:
        parser.error("Require 3 <= previous-version < version")
    models, data = ROOT / "models", ROOT / "prepared" / "extended1_v2"
    candidate = models / f"extended1_v{args.version}_bundle_candidate.json"
    previous = models / f"extended1_v{args.previous_version}_previous_bundle.json"
    raw = candidate.read_bytes()
    checksum = hashlib.sha256(raw).hexdigest()
    previous_checksum = sha256(previous)
    bundle = json.loads(raw)
    if bundle.get("form_decoder") != "greedy" or bundle.get("form_structured") is not True:
        raise ValueError("Candidate must use the evaluated greedy/structured inference configuration")
    comparisons = verify_reports(models, data, bundle, checksum, previous_checksum, args.version, args.previous_version)
    checkpoints = verify_checkpoints(models, bundle)
    replay = replay_candidate(candidate, bundle, checksum)
    atomic_json(models / f"v{args.version}_replay_candidate.json", replay)
    if replay["total"] != 8 or replay["structured_correct"] != 8:
        raise ValueError("Candidate failed 8/8 structured replay fields; deployment rejected")
    # Recheck after inference, before any active-bundle replacement.
    if candidate.read_bytes() != raw or sha256(previous) != previous_checksum:
        raise ValueError("Candidate or previous snapshot changed during verification")
    if verify_checkpoints(models, bundle) != checkpoints:
        raise ValueError("Checkpoint changed during verification")
    summary = {"bundle_sha256": checksum, "previous_bundle_sha256": previous_checksum,
               "preprocess_version": bundle["preprocess_version"], "check_only": args.check_only,
               "selection": "Validation selects field experts. Reused test is a deployment regression check, not an independent estimate.",
               "comparisons": comparisons,
               "replay_training_regression": {key: replay[key] for key in ("total", "raw_correct", "structured_correct", "pilot_labels_sha256", "role")}}
    if args.check_only:
        atomic_json(models / f"extended1_v{args.version}_promotion_check.json", summary)
    else:
        temporary = models / "extended1_bundle.json.tmp"
        temporary.write_bytes(raw)
        os.replace(temporary, models / "extended1_bundle.json")
        atomic_json(models / "extended1_bundle_evaluation.json", summary)
        atomic_json(models / f"extended1_v{args.version}_bundle_evaluation.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
