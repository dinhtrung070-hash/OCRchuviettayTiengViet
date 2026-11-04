"""Deploy a validation-selected local bundle only after regression checks pass."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parent / "models"
    candidate = root / "extended1_bundle_candidate.json"
    raw = candidate.read_bytes()
    checksum = hashlib.sha256(raw).hexdigest()
    bundle = json.loads(raw)
    summaries = {}
    for split in ("valid", "test"):
        report = json.loads((root / f"extended1_v3_final_{split}.json").read_text(encoding="utf-8"))
        if report["weights_sha256"] != checksum or report["preprocess_version"] != bundle["preprocess_version"]:
            raise ValueError("Evaluation does not match the candidate bundle/preprocessing")
        new, old = report["fine_tuned"], report["previous_v2"]
        if new["correct"] < old["correct"] or new["cer"] > old["cer"] or new["structured_correct"] < old["structured_correct"]:
            raise ValueError(f"Candidate regresses on {split}; deployment rejected")
        summaries[split] = {name: {key: report[name][key] for key in ("correct", "total", "cer", "formatted_correct", "structured_correct")}
                            for name in ("baseline", "previous_v2", "fine_tuned")}
    for entry in [bundle["base"], *bundle["fields"].values()]:
        path = (root / entry["weights"]).resolve()
        if not path.is_relative_to(root.resolve()) or hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError("Bundle checkpoint checksum changed")
    temporary = root / "extended1_bundle.json.tmp"
    temporary.write_bytes(raw)
    os.replace(temporary, root / "extended1_bundle.json")
    (root / "extended1_bundle_evaluation.json").write_text(json.dumps({"bundle_sha256": checksum,
        "selection": "Validation selects experts; reused test acts as a deployment regression check, not an independent estimate.",
        "comparisons": summaries}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
