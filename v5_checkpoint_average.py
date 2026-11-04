"""Validation/replay-only averaging of two checkpoints in one training lineage."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import torch

from finetune_v2 import cached_rows, evaluate, brief
from model_runtime import build_model
from ocr_review_app import BASE_DIR, get_model_config
from train_extended1 import read_annotations, read_regression


def main():
    torch.set_num_threads(2)
    directory = BASE_DIR / "prepared" / "extended1_v2"
    best_path = BASE_DIR / "models" / "extended1_v4_handwriting_adapt_ho_ten.pth"
    earlier_path = BASE_DIR / "models" / "extended1_v4_handwriting_adapt.pth"
    output = BASE_DIR / "models" / "extended1_v5_name_average.pth"
    left = torch.load(best_path, map_location="cpu", weights_only=True)
    right = torch.load(earlier_path, map_location="cpu", weights_only=True)
    if left.keys() != right.keys() or not all(torch.equal(value, right[key]) for key, value in left.items() if key.startswith("cnn.")):
        raise ValueError("Expected same architecture and frozen CNN within the vudang training run")
    model, vocab = build_model(get_model_config(str(best_path)))
    model.load_state_dict(left)
    valid = [(p, label) for p, label in read_annotations(directory, "valid_annotation.txt") if p.stem == "ho_ten"]
    replay = [(p, label) for p, label in read_regression(directory / "pilot") if p.stem == "ho_ten"]
    valid_features = cached_rows(model, valid, 1, 0)
    replay_features = cached_rows(model, replay, 1, 0)
    baseline = evaluate(model, vocab, valid_features)
    regression = evaluate(model, vocab, replay_features)
    report = {"split": "valid", "test_accessed": False, "field": "ho_ten",
              "preprocess": "stroke_preserving_v2", "baseline_valid": baseline, "baseline_replay": regression,
              "sources": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (best_path, earlier_path)},
              "formula": "(1-alpha)*name_epoch14 + alpha*global_epoch8; equal frozen CNN",
              "trials": [], "selected_alpha": None}
    best = baseline["cer"], -baseline["exact"]
    for alpha in (.25, .5, .75):
        state = {key: torch.lerp(value, right[key], alpha) if value.is_floating_point() else value for key, value in left.items()}
        model.load_state_dict(state)
        del state
        result = evaluate(model, vocab, valid_features)
        replay_result = evaluate(model, vocab, replay_features)
        score = result["cer"], -result["exact"]
        accepted = score < best and result["exact"] >= baseline["exact"] and replay_result["exact"] >= regression["exact"]
        if accepted:
            temporary = output.with_suffix(".pth.tmp")
            torch.save(model.state_dict(), temporary)
            os.replace(temporary, output)
            best = score
            report["selected_alpha"] = alpha
        report["trials"].append({"alpha": alpha, "valid": result, "replay": replay_result, "accepted": accepted})
        output.with_suffix(".json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(alpha, brief(result), "replay", brief(replay_result), "selected", accepted, flush=True)


if __name__ == "__main__":
    main()
