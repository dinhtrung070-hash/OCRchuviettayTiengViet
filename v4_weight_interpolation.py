"""Probe original/V2 weight interpolation for names, using validation only.

Interpolating weights retains the common initialization and tests whether less
domain adaptation can recover language knowledge. No training or test images
are read. A candidate is saved only if validation and replay gates pass.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os

import torch

from finetune_v2 import cached_rows, evaluate, brief
from model_runtime import build_model
from ocr_review_app import BASE_DIR, PRETRAINED_WEIGHTS, get_model_config
from train_extended1 import read_annotations, read_regression


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ratios", nargs="+", type=float, default=[.25, .5, .75])
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.threads < 1 or any(not 0 < ratio < 1 for ratio in args.ratios):
        parser.error("Threads must be positive and ratios in (0, 1)")
    torch.set_num_threads(args.threads)
    torch.manual_seed(20261104)
    directory = BASE_DIR / "prepared" / "extended1_v2"
    tuned_path = BASE_DIR / "models" / "extended1_v2_continued.pth"
    output = BASE_DIR / "models" / "extended1_v4_name_interpolation.pth"
    valid = [(p, label) for p, label in read_annotations(directory, "valid_annotation.txt") if p.stem == "ho_ten"]
    replay = [(p, label) for p, label in read_regression(directory / "pilot") if p.stem == "ho_ten"]
    original = torch.load(PRETRAINED_WEIGHTS, map_location="cpu", weights_only=True)
    tuned = torch.load(tuned_path, map_location="cpu", weights_only=True)
    if original.keys() != tuned.keys():
        raise ValueError("Architecture differs")
    config = get_model_config(str(tuned_path))
    model, vocab = build_model(config)
    model.load_state_dict(tuned)
    baseline_valid_records = cached_rows(model, valid, 1, 0)
    baseline_replay_records = cached_rows(model, replay, 1, 0)
    baseline_valid = evaluate(model, vocab, baseline_valid_records)
    baseline_replay = evaluate(model, vocab, baseline_replay_records)
    cnn_identical = all(torch.equal(value, tuned[key]) for key, value in original.items() if key.startswith("cnn."))
    report = {"preprocess": "stroke_preserving_v2", "field": "ho_ten", "test_accessed": False,
              "formula": "(1-ratio)*original + ratio*v2, floating tensors; integer buffers retain v2",
              "sources": {"original_sha256": hashlib.sha256(PRETRAINED_WEIGHTS.read_bytes()).hexdigest(),
                          "v2_sha256": hashlib.sha256(tuned_path.read_bytes()).hexdigest()},
              "cnn_identical": cnn_identical,
              "baseline_valid": baseline_valid, "baseline_replay": baseline_replay,
              "trials": [], "selected_v2_fraction": None,
              "selection": "Strictly lower validation CER/exact tuple, exact cannot decrease, replay exact cannot decrease"}
    best = baseline_valid["cer"], -baseline_valid["exact"]
    print("baseline", brief(baseline_valid), "replay", brief(baseline_replay), "cnn identical", cnn_identical, flush=True)
    for ratio in args.ratios:
        mixed = {key: torch.lerp(value, tuned[key], ratio) if value.is_floating_point() else tuned[key]
                 for key, value in original.items()}
        model.load_state_dict(mixed)
        del mixed
        gc.collect()
        records = baseline_valid_records if cnn_identical else cached_rows(model, valid, 1, 0)
        replay_records = baseline_replay_records if cnn_identical else cached_rows(model, replay, 1, 0)
        result = evaluate(model, vocab, records)
        regression = evaluate(model, vocab, replay_records)
        score = result["cer"], -result["exact"]
        accepted = score < best and result["exact"] >= baseline_valid["exact"] and regression["exact"] >= baseline_replay["exact"]
        if accepted:
            temporary = output.with_suffix(".pth.tmp")
            torch.save(model.state_dict(), temporary)
            os.replace(temporary, output)
            report["selected_v2_fraction"] = ratio
            best = score
        report["trials"].append({"v2_fraction": ratio, "valid": result, "replay": regression, "accepted": accepted})
        output.with_suffix(".json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print("ratio", ratio, "valid", brief(result), "replay", brief(regression), "accepted", accepted, flush=True)
        del records, replay_records
        gc.collect()
    print("selected", report["selected_v2_fraction"], flush=True)


if __name__ == "__main__":
    main()
