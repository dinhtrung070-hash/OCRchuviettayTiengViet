"""Validation-only name adaptation with a frozen language Transformer.

The experiment updates the final visual block, retaining the existing language
weights. All cropped images and augmented views originate in the train split;
validation is clean and used solely for checkpoint selection. No test is read.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import torch

from finetune_v2 import cached_rows, forward_loss, evaluate, brief, source_fingerprint
from model_runtime import build_model
from ocr_review_app import BASE_DIR, get_model_config
from train_extended1 import read_annotations, read_regression


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--initial-weights", type=Path, default=BASE_DIR / "models" / "extended1_v2_continued.pth")
    parser.add_argument("--output", type=Path, default=BASE_DIR / "models" / "extended1_v4_name_visual.pth")
    args = parser.parse_args()
    if min(args.epochs, args.patience, args.threads) < 1 or args.lr <= 0:
        parser.error("Positive training settings required")
    torch.set_num_threads(args.threads)
    torch.manual_seed(20261104)
    root = BASE_DIR / "prepared" / "extended1_v2"
    train = [(p, label) for p, label in read_annotations(root, "train_annotation.txt") if p.stem == "ho_ten"]
    valid = [(p, label) for p, label in read_annotations(root, "valid_annotation.txt") if p.stem == "ho_ten"]
    replay = [(p, label) for p, label in read_regression(root / "pilot") if p.stem == "ho_ten"]
    if {p for p, _ in train + replay} & {p for p, _ in valid}:
        raise ValueError("Training/validation overlap")
    config = get_model_config(str(args.initial_weights))
    model, vocab = build_model(config)
    model.load_state_dict(torch.load(args.initial_weights, map_location="cpu", weights_only=True))
    model.form_cnn_cut = 40
    cache_path = root / "cnn_features_v4_names_visual.pt"
    fingerprint = source_fingerprint(train + valid + replay, args.initial_weights, 3, 20261104) + "cnn_only_prefix40"
    cache = torch.load(cache_path, weights_only=False) if cache_path.is_file() else {}
    if cache.get("fingerprint") != fingerprint:
        cache = {"fingerprint": fingerprint, "train": cached_rows(model, train, 3, 20261104, True),
                 "valid": cached_rows(model, valid, 1, 20261104, True),
                 "replay": cached_rows(model, replay, 3, 20261104, True)}
        torch.save(cache, cache_path)
    for parameter in model.parameters():
        parameter.requires_grad = False
    parameters = []
    for layer in list(model.cnn.model.features[40:]) + [model.cnn.model.last_conv_1x1]:
        if isinstance(layer, torch.nn.Conv2d):
            for parameter in layer.parameters():
                parameter.requires_grad = True
                parameters.append(parameter)
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=1e-4)
    initial = evaluate(model, vocab, cache["valid"])
    initial_replay = evaluate(model, vocab, cache["replay"])
    report = {"data_dir": str(root), "preprocess": "stroke_preserving_v2", "fields": ["ho_ten"],
              "cnn_frozen": False, "transformer_frozen": True, "cnn_cut": 40,
              "initial_weights": str(args.initial_weights), "learning_rate": args.lr,
              "train_images": len(train), "valid_images": len(valid), "replay_images": len(replay),
              "augmentation_views_per_training_image": 3, "smoothing": .05,
              "clean_probability": .65, "seed": 20261104,
              "initial_valid": initial, "initial_replay": initial_replay, "history": [], "selected_epoch": None,
              "selection": "Lowest clean validation CER, exact no lower than initial; replay exact cannot drop. Test never read."}
    print("initial", brief(initial), "replay", brief(initial_replay), flush=True)
    best = (initial["cer"], -initial["exact"])
    stale = 0
    for epoch in range(1, args.epochs + 1):
        started = time.monotonic()
        model.eval()
        sequence = cache["train"] + cache["replay"] * 2
        rng = random.Random(20261104 + epoch)
        rng.shuffle(sequence)
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for i, record in enumerate(sequence):
            view = 0 if rng.random() < .65 else rng.randrange(3)
            loss = forward_loss(model, vocab, record, view, smoothing=.05)
            accumulation = min(8, len(sequence) - (i // 8) * 8)
            (loss / accumulation).backward()
            losses.append(float(loss.detach()))
            if (i + 1) % 8 == 0 or i + 1 == len(sequence):
                torch.nn.utils.clip_grad_norm_(parameters, 1.)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        result = evaluate(model, vocab, cache["valid"])
        regression = evaluate(model, vocab, cache["replay"])
        score = result["cer"], -result["exact"]
        accepted = score < best and result["exact"] >= initial["exact"] and regression["exact"] >= initial_replay["exact"]
        row = {"epoch": epoch, "train_loss": sum(losses)/len(losses), "valid": result, "replay": regression,
               "accepted": accepted, "seconds": round(time.monotonic()-started, 2)}
        report["history"].append(row)
        if accepted:
            temporary = args.output.with_suffix(".pth.tmp")
            torch.save(model.state_dict(), temporary)
            os.replace(temporary, args.output)
            report["selected_epoch"] = epoch
            best, stale = score, 0
        else:
            stale += 1
        args.output.with_suffix(".json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print("epoch", epoch, "loss", round(row["train_loss"], 4), "valid", brief(result), "replay", brief(regression), "accepted", accepted, "seconds", row["seconds"], flush=True)
        if stale >= args.patience:
            report["stop_reason"] = f"Validation plateau for {stale} epochs"
            break
    args.output.with_suffix(".json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("selected", report["selected_epoch"], flush=True)


if __name__ == "__main__":
    main()
