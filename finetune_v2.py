"""Fine-tune VietOCR with intact strokes, train-only augmentation and early stopping.

Frozen CNN prefixes are cached; optional final blocks can be fine-tuned while
batch normalization stays frozen. This keeps CPU runs practical.
The old demo forms are explicitly replay training data, not independent validation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter
from vietocr.tool.translate import process_input
from model_runtime import build_model
from handwriting_preprocess import PREPROCESS_VERSION

from ocr_review_app import BASE_DIR, PRETRAINED_WEIGHTS, get_model_config
from train_extended1 import edit_distance, read_annotations, read_regression


def augment(image: Image.Image, rng: random.Random) -> Image.Image:
    image = image.convert("RGB")
    image = image.resize((max(32, int(image.width * rng.uniform(.90, 1.12))), image.height), Image.Resampling.BICUBIC)
    image = image.rotate(rng.uniform(-1.2, 1.2), Image.Resampling.BICUBIC, expand=True, fillcolor="white")
    image = ImageEnhance.Contrast(image).enhance(rng.uniform(.8, 1.2))
    if rng.random() < .5:
        # Dots are added only to training views. The underlying word is intact.
        overlay = Image.new("RGB", image.size, "white")
        draw = ImageDraw.Draw(overlay)
        y = int(image.height * rng.uniform(.62, .78))
        radius = max(1, round(image.height * rng.uniform(.008, .017)))
        gap = max(7, radius * rng.randint(5, 8))
        tone = rng.randint(95, 180)
        for x in range(rng.randint(0, gap), image.width, gap):
            draw.ellipse((x-radius, y-radius, x+radius, y+radius), fill=(tone, tone, tone))
        image = Image.fromarray(np.minimum(np.asarray(image), np.asarray(overlay)))
    if rng.random() < .3:
        image = image.filter(ImageFilter.GaussianBlur(rng.uniform(.15, .45)))
    return image


def source_fingerprint(rows, weights: Path, views: int, seed: int) -> str:
    digest = hashlib.sha256(f"features-v2.1|{views}|{seed}".encode())
    digest.update(weights.read_bytes())
    for path, label in rows:
        digest.update(str(path.resolve()).encode())
        digest.update(label.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def cached_rows(model, rows, views: int, seed: int, prefix_only: bool = False) -> list[dict]:
    records = []
    model.eval()
    with torch.no_grad():
        for index, (path, label) in enumerate(rows):
            features = []
            with Image.open(path) as original:
                original = original.convert("RGB")
                for view in range(views):
                    image = original if view == 0 else augment(original, random.Random(seed + index*101 + view))
                    tensor = process_input(image, 32, 32, 512)
                    encoded = model.cnn.model.features[:model.form_cnn_cut](tensor) if prefix_only else model.cnn(tensor)
                    features.append(encoded.detach().clone())
            records.append({"path": str(path), "label": label, "features": features})
    return records


def finish_features(model, features: torch.Tensor) -> torch.Tensor:
    if features.ndim == 4:
        conv = model.cnn.model.features[getattr(model, "form_cnn_cut", 40):](features)
        conv = model.cnn.model.last_conv_1x1(conv)
        return conv.transpose(-1, -2).flatten(2).permute(-1, 0, 1)
    return features


def forward_loss(model, vocab, record: dict, view: int = 0, smoothing: float = 0) -> torch.Tensor:
    ids = vocab.encode(record["label"])
    source = torch.tensor(ids[:-1], dtype=torch.long).unsqueeze(1)
    target = torch.tensor(ids[1:], dtype=torch.long)
    mask = torch.zeros((1, len(ids)-1))
    logits = model.transformer(finish_features(model, record["features"][view]), source, tgt_key_padding_mask=mask)
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), target, label_smoothing=smoothing)


def greedy(model, vocab, features: torch.Tensor, maximum: int = 48) -> str:
    memory = model.transformer.forward_encoder(finish_features(model, features))
    tokens = [vocab.go]
    for _ in range(maximum):
        logits, _ = model.transformer.forward_decoder(torch.tensor(tokens).unsqueeze(1), memory)
        token = int(logits[0, -1].argmax())
        tokens.append(token)
        if token == vocab.eos:
            break
    return " ".join(vocab.decode(tokens).split())


def evaluate(model, vocab, records: list[dict]) -> dict:
    model.eval()
    rows = []
    with torch.no_grad():
        for record in records:
            text = greedy(model, vocab, record["features"][0])
            rows.append({"image": record["path"], "expected": record["label"], "predicted": text,
                         "exact": text == record["label"], "distance": edit_distance(text, record["label"])})
    return {"exact": sum(row["exact"] for row in rows), "total": len(rows),
            "cer": sum(row["distance"] for row in rows) / max(1, sum(len(row["expected"]) for row in rows)),
            "rows": rows}


def brief(result: dict) -> dict:
    return {key: result[key] for key in ("exact", "total", "cer")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=BASE_DIR / "prepared" / "extended1_v2")
    parser.add_argument("--output", type=Path, default=BASE_DIR / "models" / "extended1_v2.pth")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--stop-after", type=int,
                        help="Stop after this many epochs while preserving the planned LR schedule")
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--views", type=int, default=3)
    parser.add_argument("--initial-weights", type=Path)
    parser.add_argument("--cnn-last-block", action="store_true",
                        help="Also update final VGG convolution block; earlier blocks and batchnorm stay frozen")
    parser.add_argument("--cnn-lr-ratio", type=float, default=.1)
    parser.add_argument("--name-repeat", type=int, default=1,
                        help="Oversample name crops to emphasize variable names over repeated short fields")
    parser.add_argument("--date-repeat", type=int, default=1)
    parser.add_argument("--cnn-cut", type=int, choices=[14, 27, 40], default=40,
                        help="First trainable VGG feature layer when cnn-last-block is enabled")
    parser.add_argument("--smoothing", type=float, default=.05)
    parser.add_argument("--clean-probability", type=float, default=0,
                        help="Probability of choosing the clean view first; remaining draws sample all views")
    parser.add_argument("--fields", nargs="+", choices=["ho_ten", "ngay_sinh", "dan_toc", "ton_giao"])
    parser.add_argument("--extra-train", help="Optional extra annotation file inside data-dir, from training forms only")
    parser.add_argument("--whole-line-repeat", type=int, default=1,
                        help="Repeat original line crops relative to supplemental word crops")
    args = parser.parse_args()
    if min(args.epochs, args.patience, args.threads, args.views, args.name_repeat, args.date_repeat, args.whole_line_repeat) < 1 or args.lr <= 0 or args.cnn_lr_ratio <= 0:
        parser.error("Training settings must be positive")
    if not 0 <= args.smoothing < 1 or not 0 <= args.clean_probability <= 1:
        parser.error("smoothing and clean-probability must be in their probability ranges")
    if args.stop_after is not None and args.stop_after < 1:
        parser.error("stop-after must be positive")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    train = read_annotations(args.data_dir, "train_annotation.txt")
    if args.extra_train:
        train += read_annotations(args.data_dir, args.extra_train)
    valid = read_annotations(args.data_dir, "valid_annotation.txt")
    replay = read_regression(args.data_dir / "pilot")
    if args.fields:
        train = [(p, label) for p, label in train if p.stem in args.fields]
        valid = [(p, label) for p, label in valid if p.stem in args.fields]
        replay = [(p, label) for p, label in replay if p.stem in args.fields]
    if {p for p, _ in train + replay} & {p for p, _ in valid}:
        raise ValueError("Train/valid overlap")
    config = get_model_config(None)
    if not PRETRAINED_WEIGHTS.is_file():
        from vietocr.tool.utils import download_weights
        downloaded = download_weights(config["weights"])
        PRETRAINED_WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(downloaded, PRETRAINED_WEIGHTS)
    model, vocab = build_model(config)
    model.load_state_dict(torch.load(PRETRAINED_WEIGHTS, map_location="cpu", weights_only=True))
    if args.initial_weights:
        model.load_state_dict(torch.load(args.initial_weights, map_location="cpu", weights_only=True))
    model.form_cnn_cut = args.cnn_cut
    cache_suffix = (f"prefix{args.cnn_cut}" if args.cnn_last_block else "full") + ("_" + "_".join(args.fields) if args.fields else "")
    cache_path = args.data_dir / f"cnn_features_v3_{cache_suffix}.pt"
    fingerprint = source_fingerprint(train + valid + replay, args.initial_weights or PRETRAINED_WEIGHTS, args.views, args.seed) + cache_suffix
    cache = torch.load(cache_path, weights_only=False) if cache_path.is_file() else {}
    if cache.get("fingerprint") != fingerprint:
        print("Caching train-only augmented CNN features ...", flush=True)
        cache = {"fingerprint": fingerprint,
                 "train": cached_rows(model, train, args.views, args.seed, args.cnn_last_block),
                 "valid": cached_rows(model, valid, 1, args.seed, args.cnn_last_block),
                 "replay": cached_rows(model, replay, args.views, args.seed, args.cnn_last_block)}
        torch.save(cache, cache_path)
    for parameter in model.cnn.parameters():
        parameter.requires_grad = False
    parameters = list(model.transformer.parameters())
    groups = [{"params": parameters, "lr_ratio": 1.0}]
    if args.cnn_last_block:
        cnn_parameters = []
        for module in list(model.cnn.model.features[args.cnn_cut:]) + [model.cnn.model.last_conv_1x1]:
            if isinstance(module, torch.nn.Conv2d):
                for parameter in module.parameters():
                    parameter.requires_grad = True
                    cnn_parameters.append(parameter)
        groups.append({"params": cnn_parameters, "lr_ratio": args.cnn_lr_ratio})
        parameters = parameters + cnn_parameters
    optimizer = torch.optim.AdamW(groups, lr=args.lr, betas=(.9, .98), eps=1e-9, weight_decay=1e-4)
    initial_valid = evaluate(model, vocab, cache["valid"])
    initial_replay = evaluate(model, vocab, cache["replay"])
    profile = args.data_dir / "preprocess.json"
    version = json.loads(profile.read_text(encoding="utf-8"))["version"] if profile.is_file() else PREPROCESS_VERSION
    report = {"data_dir": str(args.data_dir), "preprocess": version, "cnn_frozen": not args.cnn_last_block,
              "cnn_last_block_lr_ratio": args.cnn_lr_ratio if args.cnn_last_block else None,
              "name_repeat": args.name_repeat,
              "date_repeat": args.date_repeat, "fields": args.fields,
              "extra_train": args.extra_train, "whole_line_repeat": args.whole_line_repeat,
              "smoothing": args.smoothing, "clean_probability": args.clean_probability,
              "cnn_cut": args.cnn_cut if args.cnn_last_block else None,
              "planned_epochs": args.epochs, "stop_after": args.stop_after,
              "optimizer_reset_from_initial_checkpoint": bool(args.initial_weights),
              "initial_weights": str(args.initial_weights or PRETRAINED_WEIGHTS),
              "train_images": len(train), "valid_images": len(valid), "replay_images": len(replay),
              "augmentation_views_per_training_image": args.views, "learning_rate": args.lr,
              "seed": args.seed, "initial_valid": initial_valid, "initial_replay": initial_replay,
              "selection": "Lowest validation CER, then highest exact; old demo replay exact cannot drop.",
              "history": [], "selected_epoch": None,
              "limits": "Labels read from images; unknown real writer IDs; reused test is exploratory. Replay forms are training data."}
    print("baseline", brief(initial_valid), "replay", brief(initial_replay), flush=True)
    best = (initial_valid["cer"], -initial_valid["exact"])
    stale = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(min(args.epochs, args.stop_after or args.epochs)):
        start = time.monotonic()
        model.transformer.train()
        model.cnn.eval()
        # Repeat old forms to retain recognizer behavior on the original demo.
        sequence = cache["train"] + cache["replay"] * 2
        sequence += [row for row in cache["train"] if Path(row["path"]).stem == "ho_ten"] * (args.name_repeat-1)
        sequence += [row for row in cache["train"] if Path(row["path"]).stem == "ngay_sinh"] * (args.date_repeat-1)
        sequence += [row for row in cache["train"] if not {"word_crops", "synthetic_names", "synthetic_dates"} & set(Path(row["path"]).parts)] * (args.whole_line_repeat-1)
        rng = random.Random(args.seed + epoch)
        rng.shuffle(sequence)
        lr = args.lr * min(1.0, (epoch + 1) / 2) * (.3 + .7 * .5 * (1 + math.cos(math.pi * epoch / args.epochs)))
        for group in optimizer.param_groups:
            group["lr"] = lr * group["lr_ratio"]
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for index, record in enumerate(sequence):
            view = 0 if rng.random() < args.clean_probability else rng.randrange(len(record["features"]))
            loss = forward_loss(model, vocab, record, view, smoothing=args.smoothing)
            group_size = min(8, len(sequence) - (index // 8) * 8)
            (loss / group_size).backward()
            losses.append(float(loss.detach()))
            if (index+1) % 8 == 0 or index+1 == len(sequence):
                torch.nn.utils.clip_grad_norm_(parameters, 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        validation = evaluate(model, vocab, cache["valid"])
        regression = evaluate(model, vocab, cache["replay"])
        score = (validation["cer"], -validation["exact"])
        accepted = score < best and regression["exact"] >= initial_replay["exact"]
        item = {"epoch": epoch+1, "learning_rate": lr, "train_loss": sum(losses)/len(losses),
                "valid": validation, "replay": regression, "accepted": accepted,
                "seconds": round(time.monotonic()-start, 1)}
        report["history"].append(item)
        if accepted:
            temporary = args.output.with_suffix(".pth.tmp")
            torch.save(model.state_dict(), temporary)
            os.replace(temporary, args.output)
            best = score
            stale = 0
            report["selected_epoch"] = epoch+1
        else:
            stale += 1
        report_path = args.output.with_suffix(".json")
        temporary_report = report_path.with_suffix(".json.tmp")
        temporary_report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temporary_report, report_path)
        print(f"epoch {epoch+1}: loss={item['train_loss']:.3f}, valid={brief(validation)}, replay={brief(regression)}, selected={accepted}, seconds={item['seconds']}", flush=True)
        if stale >= args.patience:
            print("Early stop: validation did not improve.", flush=True)
            break
    print("selected_epoch", report["selected_epoch"], "weights", args.output, flush=True)


if __name__ == "__main__":
    main()
