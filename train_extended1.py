"""Fine-tune VietOCR on grouped train/valid annotations.

Run prepare_dataset.py, label prepared/to_label.csv, then export_vietocr.py
before using this script. The two-form pilot is handled by pilot_finetune.py.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from vietocr.tool.translate import build_model, process_input, translate

from ocr_review_app import BASE_DIR, PRETRAINED_WEIGHTS, get_model_config


def read_annotations(root: Path, filename: str) -> list[tuple[Path, str]]:
    path = root / filename
    if not path.is_file():
        raise ValueError(f"Missing {path}; run export_vietocr.py after labeling")
    rows = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        try:
            relative, label = line.split("\t", 1)
        except ValueError as exc:
            raise ValueError(f"{path}:{lineno}: expected image<TAB>label") from exc
        image_path = (root / relative).resolve()
        if not image_path.is_relative_to(root.resolve()) or not image_path.is_file() or not label.strip():
            raise ValueError(f"{path}:{lineno}: missing/unsafe image or label")
        rows.append((image_path, label.strip()))
    if not rows:
        raise ValueError(f"No labeled rows in {path}")
    return rows


def tensors(image_path: Path, label: str, vocab):
    with Image.open(image_path) as image:
        picture = process_input(image, 32, 32, 512)
    ids = vocab.encode(label)
    source = torch.tensor(ids[:-1], dtype=torch.long).unsqueeze(1)
    target = torch.tensor(ids[1:], dtype=torch.long)
    mask = torch.zeros((1, len(ids) - 1), dtype=torch.float32)
    return picture, source, target, mask


def teacher_loss(model, vocab, row: tuple[Path, str]):
    image, source, target, mask = tensors(*row, vocab)
    logits = model(image, source, mask)
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), target)


def validate(model, vocab, rows: list[tuple[Path, str]]) -> dict:
    model.eval()
    exact = 0
    losses = []
    with torch.no_grad():
        for image_path, label in rows:
            image, source, target, mask = tensors(image_path, label, vocab)
            logits = model(image, source, mask)
            losses.append(float(F.cross_entropy(logits.reshape(-1, logits.shape[-1]), target).item()))
            tokens, _ = translate(image, model)
            prediction = vocab.decode(tokens[0].tolist())
            exact += prediction == label
    return {"exact": exact, "total": len(rows), "mean_loss": sum(losses) / len(losses)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=BASE_DIR / "prepared")
    parser.add_argument("--output", type=Path, default=BASE_DIR / "models" / "extended1_finetuned.pth")
    parser.add_argument("--initial-weights", type=Path,
                        help="Resume fine-tuning from a local checkpoint instead of pretrained VietOCR")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.000001)
    parser.add_argument("--scope", choices=["head", "transformer"], default="transformer")
    parser.add_argument("--accumulate", type=int, default=8)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if min(args.epochs, args.accumulate, args.threads) < 1 or args.lr <= 0:
        parser.error("epochs, accumulate, threads and lr must be positive")
    train_rows = read_annotations(args.data_dir, "train_annotation.txt")
    valid_rows = read_annotations(args.data_dir, "valid_annotation.txt")
    if {p for p, _ in train_rows} & {p for p, _ in valid_rows}:
        parser.error("Train/valid contain the same image paths")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    config = get_model_config(None)
    model, vocab = build_model(config)
    if args.initial_weights is not None:
        if not args.initial_weights.is_file():
            parser.error(f"Missing initial weights: {args.initial_weights}")
        initial = str(args.initial_weights)
    elif PRETRAINED_WEIGHTS.is_file():
        initial = str(PRETRAINED_WEIGHTS)
    else:
        from vietocr.tool.utils import download_weights
        initial = download_weights(config["weights"])
    model.load_state_dict(torch.load(initial, map_location="cpu", weights_only=True))
    for parameter in model.cnn.parameters():
        parameter.requires_grad = False
    if args.scope == "head":
        for parameter in model.transformer.parameters():
            parameter.requires_grad = False
        for parameter in model.transformer.fc.parameters():
            parameter.requires_grad = True
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.lr)
    baseline = validate(model, vocab, valid_rows)
    print(f"initial validation: {baseline}", flush=True)
    best = None
    history = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(args.epochs):
        model.train()
        model.cnn.eval()
        if args.scope == "head":
            model.transformer.eval()
        shuffled = train_rows.copy()
        random.Random(args.seed + epoch).shuffle(shuffled)
        losses = []
        optimizer.zero_grad(set_to_none=True)
        for index, row in enumerate(shuffled, 1):
            loss = teacher_loss(model, vocab, row)
            (loss / args.accumulate).backward()
            losses.append(float(loss.item()))
            if index % args.accumulate == 0 or index == len(shuffled):
                torch.nn.utils.clip_grad_norm_(parameters, 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        validation = validate(model, vocab, valid_rows)
        result = {"epoch": epoch + 1, "train_mean_loss": sum(losses) / len(losses), "valid": validation}
        history.append(result)
        print(f"epoch {epoch + 1}: {result}", flush=True)
        score = (validation["exact"], -validation["mean_loss"])
        if best is None or score > best:
            torch.save(model.state_dict(), args.output)
            best = score
    report = {
        "train_images": len(train_rows), "valid_images": len(valid_rows),
        "scope": args.scope, "learning_rate": args.lr, "initial_valid": baseline,
        "initial_weights": initial,
        "trainable_parameters": sum(p.numel() for p in parameters),
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "history": history, "best_weights": str(args.output),
        "note": "For a writer-generalization claim, group by real writer ID; provisional form groups support only exploratory evaluation.",
    }
    args.output.with_suffix(".json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved best weights: {args.output}", flush=True)


if __name__ == "__main__":
    main()
