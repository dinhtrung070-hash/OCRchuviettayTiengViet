"""Small, explicitly in-sample VietOCR fine-tuning pilot on the two demo forms.

This proves that training, black-ink simulation, inference and the UI weight
format work. It does not estimate accuracy on new writers or real black pens.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from vietocr.tool.translate import build_model, process_input

from ocr_review_app import BASE_DIR, FIELD_ORDER, PRETRAINED_WEIGHTS, get_model_config, prepare_form
from prepare_dataset import FIELDS, SCALE, read_image


def labels_from_csv(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    labels = {row["form"]: {field: row[field].strip() for field in FIELD_ORDER} for row in rows}
    if len(labels) != 2 or any(not value for item in labels.values() for value in item.values()):
        raise ValueError("pilot_labels.csv must contain two forms with all four labels")
    for name in labels:
        if not (BASE_DIR / name).is_file():
            raise FileNotFoundError(BASE_DIR / name)
    return labels


def black_ink_simulation(form_name: str, aligned: Image.Image, output: Path) -> Path:
    """Copy blue-ink pixels onto the blank template as black ink, keeping print."""
    blank = read_image(BASE_DIR / "phieu_mau.png")
    h, w = blank.shape[:2]
    blank = cv2.resize(blank, (w * SCALE, h * SCALE), interpolation=cv2.INTER_CUBIC)
    source = cv2.cvtColor(np.asarray(aligned), cv2.COLOR_RGB2BGR)
    for box in FIELDS.values():
        x1, y1, x2, y2 = [v * SCALE for v in box]
        region = source[y1:y2, x1:x2]
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([90, 45, 25]), np.array([140, 255, 255]))
        if cv2.countNonZero(mask) < 40:
            raise ValueError(f"No blue ink to simulate in {form_name}: {box}")
        # Black handwriting overlaps the printed dots as it would on paper.
        target = blank[y1:y2, x1:x2]
        target[mask > 0] = (20, 20, 20)
    output.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".png", blank)
    if not ok:
        raise ValueError("Could not encode simulated black form")
    encoded.tofile(str(output))
    return output


def training_examples(labels: dict[str, dict[str, str]], out_dir: Path):
    examples = []
    diagnostics = {}
    for form_name, fields in labels.items():
        blue = prepare_form(str(BASE_DIR / form_name))
        black_path = black_ink_simulation(form_name, blue["aligned"], out_dir / f"black_{Path(form_name).stem}.png")
        black = prepare_form(str(black_path))
        diagnostics[form_name] = {
            "blue_crop_methods": blue["methods"],
            "synthetic_black_crop_methods": black["methods"],
        }
        for field in FIELD_ORDER:
            label = fields[field]
            examples.append((blue["crops"][field], label, form_name, field, "blue"))
            examples.append((black["crops"][field], label, form_name, field, "synthetic_black"))
    return examples, diagnostics


def train(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.set_num_threads(args.threads)
    labels = labels_from_csv(args.labels)
    config = get_model_config(None)
    examples, diagnostics = training_examples(labels, args.output.parent / "pilot_generated")
    model, vocab = build_model(config)
    if not PRETRAINED_WEIGHTS.is_file():
        from vietocr.tool.utils import download_weights
        source = download_weights(config["weights"])
    else:
        source = str(PRETRAINED_WEIGHTS)
    model.load_state_dict(torch.load(source, map_location="cpu", weights_only=True))
    # With only two forms, update the final vocabulary projection using the
    # average gradient over every crop. Keep the pretrained recognizer intact.
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.transformer.fc.parameters():
        parameter.requires_grad = True
    optimizer = torch.optim.AdamW(model.transformer.fc.parameters(), lr=args.lr, weight_decay=0.0)
    losses = []
    model.eval()
    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        epoch_loss = 0.0
        for image, label, form, field, ink in examples:
            image_tensor = process_input(image, 32, 32, 512)
            ids = vocab.encode(label)
            target_input = torch.tensor(ids[:-1], dtype=torch.long).unsqueeze(1)
            target_output = torch.tensor(ids[1:], dtype=torch.long)
            padding_mask = torch.zeros((1, len(ids) - 1), dtype=torch.float32)
            logits = model(image_tensor, target_input, padding_mask)
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), target_output)
            (loss / len(examples)).backward()
            epoch_loss += float(loss.item()) / len(examples)
        torch.nn.utils.clip_grad_norm_(model.transformer.fc.parameters(), 1.0)
        optimizer.step()
        item = {"epoch": step + 1, "mean_teacher_forced_loss": round(epoch_loss, 5)}
        losses.append(item)
        print(f"epoch {step + 1}/{args.steps}: mean loss={item['mean_teacher_forced_loss']}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.output)
    report = {
        "type": "in_sample_pilot_not_generalization_score",
        "source_forms": list(labels),
        "real_ink_colors": ["blue"],
        "simulated_ink_colors": ["black"],
        "training_examples": len(examples),
        "epochs": args.steps,
        "trainable_scope": "transformer.fc only",
        "learning_rate": args.lr,
        "weights": str(args.output),
        "diagnostics": diagnostics,
        "losses": losses,
    }
    report_path = args.output.with_suffix(".json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {args.output} and {report_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=BASE_DIR / "pilot_labels.csv")
    parser.add_argument("--output", type=Path, default=BASE_DIR / "models" / "pilot_finetuned.pth")
    parser.add_argument("--steps", type=int, default=10, help="Full passes over all 16 examples")
    parser.add_argument("--lr", type=float, default=0.00005)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.steps < 1 or args.threads < 1:
        parser.error("steps and threads must be positive")
    train(args)


if __name__ == "__main__":
    main()
