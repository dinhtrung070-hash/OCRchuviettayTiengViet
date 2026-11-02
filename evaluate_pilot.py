"""Compare pretrained and pilot weights on the *training* forms and simulated ink.

These are smoke-test numbers, never held-out accuracy estimates.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from vietocr.tool.predictor import Predictor

from ocr_review_app import BASE_DIR, get_model_config
from pilot_finetune import labels_from_csv, training_examples


def evaluate(weights: Path | None, examples: list) -> list[dict]:
    config = get_model_config(str(weights.resolve()) if weights else None)
    predictor = Predictor(config)
    rows = []
    for image, label, form, field, ink in examples:
        prediction = " ".join(predictor.predict(image).split())
        rows.append({
            "form": form,
            "field": field,
            "ink": ink,
            "expected": label,
            "predicted": prediction,
            "exact": prediction == label,
        })
        print(f"{ink:15} {form:20} {field:12} {ascii(prediction)} -> {ascii(label)}", flush=True)
    return rows


def summary(rows: list[dict]) -> dict:
    return {
        ink: {"correct": sum(r["exact"] for r in rows if r["ink"] == ink),
              "total": sum(r["ink"] == ink for r in rows)}
        for ink in ("blue", "synthetic_black")
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=BASE_DIR / "models" / "pilot_finetuned.pth")
    parser.add_argument("--output", type=Path, default=BASE_DIR / "models" / "pilot_evaluation.json")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if not args.weights.is_file():
        parser.error(f"Missing fine-tuned weights: {args.weights}")
    torch.set_num_threads(args.threads)
    labels = labels_from_csv(BASE_DIR / "pilot_labels.csv")
    examples, _ = training_examples(labels, args.output.parent / "pilot_generated")
    print("Pretrained baseline:", flush=True)
    baseline = evaluate(None, examples)
    print("Pilot fine-tuned:", flush=True)
    tuned = evaluate(args.weights, examples)
    report = {
        "warning": "All 16 images use the same two source forms as training. Black images are synthetic recolorings. These numbers do not measure performance on new forms, writers or real black pens.",
        "baseline": {"summary": summary(baseline), "rows": baseline},
        "fine_tuned": {"summary": summary(tuned), "rows": tuned},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"baseline": report["baseline"]["summary"], "fine_tuned": report["fine_tuned"]["summary"]}), flush=True)
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
