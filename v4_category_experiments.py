"""Audit image-conditioned category likelihoods on validation only.

Generic category vocabularies already used by the app are retained. Ground
truth is used solely to summarize candidate policies, never to build choices.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch
from PIL import Image
from vietocr.tool.translate import process_input

from field_inference import predict_field, text_distance, normalize_value
from ocr_review_app import BASE_DIR, get_field_predictor
from structured_decode import string_scores, vocabularies
from train_extended1 import read_annotations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=BASE_DIR / "prepared" / "extended1_v2")
    parser.add_argument("--weights", type=Path, default=BASE_DIR / "models" / "extended1_bundle.json")
    parser.add_argument("--output", type=Path, default=BASE_DIR / "models" / "v4_category_valid.json")
    args = parser.parse_args()
    records = []
    for path, expected in read_annotations(args.data_dir, "valid_annotation.txt"):
        field = path.stem
        if field not in ("dan_toc", "ton_giao"):
            continue
        predictor = get_field_predictor(str(args.weights.resolve()), field)
        torch.set_num_threads(2)
        model, vocab, config = predictor.model, predictor.vocab, predictor.config
        model.eval()
        with Image.open(path) as image:
            detail = predict_field(predictor, image, field)
            tensor = process_input(image, config["dataset"]["image_height"], config["dataset"]["image_min_width"], config["dataset"]["image_max_width"])
        options = list({text.casefold(): text for text in reversed(vocabularies()[field])}.values())
        with torch.inference_mode():
            memory = model.transformer.forward_encoder(model.cnn(tensor))
            scores = string_scores(model, memory, vocab, options)
        rows = [{"value": text, "mean_log_probability": score, "probability": math.exp(score),
                 "distance_ratio": text_distance(detail["raw"].casefold(), text.casefold()) / max(len(text), len(detail["raw"]))}
                for text, score in zip(options, scores)]
        rows.sort(key=lambda row: row["mean_log_probability"], reverse=True)
        record = {"field": field, "form": path.parent.name, "expected": expected,
                  "raw": detail["raw"], "normalized": normalize_value(field, detail["raw"]),
                  "current_value": detail["value"], "candidates": rows}
        records.append(record)
        print(field, path.parent.name, repr(expected), repr(detail["raw"]),
              [(row["value"], round(row["probability"], 3), round(row["distance_ratio"], 3)) for row in rows[:3]], flush=True)
    configs = []
    for radius in (.65, 1.):
        for threshold in (.3, .4, .45, .5, .55, .6):
            for margin in (0., .05, .08, .1):
                rows = []
                for record in records:
                    normalized = record["normalized"]
                    options = record["candidates"]
                    is_exact = any(row["value"].casefold() == normalized.casefold() for row in options)
                    candidates = [row for row in options if row["distance_ratio"] <= radius]
                    value = normalized
                    if not is_exact and candidates:
                        best = candidates[0]
                        actual_margin = best["mean_log_probability"]-candidates[1]["mean_log_probability"] if len(candidates) > 1 else 1.
                        if best["probability"] >= threshold and actual_margin >= margin:
                            value = best["value"]
                    rows.append({"field": record["field"], "form": record["form"], "value": value,
                                 "expected": record["expected"], "exact": value.casefold() == record["expected"].casefold()})
                configs.append({"radius": radius, "probability": threshold, "margin": margin,
                                "exact": sum(row["exact"] for row in rows), "total": len(rows), "rows": rows})
    configs.sort(key=lambda row: (row["exact"], row["probability"], row["margin"], -row["radius"]), reverse=True)
    report = {"split": "valid", "records": records, "configs": configs,
              "current_exact": sum(row["current_value"].casefold() == row["expected"].casefold() for row in records),
              "annotation_sha256": hashlib.sha256((args.data_dir / "valid_annotation.txt").read_bytes()).hexdigest(),
              "note": "Parameter search on validation, exploratory; no test read and no vocabulary from labels."}
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("CURRENT", report["current_exact"], "BEST", [{key: row[key] for key in ("radius", "probability", "margin", "exact")} for row in configs[:5]], flush=True)


if __name__ == "__main__":
    main()
