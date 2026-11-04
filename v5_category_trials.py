"""Train-calibrated, validation-only finite-field decoding experiments.

No test annotations are opened. Vocabulary choices come from the existing
schema, never from expected labels. Raw OCR is preserved in every record.
"""

from __future__ import annotations

import gc
import hashlib
import json
import math
from pathlib import Path

import torch
from PIL import Image
from vietocr.tool.translate import process_input

from field_inference import normalize_value, predict_field, text_distance
from ocr_review_app import BASE_DIR, get_field_predictor, get_predictor
from field_bundle import load_bundle
from structured_decode import string_scores, vocabularies
from train_extended1 import read_annotations


DATA = BASE_DIR / "prepared" / "extended1_v2"
WEIGHTS = BASE_DIR / "models" / "extended1_bundle.json"
OUTPUT = BASE_DIR / "models" / "v5_category_valid.json"
FIELDS = ("dan_toc", "ton_giao")


def same(left, right):
    return left.casefold() == right.casefold()


def spelling_forms(text):
    # Formatting variants, not alternative words or substitutions. Keep the
    # returned class label separate from the image-conditioned spelling score.
    return list(dict.fromkeys((text, text.lower(), text.title(), text.upper())))


def policy_value(record, variant=False, threshold=.55, margin=.08):
    if record["in_vocabulary"]:
        return record["normalized"]
    ranked = record["variant_candidates" if variant else "candidates"]
    if not ranked:
        return record["normalized"]
    gap = ranked[0]["score"] - ranked[1]["score"] if len(ranked) > 1 else 1.0
    return (ranked[0]["value"] if ranked[0]["probability"] >= threshold and gap >= margin
            else record["normalized"])


def summarize(records, policy):
    rows = [{"form": r["form"], "field": r["field"], "expected": r["expected"],
             "raw": r["raw"], "value": policy(r), "correct": same(policy(r), r["expected"])} for r in records]
    return {"correct": sum(r["correct"] for r in rows), "total": len(rows),
            "by_field": {f: {"correct": sum(r["correct"] for r in rows if r["field"] == f),
                              "total": sum(r["field"] == f for r in rows)} for f in FIELDS}, "rows": rows}


def collect():
    records = []
    for field in FIELDS:
        predictor = get_field_predictor(str(WEIGHTS.resolve()), field)
        torch.set_num_threads(2)
        model, vocab, config = predictor.model, predictor.vocab, predictor.config
        model.eval()
        options = list({text.casefold(): text for text in reversed(vocabularies()[field])}.values())
        for split in ("train", "valid"):
            for path, expected in read_annotations(DATA, f"{split}_annotation.txt"):
                if path.stem != field:
                    continue
                with Image.open(path) as image:
                    detail = predict_field(predictor, image, field)
                    picture = process_input(image, config["dataset"]["image_height"],
                                            config["dataset"]["image_min_width"], config["dataset"]["image_max_width"])
                raw = detail["raw"]
                normalized = normalize_value(field, raw)
                exact = any(same(normalized, option) for option in options)
                record = {"split": split, "field": field, "form": path.parent.name, "expected": expected,
                          "raw": raw, "normalized": normalized, "current_value": detail["value"],
                          "in_vocabulary": exact, "candidates": [], "variant_candidates": []}
                if not exact:
                    choices = [option for option in options if text_distance(raw.casefold(), option.casefold()) /
                               max(1, len(raw), len(option)) <= .65]
                    forms = list(dict.fromkeys(form for option in choices for form in spelling_forms(option)))
                    if forms:
                        with torch.inference_mode():
                            memory = model.transformer.forward_encoder(model.cnn(picture))
                            scores = dict(zip(forms, string_scores(model, memory, vocab, forms)))
                        for option in choices:
                            score = scores[option]
                            best_form = max(spelling_forms(option), key=scores.__getitem__)
                            variant_score = scores[best_form]
                            record["candidates"].append({"value": option, "score": score, "probability": math.exp(score)})
                            record["variant_candidates"].append({"value": option, "score": variant_score,
                                                                  "probability": math.exp(variant_score), "spelling": best_form})
                        record["candidates"].sort(key=lambda r: r["score"], reverse=True)
                        record["variant_candidates"].sort(key=lambda r: r["score"], reverse=True)
                records.append(record)
                if not exact:
                    print(split, field, path.parent.name, repr(raw),
                          [(r["value"], round(r["probability"], 3), r["spelling"]) for r in record["variant_candidates"][:2]], flush=True)
        del predictor, model
        get_predictor.cache_clear()
        gc.collect()
    return records


def calibrate(train):
    # Pick thresholds on TRAIN only. A floor prevents the initial 3:1-only
    # experiment from selecting a tiny absolute score merely for extra coverage.
    # Fixed odds alternatives are generic rejection rules, not class priors.
    thresholds = sorted({.25, .55, *[r["variant_candidates"][0]["probability"] for r in train
                                    if r["variant_candidates"] and r["variant_candidates"][0]["probability"] >= .25]})
    policies = []
    for odds in (2, 3, 5):
        for threshold in thresholds:
            predictions = [(r, policy_value(r, True, threshold, math.log(odds))) for r in train]
            changed = [(r, value) for r, value in predictions if value != r["normalized"]]
            correct = sum(same(value, r["expected"]) for r, value in changed)
            precision = correct / max(1, len(changed))
            policies.append({"threshold": threshold, "margin": math.log(odds), "odds": odds, "support": len(changed),
                             "correct": correct, "precision": precision})
    eligible = [r for r in policies if r["support"] >= 4 and r["precision"] >= .95]
    selected = max(eligible, key=lambda r: (r["support"], r["margin"], r["threshold"])) if eligible else None
    return selected, policies


def main():
    torch.set_num_threads(2)
    records = collect()
    train = [r for r in records if r["split"] == "train"]
    valid = [r for r in records if r["split"] == "valid"]
    selected, calibration = calibrate(train)
    results = {}
    for split, rows in (("train", train), ("valid", valid)):
        results[split] = {"current": summarize(rows, lambda r: r["current_value"]),
                          "case_variants_same_gate": summarize(rows, lambda r: policy_value(r, True))}
        if selected:
            results[split]["train_selected_gate"] = summarize(rows, lambda r: policy_value(r, True, selected["threshold"], selected["margin"]))
    report = {"test_accessed": False, "weights_sha256": hashlib.sha256(WEIGHTS.read_bytes()).hexdigest(),
              "preprocess_version": load_bundle(str(WEIGHTS.resolve()))["preprocess_version"],
              "annotation_sha256": {s: hashlib.sha256((DATA / f"{s}_annotation.txt").read_bytes()).hexdigest() for s in ("train", "valid")},
              "records": records, "selected_from_train": selected, "calibration": calibration, "results": results,
              "note": "Character scores are not calibrated probabilities; raw OCR is retained. Train precision is optimistic and validation is small."}
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({s: {p: {k: v[k] for k in ("correct", "total", "by_field")} for p, v in ps.items()} for s, ps in results.items()}, ensure_ascii=False), flush=True)
    print("SELECTED", selected, flush=True)


if __name__ == "__main__":
    main()
