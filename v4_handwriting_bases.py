"""Fetch pinned public handwriting initializations and evaluate VALID only.

No remote Python code is executed and no form images are uploaded. Downloads
must match the pinned SHA-256; checkpoints are loaded with weights_only=True.
External checkpoints are separately attributed, not claimed as team training.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
from pathlib import Path

import requests
import torch
import yaml
from PIL import Image

from evaluate_test import edit_distance, summarize
from field_inference import normalize_value, predict_field
from model_runtime import Predictor
from ocr_review_app import BASE_DIR, get_model_config
from train_extended1 import read_annotations


SOURCES = [
    {"id": "dunghugging", "repo": "DungHugging/vietocr-handwritten-finetune",
     "revision": "0827b42e17cb0e1c1bbda5aac11f7cc5c816e0ca", "filename": "transformerocr.pth",
     "sha256": "d5357730c328e40dd1aeae7288994d8ad85979f8334c03c9bfad53c320114fa7",
     "license_from_model_card": "mit", "claimed_training_dataset": "UIT_HWDB_line"},
    {"id": "vudang449", "repo": "vudang449/vietocr-vietnamese-handwriting",
     "revision": "f1973b930f907c378f6e3152ce9d8658724b228a", "filename": "best_ep11.pth",
     "sha256": "f73bc43c152f882a7de3adc38be93901070ecc4b1bbf6ffa02440fd0906fa3ff",
     "license_from_model_card": "apache-2.0", "claimed_training_dataset": "5CD-AI/Viet-Handwriting-OCR-v2"},
]


def digest(path):
    checksum = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def fetch(source):
    root = BASE_DIR / "models"
    output = root / f"external_{source['id']}_handwriting.pth"
    base = f"https://huggingface.co/{source['repo']}/resolve/{source['revision']}"
    if not output.is_file() or digest(output) != source["sha256"]:
        temporary = output.with_suffix(".pth.tmp")
        checksum = hashlib.sha256()
        with requests.get(f"{base}/{source['filename']}", stream=True, timeout=(30, 120)) as response:
            response.raise_for_status()
            size = 0
            with temporary.open("wb") as handle:
                for chunk in response.iter_content(1024*1024):
                    handle.write(chunk)
                    checksum.update(chunk)
                    size += len(chunk)
                    if size // (32*1024*1024) != (size-len(chunk)) // (32*1024*1024):
                        print("download", source["id"], round(size/1024/1024), "MiB", flush=True)
        if checksum.hexdigest() != source["sha256"]:
            raise ValueError("External checkpoint SHA-256 mismatch")
        os.replace(temporary, output)
    card_response = requests.get(f"{base}/README.md", timeout=30)
    card_response.raise_for_status()
    provenance = dict(source, local_weights=output.name, model_card_url=f"https://huggingface.co/{source['repo']}",
                      model_card_text=card_response.text, weights_sha256_verified=digest(output),
                      note="External handwriting initialization, not the team's own fine-tuning. No dataset images uploaded.")
    config = get_model_config(str(output))
    if source["id"] == "dunghugging":
        response = requests.get(f"{base}/config.yml", timeout=30)
        response.raise_for_status()
        external_config = yaml.safe_load(response.text)
        for key in ("vocab", "backbone", "seq_modeling", "transformer"):
            if external_config[key] != config[key]:
                raise ValueError(f"External architecture/config mismatch: {key}")
        for key in ("ss", "ks", "hidden"):
            if external_config["cnn"][key] != config["cnn"][key]:
                raise ValueError(f"External CNN mismatch: {key}")
        provenance["external_config_checked"] = True
    else:
        response = requests.get(f"{base}/vocab.txt", timeout=30)
        response.raise_for_status()
        # The special mask token spelling differs; its index is not decoded.
        external_chars = response.content.decode("utf-8").splitlines()[4:]
        if external_chars != list(config["vocab"]):
            raise ValueError("External character ordering differs")
        provenance["external_vocab_order_checked"] = True
    (root / f"external_{source['id']}_provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")
    return output, provenance


def evaluate(weights, rows, metadata, threads):
    # The constructor enforces strict matching layer keys/shapes and loads only
    # tensor state. No external repository code/config is imported or executed.
    predictor = Predictor(get_model_config(str(weights)))
    torch.set_num_threads(threads)
    results = []
    for path, label in rows:
        info = metadata[path.relative_to(metadata["_root"]).as_posix()]
        with Image.open(path) as image:
            result = predict_field(predictor, image.convert("RGB"), info["field"], "greedy", True)
        raw, field = result["raw"], info["field"]
        results.append({"form": info["form"], "field": field, "expected": label, "predicted": raw,
                        "exact": raw == label, "edit_distance": edit_distance(label, raw),
                        "formatted": normalize_value(field, raw),
                        "formatted_exact": normalize_value(field, raw).casefold() == normalize_value(field, label).casefold(),
                        "structured_value": result["value"], "structured_proposal": result["structured_proposal"],
                        "structured_exact": result["value"].casefold() == normalize_value(field, label).casefold(),
                        "model_score": result["model_score"]})
    summary = summarize(results)
    summary["by_field"] = {field: summarize([r for r in results if r["field"] == field]) for field in sorted({r["field"] for r in results})}
    summary["rows"] = results
    del predictor
    gc.collect()
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--source", choices=[s["id"] for s in SOURCES], nargs="+")
    parser.add_argument("--fetch-only", action="store_true")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("Positive thread count required")
    torch.set_num_threads(args.threads)
    root = BASE_DIR / "prepared" / "extended1_v2"
    rows = read_annotations(root, "valid_annotation.txt")
    with (root / "to_label.csv").open(encoding="utf-8-sig", newline="") as handle:
        metadata = {row["image"]: row for row in csv.DictReader(handle)}
    metadata["_root"] = root.resolve()
    for source in SOURCES:
        if args.source and source["id"] not in args.source:
            continue
        weights, provenance = fetch(source)
        print("verified", source["id"], weights.name, flush=True)
        if args.fetch_only:
            continue
        result = evaluate(weights, rows, metadata, args.threads)
        report = {"split": "valid", "test_accessed": False, "preprocess_version": "stroke_preserving_v2",
                  "annotation_sha256": digest(root / "valid_annotation.txt"), "weights": str(weights),
                  "weights_sha256": provenance["weights_sha256_verified"], "source": {k:v for k,v in source.items()},
                  "initialization_only": True, "decoder": "greedy", "structured_decoding": True,
                  "result": result,
                  "limits": "External training overlap with private forms unknown; writer IDs unknown; labels not cross-verified. Not team's own fine-tuning."}
        report_path = BASE_DIR / "models" / f"extended1_v4_external_{source['id']}_valid.json"
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(source["id"], json.dumps({k:v for k,v in result.items() if k != "rows"}, ensure_ascii=False), flush=True)
        gc.collect()


if __name__ == "__main__":
    main()
