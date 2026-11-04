"""Compare local ethnicity checkpoints using TRAIN/VALID/replay only.

This diagnostic never reads test annotations or changes production artifacts.
Training fit is reported separately and is not an independent accuracy score.
"""

from __future__ import annotations

import gc
import hashlib
import json
from pathlib import Path

import torch
from PIL import Image

from field_inference import normalize_value, predict_field, text_distance
from model_runtime import Predictor
from ocr_review_app import BASE_DIR, get_model_config
from train_extended1 import read_annotations, read_regression


def summary(rows):
    return {"correct": sum(row["exact"] for row in rows), "total": len(rows),
            "cer": sum(row["distance"] for row in rows)/sum(len(row["expected"]) for row in rows),
            "formatted_correct": sum(row["formatted_exact"] for row in rows),
            "structured_correct": sum(row["structured_exact"] for row in rows), "rows": rows}


def main():
    torch.set_num_threads(2)
    root = BASE_DIR / "prepared" / "extended1_v2"
    groups = {split: [(path, label) for path, label in read_annotations(root, f"{split}_annotation.txt")
                      if path.stem == "dan_toc"] for split in ("train", "valid")}
    groups["replay_train"] = [(path, label) for path, label in read_regression(root / "pilot") if path.stem == "dan_toc"]
    if {path for path, _ in groups["train"] + groups["replay_train"]} & {path for path, _ in groups["valid"]}:
        raise ValueError("Train and validation overlap")
    checkpoints = ("extended1_v2_continued.pth", "extended1_v4_handwriting_adapt_dan_toc.pth", "extended1_v4_dung_adapt_dan_toc.pth")
    report = {"field": "dan_toc", "test_accessed": False,
              "preprocess_version": "stroke_preserving_v2", "decoder": "greedy", "structured_decoding": True,
              "limits": "Train and replay are fit diagnostics, not held-out scores. Unknown writer IDs, first-pass labels, reused validation.",
              "results": {}}
    for filename in checkpoints:
        path = BASE_DIR / "models" / filename
        checksum = hashlib.sha256(path.read_bytes()).hexdigest()
        predictor = Predictor(get_model_config(str(path.resolve())))
        torch.set_num_threads(2)
        result = {"weights_sha256": checksum, "splits": {}}
        for split, samples in groups.items():
            rows = []
            for source, expected in samples:
                with Image.open(source) as picture:
                    detail = predict_field(predictor, picture.convert("RGB"), "dan_toc", "greedy", True)
                rows.append({"image": str(source), "expected": expected, "raw": detail["raw"],
                    "value": detail["value"], "model_score": detail["model_score"],
                    "proposal": detail["structured_proposal"], "warnings": detail["warnings"],
                    "exact": detail["raw"] == expected, "distance": text_distance(expected, detail["raw"]),
                    "formatted_exact": normalize_value("dan_toc", detail["raw"]).casefold() == normalize_value("dan_toc", expected).casefold(),
                    "structured_exact": detail["value"].casefold() == normalize_value("dan_toc", expected).casefold()})
            result["splits"][split] = summary(rows)
        report["results"][filename] = result
        print(filename, {split: {k: v for k, v in stats.items() if k != "rows"} for split, stats in result["splits"].items()}, flush=True)
        (BASE_DIR / "models" / "v5_ethnicity_heads.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        del predictor
        gc.collect()


if __name__ == "__main__":
    main()
