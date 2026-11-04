"""Validation-only N-best decoding experiments; never changes active models.

Length penalties and a generic person-name alphabet are compared without
reading test annotations or constructing a name dictionary from labels.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path

import torch
from PIL import Image
from vietocr.tool.translate import process_input

from field_inference import canonical_text, text_distance
from ocr_review_app import BASE_DIR, get_field_predictor, get_predictor
from train_extended1 import read_annotations


def nbest(model, memory, vocab, maximum=48, width=8, name_alphabet=False):
    allowed = torch.ones(len(vocab), dtype=torch.bool, device=memory.device)
    allowed[[vocab.pad, vocab.go, vocab.mask_token]] = False
    if name_alphabet:
        allowed[:] = False
        allowed[vocab.eos] = True
        for char, index in vocab.c2i.items():
            if char.isalpha() or char in " '-":
                allowed[index] = True
    active, finished = [([vocab.go], 0.)], []
    for _ in range(maximum):
        inputs = torch.tensor([tokens for tokens, _ in active], device=memory.device).T
        logits, _ = model.transformer.forward_decoder(inputs, memory.repeat(1, len(active), 1))
        log_probs = logits[:, -1].log_softmax(-1).masked_fill(~allowed, float("-inf"))
        values, ids = log_probs.topk(width, dim=-1)
        candidates = [(tokens + [int(ids[i, k])], score + float(values[i, k]))
                      for i, (tokens, score) in enumerate(active) for k in range(width)]
        candidates.sort(key=lambda item: item[1], reverse=True)
        active = []
        for tokens, score in candidates:
            if tokens[-1] == vocab.eos:
                finished.append((tokens, score))
            elif len(active) < width:
                active.append((tokens, score))
        finished.sort(key=lambda item: item[1], reverse=True)
        finished = finished[:width]
        if not active or (len(finished) == width and active[0][1] <= finished[-1][1]):
            break
    pool = finished or active
    return [{"text": canonical_text(vocab.decode(tokens)), "log_probability": score,
             "token_count": len(tokens)-1, "ended": tokens[-1] == vocab.eos}
            for tokens, score in pool]


def summarize(rows):
    return {"exact": sum(row["exact"] for row in rows), "total": len(rows),
            "cer": sum(row["distance"] for row in rows) / max(1, sum(len(row["expected"]) for row in rows)),
            "rows": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=BASE_DIR / "prepared" / "extended1_v2")
    parser.add_argument("--weights", type=Path, default=BASE_DIR / "models" / "extended1_bundle.json")
    parser.add_argument("--output", type=Path, default=BASE_DIR / "models" / "v4_decode_valid.json")
    parser.add_argument("--fields", nargs="+", default=["ho_ten", "ngay_sinh"])
    parser.add_argument("--width", type=int, default=8)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    data = read_annotations(args.data_dir, "valid_annotation.txt")
    results = {}
    for field in args.fields:
        predictor = get_field_predictor(str(args.weights.resolve()), field)
        torch.set_num_threads(args.threads)
        model, vocab, config = predictor.model, predictor.vocab, predictor.config
        model.eval()
        records = []
        for path, expected in data:
            if path.stem != field:
                continue
            with Image.open(path) as image:
                tensor = process_input(image, config["dataset"]["image_height"], config["dataset"]["image_min_width"], config["dataset"]["image_max_width"])
            with torch.inference_mode():
                memory = model.transformer.forward_encoder(model.cnn(tensor.to(config["device"])))
                pools = {"free": nbest(model, memory, vocab, 48 if field == "ho_ten" else 32, args.width)}
                if field == "ho_ten":
                    pools["name_alphabet"] = nbest(model, memory, vocab, 48, args.width, True)
            records.append({"form": path.parent.name, "expected": expected, "pools": pools})
            print(field, path.parent.name, {key: value[0]["text"] for key, value in pools.items()}, flush=True)
        configurations = {}
        for alphabet in ("free", "name_alphabet") if field == "ho_ten" else ("free",):
            for alpha in (0., .5, 1.):
                rows = []
                for record in records:
                    best = max(record["pools"][alphabet], key=lambda row: row["log_probability"] / row["token_count"]**alpha)
                    rows.append({"form": record["form"], "expected": record["expected"], "predicted": best["text"],
                                 "exact": best["text"] == record["expected"], "distance": text_distance(best["text"], record["expected"])})
                configurations[f"{alphabet}_alpha{alpha}"] = summarize(rows)
        results[field] = {"model": predictor.form_metadata, "configurations": configurations, "nbest": records}
        print(field, {key: {k: row[k] for k in ("exact", "total", "cer")} for key, row in configurations.items()}, flush=True)
        del predictor, model, memory
        get_predictor.cache_clear()
        gc.collect()
    report = {"split": "valid", "width": args.width, "weights": str(args.weights),
              "weights_sha256": hashlib.sha256(args.weights.read_bytes()).hexdigest(),
              "annotation_sha256": hashlib.sha256((args.data_dir / "valid_annotation.txt").read_bytes()).hexdigest(),
              "fields": results, "note": "Validation-only inference experiment; generic alphabet, no name label dictionary."}
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
