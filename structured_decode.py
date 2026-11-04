"""Optional model-based reranking for finite fields and invalid date strings.

All proposals use image-conditioned VietOCR likelihoods. The raw free decode
is retained; proposals are explicitly flagged for user review.
"""

from __future__ import annotations

import datetime
import json
import math
from functools import lru_cache
from pathlib import Path

import torch


@lru_cache(maxsize=1)
def vocabularies() -> dict:
    return json.loads((Path(__file__).parent / "field_vocab.json").read_text(encoding="utf-8"))


def string_scores(model, memory, vocab, texts: list[str]) -> list[float]:
    result = []
    for offset in range(0, len(texts), 8):
        labels = [vocab.encode(text) for text in texts[offset:offset+8]]
        maximum = max(map(len, labels))-1
        source = torch.zeros((maximum, len(labels)), dtype=torch.long, device=memory.device)
        for i, ids in enumerate(labels):
            source[:len(ids)-1, i] = torch.tensor(ids[:-1], device=memory.device)
        logits, _ = model.transformer.forward_decoder(source, memory.repeat(1, len(labels), 1))
        log_probs = logits.log_softmax(-1)
        for i, ids in enumerate(labels):
            score = sum(float(log_probs[i, j, token]) for j, token in enumerate(ids[1:])) / (len(ids)-1)
            result.append(score)
    return result


def valid_date(text: str) -> bool:
    try:
        parts = text.split("/")
        return (len(parts) == 3 and len(parts[2]) == 4 and
                datetime.date(int(parts[2]), int(parts[1]), int(parts[0])) <= datetime.date.today())
    except (ValueError, TypeError):
        return False


def allowed_date_characters(prefix: str) -> str:
    position = len(prefix)
    if position == 0:
        return "0123"
    if position == 1:
        return "123456789" if prefix == "0" else "01" if prefix == "3" else "0123456789"
    if position in (2, 5):
        return "/"
    if position == 3:
        return "01"
    if position == 4:
        return "123456789" if prefix[3] == "0" else "012"
    year_prefix = prefix[6:]
    maximum_year = datetime.date.today().year
    return "".join(char for char in "0123456789"
                   if int((year_prefix+char).ljust(4, "0")) <= maximum_year
                   and int((year_prefix+char).ljust(4, "9")) >= 1000)


def date_candidate(model, memory, vocab, width: int = 8) -> tuple[str | None, float]:
    active = [("", 0.0)]
    for _ in range(10):
        inputs = torch.tensor([vocab.encode(text)[:-1] for text, _ in active], device=memory.device).T
        logits, _ = model.transformer.forward_decoder(inputs, memory.repeat(1, len(active), 1))
        log_probs = logits[:, -1].log_softmax(-1)
        candidates = []
        for i, (text, score) in enumerate(active):
            for char in allowed_date_characters(text):
                candidates.append((text+char, score + float(log_probs[i, vocab.c2i[char]])))
        active = sorted(candidates, key=lambda row: row[1], reverse=True)[:width]
    texts = [text for text, _ in active if valid_date(text)]
    if not texts:
        return None, 0.0
    scores = string_scores(model, memory, vocab, texts)
    index = max(range(len(scores)), key=scores.__getitem__)
    return texts[index], math.exp(scores[index])


def propose(model, memory, vocab, field: str, raw: str, normalized: str, distance) -> dict | None:
    if field == "ngay_sinh" and not valid_date(normalized):
        value, score = date_candidate(model, memory, vocab)
        if value and score >= .5:
            return {"value": value, "score": round(score, 4), "source": "date_grammar",
                    "warning": "Nhận dạng theo khuôn ngày; cần đối chiếu lại chữ số với ảnh"}
    if field not in ("dan_toc", "ton_giao") or not raw:
        return None
    options = list({text.casefold(): text for text in reversed(vocabularies()[field])}.values())
    exact = next((text for text in options if text.casefold() == normalized.casefold()), None)
    if exact:
        return None
    candidates = [text for text in options
                  if distance(raw.casefold(), text.casefold()) / max(len(raw), len(text)) <= .65]
    if not candidates:
        return None
    scores = string_scores(model, memory, vocab, candidates)
    order = sorted(range(len(scores)), key=scores.__getitem__, reverse=True)
    first = order[0]
    probability = math.exp(scores[first])
    margin = scores[first] - scores[order[1]] if len(order) > 1 else 1.0
    if probability >= .55 and margin >= .08:
        return {"value": candidates[first], "score": round(probability, 4), "source": "field_vocabulary",
                "warning": "Nhận dạng theo danh mục; cần xác nhận lại với ảnh"}
    return None
