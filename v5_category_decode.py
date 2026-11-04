"""Opt-in finite-field proposals; preserves legacy/v4 decoding by default.

The caller supplies the TRAIN-selected score/odds settings from its bundle.
No name/date dictionary, expected labels, or image-specific rules are used.
"""

from __future__ import annotations

import math

from structured_decode import string_scores, vocabularies


def spelling_forms(text):
    return list(dict.fromkeys((text, text.lower(), text.title(), text.upper())))


def propose_category(model, memory, vocab, field, raw, normalized, distance, settings):
    if field not in ("dan_toc", "ton_giao") or not raw:
        return None
    threshold, odds = float(settings["threshold"]), float(settings["odds"])
    if not .25 <= threshold <= 1 or not 2 <= odds <= 10:
        raise ValueError("Invalid finite-field rejection settings")
    options = list({text.casefold(): text for text in reversed(vocabularies()[field])}.values())
    if any(text.casefold() == normalized.casefold() for text in options):
        return None
    choices = [text for text in options if distance(raw.casefold(), text.casefold()) /
               max(1, len(raw), len(text)) <= .65]
    forms = list(dict.fromkeys(form for text in choices for form in spelling_forms(text)))
    if not forms:
        return None
    scores = dict(zip(forms, string_scores(model, memory, vocab, forms)))
    ranked = sorted(((max(scores[form] for form in spelling_forms(text)), text) for text in choices), reverse=True)
    score, text = ranked[0]
    gap = score-ranked[1][0] if len(ranked) > 1 else 1.0
    probability = math.exp(score)
    if probability < threshold or gap < math.log(odds):
        return None
    return {"value": text, "score": round(probability, 4), "source": "field_vocabulary_v5",
            "score_note": "Image-conditioned mean token likelihood, not calibrated correctness probability.",
            "warning": "Nhận dạng theo danh mục; cần xác nhận lại với ảnh"}
