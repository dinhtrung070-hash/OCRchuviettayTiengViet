"""Bounded VietOCR inference, conservative formatting and review hints."""

from __future__ import annotations

import datetime
import re
import unicodedata

import torch
from vietocr.tool.translate import process_input


# Review hints only: these are examples, not a closed schema or auto-corrections.
ETHNICITY_HINTS = ("Kinh", "Tày", "Thái", "Mường", "Khmer", "Hoa", "Nùng", "Mông", "Dao", "Gia Rai",
                   "Ê Đê", "Ba Na", "Sán Chay", "Sán Chỉ", "Chăm", "Cơ Ho", "Xơ Đăng", "Sán Dìu", "Hán")
RELIGION_HINTS = ("Không", "Không có", "Phật giáo", "Công giáo", "Tin Lành", "Hồi giáo",
                  "Ki tô giáo", "Ấn Độ giáo", "Do Thái giáo", "Cao Đài", "Hòa Hảo")


def canonical_text(value: str) -> str:
    return " ".join(unicodedata.normalize("NFC", value).split())


def normalize_value(field: str, value: str) -> str:
    value = canonical_text(value)
    if field == "ngay_sinh" and re.fullmatch(r"[\d\s./-]+", value):
        parts = re.findall(r"\d+", value)
        if len(parts) == 3 and len(parts[2]) == 4:
            try:
                date = datetime.date(int(parts[2]), int(parts[1]), int(parts[0]))
                return f"{date.day:02d}/{date.month:02d}/{date.year:04d}"
            except ValueError:
                pass
    return value


def text_distance(left: str, right: str) -> int:
    previous = list(range(len(right)+1))
    for i, a in enumerate(left, 1):
        current = [i]
        for j, b in enumerate(right, 1):
            current.append(min(current[-1]+1, previous[j]+1, previous[j-1]+(a != b)))
        previous = current
    return previous[-1]


def review_hints(field: str, text: str, score: float, ended: bool) -> tuple[list[str], str | None]:
    warnings = []
    suggestion = None
    if not ended:
        warnings.append("Model chạm giới hạn độ dài; cần kiểm tra lại ảnh và nội dung")
    if score < .85:
        warnings.append("Model chưa chắc về một số ký tự")
    if re.search(r"(.)\1{5,}", text):
        warnings.append("Có ký tự lặp bất thường, có thể còn nhiễu dòng chấm")
    choices = ETHNICITY_HINTS if field == "dan_toc" else RELIGION_HINTS if field == "ton_giao" else ()
    if choices and text and text.casefold() not in {item.casefold() for item in choices}:
        nearest = min(choices, key=lambda item: text_distance(text.casefold(), item.casefold()))
        distance = text_distance(text.casefold(), nearest.casefold())
        if distance / max(len(text), len(nearest)) <= .4:
            suggestion = nearest
            warnings.append("Có gợi ý chính tả; cần đối chiếu ảnh trước khi sửa")
    return warnings, suggestion


def beam_tokens(model, memory, vocab, maximum: int, width: int = 4) -> list[int]:
    active = [([vocab.go], 0.0)]
    finished = []
    for _ in range(maximum):
        inputs = torch.tensor([tokens for tokens, _ in active], device=memory.device).T
        logits, _ = model.transformer.forward_decoder(inputs, memory.repeat(1, len(active), 1))
        log_probs = logits[:, -1].log_softmax(-1)
        candidates = []
        for i, (tokens, score) in enumerate(active):
            values, ids = log_probs[i].topk(width)
            candidates.extend((tokens + [int(token)], score + float(value)) for token, value in zip(ids, values))
        candidates.sort(key=lambda row: row[1], reverse=True)
        active = []
        for tokens, score in candidates[:width]:
            if tokens[-1] == vocab.eos:
                finished.append((tokens, score))
            else:
                active.append((tokens, score))
        if not active or (finished and active[0][1] <= max(score for _, score in finished)):
            break
    return max(finished or active, key=lambda row: row[1])[0]


def predict_field(predictor, image, field: str, decoder: str | None = None,
                  structured: bool | None = None, category_policy: dict | None = None) -> dict:
    config, model, vocab = predictor.config, predictor.model, predictor.vocab
    picture = process_input(image, config["dataset"]["image_height"],
                            config["dataset"]["image_min_width"], config["dataset"]["image_max_width"])
    picture = picture.to(config["device"])
    maximum = 64 if field == "ho_ten" else 32
    model.eval()
    tokens, probabilities = [vocab.go], []
    decoder = decoder or config.get("form_decoder", "greedy")
    with torch.inference_mode():
        features = model.cnn(picture)
        memory = model.transformer.forward_encoder(features)
        if decoder == "beam":
            tokens = beam_tokens(model, memory, vocab, maximum)
            inputs = torch.tensor(tokens[:-1], device=picture.device).unsqueeze(1)
            logits, _ = model.transformer.forward_decoder(inputs, memory)
            probabilities_all = logits[0].softmax(-1)
            probabilities = [float(probabilities_all[i, token]) for i, token in enumerate(tokens[1:]) if token >= 4]
        else:
            for _ in range(maximum):
                inputs = torch.tensor(tokens, device=picture.device).unsqueeze(1)
                logits, _ = model.transformer.forward_decoder(inputs, memory)
                probabilities_all = logits[0, -1].softmax(-1)
                token = int(probabilities_all.argmax())
                tokens.append(token)
                if token >= 4:
                    probabilities.append(float(probabilities_all[token]))
                if token == vocab.eos:
                    break
    raw = canonical_text(vocab.decode(tokens))
    score = sum(probabilities)/len(probabilities) if probabilities else 0.0
    value = normalize_value(field, raw)
    structured = config.get("form_structured", False) if structured is None else structured
    proposal = None
    if structured:
        from structured_decode import propose
        with torch.inference_mode():
            proposal = propose(model, memory, vocab, field, raw, value, text_distance,
                               category_policy=category_policy)
        if proposal:
            value = proposal["value"]
    warnings, suggestion = review_hints(field, value, score, tokens[-1] == vocab.eos)
    if proposal:
        warnings.append(proposal["warning"])
    result = {"raw": raw, "value": value, "model_score": round(score, 4),
            "score_note": "Mean token score; not a calibrated probability of correctness.",
            "warnings": warnings, "suggestion": suggestion, "ended": tokens[-1] == vocab.eos,
            "structured_proposal": proposal,
            "decoder": decoder}
    if category_policy is not None and structured and field in ("dan_toc", "ton_giao"):
        result["category_policy"] = dict(category_policy)
    return result
