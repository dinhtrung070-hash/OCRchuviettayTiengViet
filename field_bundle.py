"""Validated, local-only checkpoint routing for each fixed form field."""

from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=4)
def load_bundle(filename: str) -> dict:
    path = Path(filename).resolve()
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("format") != "vietocr_field_bundle_v1":
        raise ValueError("Không đúng định dạng model bundle VietOCR")
    for entry in [data["base"], *data.get("fields", {}).values()]:
        model = (path.parent / entry["weights"]).resolve()
        if not model.is_relative_to(path.parent) or not model.is_file():
            raise ValueError(f"Thiếu checkpoint trong bundle: {entry['weights']}")
        if not isinstance(entry.get("sha256"), str) or len(entry["sha256"]) != 64:
            raise ValueError("Bundle thiếu checksum checkpoint")
    rules = data.get("category_policies", {})
    if not isinstance(rules, dict) or not set(rules) <= {"dan_toc", "ton_giao"}:
        raise ValueError("Chính sách giải mã chỉ áp dụng cho dân tộc/tôn giáo")
    for policy in rules.values():
        if not isinstance(policy, dict) or set(policy) != {"threshold", "odds"}:
            raise ValueError("Chính sách danh mục phải có threshold và odds")
        if any(type(value) not in (int, float) or not math.isfinite(value) for value in policy.values()):
            raise ValueError("Ngưỡng giải mã phải là số hữu hạn")
        if not .25 <= policy["threshold"] <= 1 or not 2 <= policy["odds"] <= 10:
            raise ValueError("Ngưỡng giải mã ngoài phạm vi hỗ trợ")
    return data


def resolve_model(weights: str | None, field: str) -> tuple[str | None, str | None]:
    if not weights or Path(weights).suffix.lower() != ".json":
        return weights, None
    path = Path(weights).resolve()
    data = load_bundle(str(path))
    entry = data.get("fields", {}).get(field, data["base"])
    return str((path.parent / entry["weights"]).resolve()), entry["sha256"]


def category_policy(weights: str | None, field: str) -> dict | None:
    if not weights or Path(weights).suffix.lower() != ".json":
        return None
    data = load_bundle(str(Path(weights).resolve()))
    policy = data.get("category_policies", {}).get(field)
    return dict(policy) if policy is not None else None
