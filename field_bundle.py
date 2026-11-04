"""Validated, local-only checkpoint routing for each fixed form field."""

from __future__ import annotations

import json
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
    return data


def resolve_model(weights: str | None, field: str) -> tuple[str | None, str | None]:
    if not weights or Path(weights).suffix.lower() != ".json":
        return weights, None
    path = Path(weights).resolve()
    data = load_bundle(str(path))
    entry = data.get("fields", {}).get(field, data["base"])
    return str((path.parent / entry["weights"]).resolve()), entry["sha256"]
