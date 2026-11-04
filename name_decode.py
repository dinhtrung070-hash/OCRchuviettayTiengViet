"""Unsupervised whitespace splitting for optional name-line recognition."""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image


def split_name(image: Image.Image) -> list[Image.Image]:
    gray = np.array(image.convert("L"))
    height, width = gray.shape
    n, ids, stats, _ = cv2.connectedComponentsWithStats(np.uint8(gray < 175))
    core = np.zeros_like(gray, dtype=bool)
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if h >= height*.12 and area >= 25 and y < height*.75 and y+h > height*.25:
            core[ids == i] = True
    columns = core.any(axis=0)
    occupied = np.flatnonzero(columns)
    if not len(occupied):
        return []
    gaps = []
    start = None
    for x in range(int(occupied[0]), int(occupied[-1])+1):
        if not columns[x] and start is None:
            start = x
        elif columns[x] and start is not None:
            gaps.append((start, x))
            start = None
    if not gaps:
        return []
    threshold = max(12, height*.13, max(b-a for a, b in gaps)*.45)
    gaps = [(a, b) for a, b in gaps if b-a >= threshold]
    if not 1 <= len(gaps) <= 5:
        return []
    boundaries = [0] + [(a+b)//2 for a, b in gaps] + [width]
    pieces = []
    for a, b in zip(boundaries[:-1], boundaries[1:]):
        if b-a < height*.25:
            return []
        piece = gray[:, a:b]
        ys, xs = np.where(piece < 180)
        if not len(xs):
            return []
        left, top = max(0, int(xs.min())-8), max(0, int(ys.min())-8)
        right, bottom = min(piece.shape[1], int(xs.max())+9), min(height, int(ys.max())+9)
        pieces.append(Image.fromarray(piece[top:bottom, left:right]).convert("RGB"))
    return pieces


def split_candidate(image: Image.Image, recognize) -> dict | None:
    pieces = split_name(image)
    if not pieces:
        return None
    details = [recognize(piece) for piece in pieces]
    if any(not item["raw"] or not item["ended"] for item in details):
        return None
    text = " ".join(item["raw"] for item in details)
    total = sum(max(1, len(item["raw"])) for item in details)
    score = sum(item["model_score"]*max(1, len(item["raw"])) for item in details)/total
    return {"raw": text, "model_score": score, "pieces": len(pieces), "ended": True}
