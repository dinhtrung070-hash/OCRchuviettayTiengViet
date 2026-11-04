"""Conservative dotted-line removal that keeps strokes touching the baseline."""

from __future__ import annotations

import cv2
import numpy as np


PREPROCESS_VERSION = "stroke_preserving_v2"


def normalize_paper(region: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    # Opening the light paper estimates illumination without thinning ink.
    background = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, np.ones((31, 31), np.uint8))
    background = cv2.GaussianBlur(background, (0, 0), 5)
    return cv2.divide(gray, np.maximum(background, 1), scale=255)


def blue_signal(region: np.ndarray) -> tuple[np.ndarray, bool]:
    pixels = region.astype(np.float32)
    light = pixels.min(axis=2) > 190
    difference = pixels[:, :, 0] - pixels[:, :, 2]
    cast = float(np.median(difference[light])) if light.any() else 0.0
    signal = np.maximum(difference - cast, 0)
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    colored = ((hsv[:, :, 0] >= 85) & (hsv[:, :, 0] <= 145) &
               (hsv[:, :, 1] > 45) & (signal > 14))
    dark = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY) < 190
    blue = int(colored.sum()) >= 60 and colored.sum() / max(1, dark.sum()) > 0.12
    return signal, blue


def remove_isolated_dots(gray: np.ndarray, blank_region: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    """Erase small detached print dots; never erase a component joined to a stroke."""
    gray = gray.copy()
    foreground = np.uint8(gray < 190) * 255
    n, ids, boxes, _ = cv2.connectedComponentsWithStats(foreground)
    height, width = gray.shape
    # A title descender or the next row may enter a fixed field at its edge.
    # Only remove short clipped fragments; complete accents remain untouched.
    for i in range(1, n):
        x, y, w, h, area = boxes[i]
        if (y == 0 or y + h == height) and h < height * 0.22:
            fragment = np.uint8(ids == i)
            fragment = cv2.dilate(fragment, np.ones((3, 3), np.uint8))
            gray[fragment > 0] = 255
            foreground[fragment > 0] = 0
    count, components, stats, centroids = cv2.connectedComponentsWithStats(foreground)
    # Handwriting components must extend appreciably above/below the dot band.
    major = [i for i in range(1, count)
             if stats[i, cv2.CC_STAT_HEIGHT] >= max(12, height * 0.14)
             and stats[i, cv2.CC_STAT_AREA] >= 30]
    if not major:
        return gray.copy(), foreground
    if blank_region is not None:
        blank = cv2.cvtColor(blank_region, cv2.COLOR_BGR2GRAY)
        if blank.shape != gray.shape:
            blank = cv2.resize(blank, (width, height))
        n, _, boxes, centers = cv2.connectedComponentsWithStats(np.uint8(blank < 190))
        dots = [centers[i] for i in range(1, n)
                if boxes[i, cv2.CC_STAT_WIDTH] <= 16 and boxes[i, cv2.CC_STAT_HEIGHT] <= 16
                and 3 <= boxes[i, cv2.CC_STAT_AREA] <= 160]
    else:
        dots = []
    if dots:
        baseline = float(np.median([point[1] for point in dots]))
    else:
        small = [centroids[i, 1] for i in range(1, count)
                 if 3 <= stats[i, cv2.CC_STAT_AREA] <= 150 and stats[i, cv2.CC_STAT_HEIGHT] <= 14]
        baseline = float(np.median(small)) if len(small) >= 4 else height * 0.7
    left = min(stats[i, cv2.CC_STAT_LEFT] for i in major)
    right = max(stats[i, cv2.CC_STAT_LEFT] + stats[i, cv2.CC_STAT_WIDTH] for i in major)
    output = gray.copy()
    for i in range(1, count):
        x, y, w, h, area = stats[i]
        center_x, center_y = centroids[i]
        small_dot = (3 <= area <= 160 and w <= 16 and h <= 16 and
                     abs(center_y - baseline) <= 8)
        # Inside the word, a tiny detached mark could be a Vietnamese accent
        # or date separator. Keep it; train the recognizer on remaining dots.
        outside_word = center_x < left - 5 or center_x > right + 5
        if small_dot and outside_word:
            mask = np.uint8(components == i)
            mask = cv2.dilate(mask, np.ones((3, 3), np.uint8))
            output[mask > 0] = 255
            foreground[mask > 0] = 0
    return output, foreground


def trim_ink(gray: np.ndarray, foreground: np.ndarray, pad: int = 10) -> np.ndarray:
    count, _, stats, _ = cv2.connectedComponentsWithStats(foreground)
    substantial = [i for i in range(1, count) if stats[i, cv2.CC_STAT_AREA] >= 12]
    if not substantial:
        return gray
    left = min(stats[i, cv2.CC_STAT_LEFT] for i in substantial)
    top = min(stats[i, cv2.CC_STAT_TOP] for i in substantial)
    right = max(stats[i, cv2.CC_STAT_LEFT] + stats[i, cv2.CC_STAT_WIDTH] for i in substantial)
    bottom = max(stats[i, cv2.CC_STAT_TOP] + stats[i, cv2.CC_STAT_HEIGHT] for i in substantial)
    return gray[max(0, top-pad):min(gray.shape[0], bottom+pad),
                max(0, left-pad):min(gray.shape[1], right+pad)]


def clean_name_line(region: np.ndarray, blank_region: np.ndarray) -> np.ndarray:
    """Keep the complete name while excluding distant fragments below its baseline."""
    gray = normalize_paper(region)
    cleaned, foreground = remove_isolated_dots(gray, blank_region)
    blank = cv2.cvtColor(blank_region, cv2.COLOR_BGR2GRAY)
    n, _, boxes, centers = cv2.connectedComponentsWithStats(np.uint8(blank < 190))
    dots = [centers[i, 1] for i in range(1, n)
            if 3 <= boxes[i, cv2.CC_STAT_AREA] <= 160 and boxes[i, cv2.CC_STAT_HEIGHT] <= 16
            and boxes[i, cv2.CC_STAT_WIDTH] <= 16]
    if len(dots) < 4:
        return cv2.cvtColor(trim_ink(cleaned, foreground), cv2.COLOR_GRAY2BGR)
    baseline = float(np.median(dots))
    height, width = gray.shape
    n, ids, stats, _ = cv2.connectedComponentsWithStats(foreground)
    core = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= 30
            and stats[i, cv2.CC_STAT_HEIGHT] >= height*.14
            and stats[i, cv2.CC_STAT_TOP] < baseline-8
            and stats[i, cv2.CC_STAT_TOP]+stats[i, cv2.CC_STAT_HEIGHT] > baseline-height*.55]
    if not core:
        return cv2.cvtColor(trim_ink(cleaned, foreground), cv2.COLOR_GRAY2BGR)
    margin = max(24, int(height*.22))
    left = max(0, min(stats[i, cv2.CC_STAT_LEFT] for i in core)-margin)
    right = min(width, max(stats[i, cv2.CC_STAT_LEFT]+stats[i, cv2.CC_STAT_WIDTH] for i in core)+margin)
    keep = np.zeros_like(foreground)
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if x+w >= left and x <= right and y <= baseline+height*.16:
            keep[ids == i] = 255
    output = cleaned.copy()
    remove = cv2.dilate(np.uint8((foreground > 0) & (keep == 0)), np.ones((3, 3), np.uint8))
    output[remove > 0] = 255
    return cv2.cvtColor(trim_ink(output, keep), cv2.COLOR_GRAY2BGR)


def clean_handwriting(region: np.ndarray, blank_region: np.ndarray | None = None,
                      mode: str = "chroma") -> tuple[np.ndarray, str]:
    gray = normalize_paper(region)
    signal, blue = blue_signal(region)
    ink = "blue" if blue else "dark"
    if blue and mode == "chroma":
        level = max(25.0, float(np.percentile(signal[signal > 14], 85)))
        # Preserve continuous stroke edges instead of hard HSV thresholding.
        gray = np.uint8(255 - np.clip((signal - 5) / level, 0, 1) * 245)
    if mode == "raw":
        foreground = np.uint8(gray < 190) * 255
    else:
        gray, foreground = remove_isolated_dots(gray, blank_region)
    cropped = trim_ink(gray, foreground)
    return cv2.cvtColor(cropped, cv2.COLOR_GRAY2BGR), ink
