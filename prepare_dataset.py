"""Align the fixed form and export handwriting crops for labeling/OCR.

Example:
    python Demo/prepare_dataset.py --input-dir Demo --output-dir Demo/prepared
"""

from __future__ import annotations

import argparse
import csv
import re
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np


# Coordinates on Demo/phieu_mau.png (863 x 242). Keep the printed labels out.
FIELDS = {
    # The second row can rise into the name row; stop before its Dân tộc field.
    "ho_ten": (195, 96, 470, 150),
    "ngay_sinh": (195, 145, 338, 188),
    "dan_toc": (415, 145, 545, 188),
    "ton_giao": (638, 145, 825, 188),
}
LEGACY_FIELDS = {
    "ho_ten": (185, 105, 820, 157),
    "ngay_sinh": (185, 157, 365, 212),
    "dan_toc": (401, 157, 590, 212),
    "ton_giao": (635, 157, 825, 212),
}
SCALE = 3


def writer_id_from_filename(path: Path) -> str:
    """Read writer id from names such as w01_f01.jpg."""
    match = re.match(r"^(w\d{2,4})_", path.stem, flags=re.IGNORECASE)
    return match.group(1).lower() if match else ""


def read_image(path: Path) -> np.ndarray:
    # imdecode supports Unicode file names on Windows.
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot read image: {path}")
    return image


def write_image(path: Path, image: np.ndarray) -> None:
    ok, encoded = cv2.imencode(path.suffix, image)
    if not ok:
        raise ValueError(f"Cannot encode image: {path}")
    encoded.tofile(str(path))


def estimate_alignment(template: np.ndarray, photo: np.ndarray) -> tuple[np.ndarray, int]:
    sift = cv2.SIFT_create(nfeatures=3000)
    key_t, desc_t = sift.detectAndCompute(cv2.cvtColor(template, cv2.COLOR_BGR2GRAY), None)
    key_p, desc_p = sift.detectAndCompute(cv2.cvtColor(photo, cv2.COLOR_BGR2GRAY), None)
    if desc_t is None or desc_p is None:
        raise ValueError("Not enough image features to align the form")
    pairs = cv2.BFMatcher().knnMatch(desc_t, desc_p, k=2)
    matches = [m for m, n in pairs if m.distance < 0.72 * n.distance]
    if len(matches) < 12:
        raise ValueError(f"Only {len(matches)} good template matches")

    source = np.float32([key_t[m.queryIdx].pt for m in matches])
    target = np.float32([key_p[m.trainIdx].pt for m in matches])
    transform, inliers = cv2.findHomography(source, target, cv2.RANSAC, 5)
    count = int(inliers.sum()) if inliers is not None else 0
    if transform is None or count < 12:
        raise ValueError(f"Alignment is unreliable ({count} inliers)")
    return transform, count


def detect_layout(template: np.ndarray, photo: np.ndarray) -> tuple[str, np.ndarray, int]:
    """Distinguish the old uppercase-heading form from the current template."""
    current_transform, current_inliers = estimate_alignment(template, photo)
    old_reference = Path(__file__).parent / "phieu_da_viet.png"
    if old_reference.is_file():
        try:
            old = legacy_reference(str(old_reference))
            ratio = min(1.0, 1000 / photo.shape[1])
            small_photo = cv2.resize(photo, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_AREA)
            _, old_inliers = estimate_alignment(old, small_photo)
            if old_inliers >= 25 and old_inliers > current_inliers:
                return "legacy", current_transform, current_inliers
        except ValueError:
            pass
    return "current", current_transform, current_inliers


@lru_cache(maxsize=1)
def legacy_reference(path: str) -> np.ndarray:
    original = read_image(Path(path))
    ratio = 900 / original.shape[1]
    return cv2.resize(original, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_AREA)


def extract_handwriting(region: np.ndarray, blank_region: np.ndarray, layout: str) -> tuple[np.ndarray, str]:
    """Use the same crop cleaning for dataset generation and live inference."""
    cleaned, blue = blue_handwriting_only(region)
    if blue:
        return cleaned, "blue"
    if layout == "legacy":
        # The blank template belongs to the current layout; subtraction would
        # remove unrelated strokes from the old form. Keep the visible crop.
        return region, "raw"
    cleaned, dark = dark_handwriting_only(region, blank_region)
    return cleaned, "dark" if dark else "raw"


def blue_handwriting_only(region: np.ndarray, blank_region: np.ndarray | None = None) -> tuple[np.ndarray, bool]:
    """Keep blue handwriting and suppress printed text and dotted lines."""
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([90, 45, 25]), np.array([140, 255, 255]))
    if blank_region is not None:
        blank = cv2.cvtColor(blank_region, cv2.COLOR_BGR2GRAY)
        if blank.shape != mask.shape:
            blank = cv2.resize(blank, (mask.shape[1], mask.shape[0]))
        printed = cv2.dilate(np.uint8(blank < 215) * 255, np.ones((9, 9), np.uint8))
        mask[printed > 0] = 0
    if cv2.countNonZero(mask) < 40:
        return region, False
    count, components, stats, _ = cv2.connectedComponentsWithStats(mask)
    major = []
    for component in range(1, count):
        area = stats[component, cv2.CC_STAT_AREA]
        if area >= 35:
            major.append(component)
    if not major:
        return region, False
    # Long dotted baselines sometimes acquire a weak blue cast in photos.
    # Bound the crop by actual handwriting components, retaining small accents
    # within that area but dropping distant dots and sensor noise.
    left = min(stats[component, cv2.CC_STAT_LEFT] for component in major)
    top = min(stats[component, cv2.CC_STAT_TOP] for component in major)
    right = max(stats[component, cv2.CC_STAT_LEFT] + stats[component, cv2.CC_STAT_WIDTH] for component in major)
    bottom = max(stats[component, cv2.CC_STAT_TOP] + stats[component, cv2.CC_STAT_HEIGHT] for component in major)
    x1, x2 = max(0, left - 18), min(region.shape[1], right + 18)
    y1, y2 = max(0, top - 18), min(region.shape[0], bottom + 18)
    mask[:y1] = 0
    mask[y2:] = 0
    mask[:, :x1] = 0
    mask[:, x2:] = 0
    mask = cv2.dilate(mask, np.ones((2, 2), np.uint8), iterations=1)
    cleaned = np.full_like(region, 255)
    cleaned[mask > 0] = region[mask > 0]
    return cleaned[y1:y2, x1:x2], True


def dark_handwriting_only(region: np.ndarray, blank_region: np.ndarray) -> tuple[np.ndarray, bool]:
    """Remove printed text and dots using the registered blank template."""
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    blank = cv2.cvtColor(blank_region, cv2.COLOR_BGR2GRAY)
    if blank.shape != gray.shape:
        blank = cv2.resize(blank, (gray.shape[1], gray.shape[0]))
    # Printing is slightly misregistered after homography. Cover its full
    # stroke and a small halo before retaining dark pixels from the photo.
    printed = np.uint8(blank < 215) * 255
    printed = cv2.dilate(printed, np.ones((9, 9), np.uint8))
    mask = np.uint8((gray < 175) & (printed == 0)) * 255
    count, components, stats, _ = cv2.connectedComponentsWithStats(mask)
    for component in range(1, count):
        if stats[component, cv2.CC_STAT_AREA] < 16:
            mask[components == component] = 0
    if cv2.countNonZero(mask) < 40:
        return region, False
    mask = cv2.dilate(mask, np.ones((2, 2), np.uint8), iterations=1)
    ys, xs = np.where(mask > 0)
    pad = 12
    x1, x2 = max(0, int(xs.min()) - pad), min(region.shape[1], int(xs.max()) + pad + 1)
    y1, y2 = max(0, int(ys.min()) - pad), min(region.shape[0], int(ys.max()) + pad + 1)
    cleaned = np.full_like(region, 255)
    cleaned[mask > 0] = region[mask > 0]
    return cleaned[y1:y2, x1:x2], True


def process_form(template: np.ndarray, source: Path, output: Path) -> list[dict[str, str]]:
    photo = read_image(source)
    layout, transform, inliers = detect_layout(template, photo)
    height, width = template.shape[:2]
    scale_matrix = np.diag([SCALE, SCALE, 1.0])
    aligned = cv2.warpPerspective(
        photo,
        scale_matrix @ np.linalg.inv(transform),
        (width * SCALE, height * SCALE),
        flags=cv2.INTER_CUBIC,
        borderValue=(255, 255, 255),
    )
    scaled_template = cv2.resize(template, (width * SCALE, height * SCALE), interpolation=cv2.INTER_CUBIC)
    form_dir = output / source.stem
    form_dir.mkdir(parents=True, exist_ok=True)
    write_image(form_dir / "aligned.png", aligned)

    debug = aligned.copy()
    rows = []
    field_boxes = LEGACY_FIELDS if layout == "legacy" else FIELDS
    for field, (x1, y1, x2, y2) in field_boxes.items():
        box = tuple(value * SCALE for value in (x1, y1, x2, y2))
        a, b, c, d = box
        region = aligned[b:d, a:c]
        cleaned, ink = extract_handwriting(region, scaled_template[b:d, a:c], layout)
        crop_file = form_dir / f"{field}.png"
        write_image(crop_file, cleaned)
        cv2.rectangle(debug, (a, b), (c, d), (0, 0, 255), 2)
        rows.append({
            "form": source.name,
            "writer_id": writer_id_from_filename(source),
            "field": field,
            "image": str(crop_file.relative_to(output)).replace("\\", "/"),
            "label": "",
            "blue_ink_detected": str(ink == "blue"),
            "alignment_inliers": str(inliers),
            "layout": layout,
        })
    write_image(form_dir / "debug_fields.png", debug)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", type=Path, default=Path(__file__).parent / "phieu_mau.png")
    parser.add_argument("--input-dir", type=Path, default=Path(__file__).parent)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "prepared")
    parser.add_argument(
        "--ground-truth",
        type=Path,
        help="Optional CSV with form, writer_id and one column per field",
    )
    args = parser.parse_args()
    template = read_image(args.template)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    extensions = {".png", ".jpg", ".jpeg"}
    candidates = sorted(p for p in args.input_dir.iterdir() if p.is_file() and p.suffix.lower() in extensions and p.resolve() != args.template.resolve())
    # The demo folder also contains crop/debug images. A dedicated input folder
    # accepts any filename (e.g. IMG_0123.jpg).
    if args.input_dir.resolve() == Path(__file__).parent.resolve():
        candidates = [p for p in candidates if "phieu" in p.stem.lower()]
    for source in candidates:
        try:
            produced = process_form(template, source, args.output_dir)
            rows.extend(produced)
            print(f"OK: {source.name} ({produced[0]['alignment_inliers']} alignment inliers)")
        except (ValueError, np.linalg.LinAlgError) as exc:
            print(f"SKIP: {source.name}: {exc}")
    manifest = args.output_dir / "to_label.csv"
    previous = {}
    if manifest.exists():
        with manifest.open(newline="", encoding="utf-8-sig") as handle:
            previous = {(row["form"], row["field"]): row for row in csv.DictReader(handle)}
    for row in rows:
        old = previous.get((row["form"], row["field"]), {})
        row["writer_id"] = old.get("writer_id", "") or row["writer_id"]
        row["label"] = old.get("label", "")
    if args.ground_truth:
        with args.ground_truth.open(newline="", encoding="utf-8-sig") as handle:
            truth = {item["form"]: item for item in csv.DictReader(handle)}
        for row in rows:
            item = truth.get(row["form"])
            if item:
                row["writer_id"] = item.get("writer_id", "").strip()
                row["label"] = item[row["field"]].strip()
    updated = {(row["form"], row["field"]) for row in rows}
    rows.extend(old for key, old in previous.items() if key not in updated and (args.output_dir / old["image"]).is_file())
    with manifest.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=["form", "writer_id", "field", "image", "label", "blue_ink_detected", "alignment_inliers", "layout"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Manifest contains {len(rows)} crops: {manifest}")


if __name__ == "__main__":
    main()
