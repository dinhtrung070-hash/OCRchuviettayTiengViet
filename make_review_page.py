"""Build a local visual audit page for form crops and first-pass labels."""

from __future__ import annotations

import argparse
import csv
import html
import os
from collections import defaultdict
from pathlib import Path


def relative_image(image: Path, page_dir: Path) -> str:
    return html.escape(Path(os.path.relpath(image.resolve(), page_dir.resolve())).as_posix(), quote=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    args = parser.parse_args()
    with (args.data_dir / "to_label.csv").open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    by_form = defaultdict(list)
    for row in rows:
        by_form[row["form"]].append(row)
    cards = []
    def form_key(value: str):
        stem = Path(value).stem
        return (0, int(stem)) if stem.isdigit() else (1, stem)

    for form in sorted(by_form, key=form_key):
        fields = by_form[form]
        source = args.source_dir / form
        source_html = f'<img class="source" src="{relative_image(source, args.data_dir)}" alt="{html.escape(form)}">'
        crops = []
        for row in fields:
            crop = args.data_dir / row["image"]
            label = html.escape(row["label"] or "CHƯA GÁN NHÃN")
            crops.append(f'<div><b>{html.escape(row["field"])}:</b> {label}'
                         f'<br><img src="{relative_image(crop, args.data_dir)}" alt="crop"></div>')
        status = "Cần kiểm tra" if any(not row["label"] for row in fields) else "Đã gán nhãn lượt đầu"
        cards.append(f'<section><h2>{html.escape(form)} · {status}</h2>{source_html}'
                     f'<div class="grid">{"".join(crops)}</div></section>')
    page = ('<!doctype html><html lang="vi"><meta charset="utf-8"><title>Kiểm tra nhãn OCR</title>'
            '<style>body{font:16px system-ui;max-width:1250px;margin:25px auto;background:#f5f5f5}'
            'section{background:white;padding:20px;margin:24px 0;border:1px solid #ccc;border-radius:8px}'
            '.source{width:100%;height:auto}.grid{display:grid;grid-template-columns:1fr 1fr;gap:18px;margin-top:16px}'
            '.grid img{max-width:100%;max-height:170px;object-fit:contain;background:#fafafa}'
            'h2{margin-top:0}</style><body><h1>Kiểm tra nhãn và crop · 50 phiếu</h1>'
            '<p>Nhãn được đọc trực tiếp từ ảnh, chưa có danh sách đáp án hoặc mã người viết để đối chiếu.</p>'
            + ''.join(cards) + '</body></html>')
    output = args.data_dir / "review.html"
    output.write_text(page, encoding="utf-8")
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
