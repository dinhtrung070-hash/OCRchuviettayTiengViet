"""Build a local visual audit page for form crops and first-pass labels."""

from __future__ import annotations

import argparse
import csv
import html
import os
import json
from collections import defaultdict
from pathlib import Path


def relative_image(image: Path, page_dir: Path) -> str:
    return html.escape(Path(os.path.relpath(image.resolve(), page_dir.resolve())).as_posix(), quote=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, help="Optional evaluate_test.py JSON; show evaluated forms and predictions")
    args = parser.parse_args()
    with (args.data_dir / "to_label.csv").open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    by_form = defaultdict(list)
    for row in rows:
        by_form[row["form"]].append(row)
    evaluation = json.loads(args.evaluation.read_text(encoding="utf-8")) if args.evaluation else None
    predictions = {}
    if evaluation:
        for model in ("baseline", "fine_tuned"):
            predictions[model] = {(row["form"], row["field"]): row for row in evaluation[model]["rows"]}
        evaluated_forms = {key[0] for key in predictions["baseline"]}
        by_form = {key: value for key, value in by_form.items() if key in evaluated_forms}
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
            result_html = ""
            if evaluation:
                base = predictions["baseline"][(form, row["field"])]
                tuned = predictions["fine_tuned"][(form, row["field"])]
                result_html = (f'<p>Gốc: <b>{html.escape(base["predicted"])}</b><br>'
                               f'Fine-tune, OCR thô: <b>{html.escape(tuned["predicted"])}</b><br>'
                               f'Kết quả có cấu trúc: <b>{html.escape(tuned.get("structured_value", tuned["predicted"]))}</b><br>'
                               f'Khớp toàn chuỗi OCR thô: {"ĐÚNG" if tuned["exact"] else "SAI"}</p>')
            raw = crop.with_name(crop.stem + "_raw.png")
            raw_html = (f'<details><summary>Xem crop gốc còn chấm</summary><img src="{relative_image(raw, args.data_dir)}" alt="raw"></details>'
                        if raw.is_file() else "")
            crops.append(f'<div><b>{html.escape(row["field"])}:</b> {label}'
                         f'<br><img src="{relative_image(crop, args.data_dir)}" alt="crop">{raw_html}{result_html}</div>')
        status = "Cần kiểm tra" if any(not row["label"] for row in fields) else "Đã gán nhãn lượt đầu"
        cards.append(f'<section><h2>{html.escape(form)} · {status}</h2>{source_html}'
                     f'<div class="grid">{"".join(crops)}</div></section>')
    summary = '<h1>Kiểm tra nhãn và crop</h1>'
    if evaluation:
        summary = '<h1>Đối chiếu VietOCR gốc và fine-tune</h1>'
        for model, name in (("baseline", "Gốc"), ("fine_tuned", "Fine-tune")):
            stats = evaluation[model]
            summary += (f'<p><b>{name}</b>: OCR thô đúng {stats["correct"]}/{stats["total"]}, CER {stats["cer"]:.1%}; '
                        f'có định dạng/danh mục đúng {stats.get("structured_correct", stats["correct"])}/{stats["total"]}.</p>')
        summary += '<p>Tập test đã được dùng khảo sát trong các lần thử trước; chưa biết ID người viết. Không coi đây là kiểm định độc lập.</p>'
    page = ('<!doctype html><html lang="vi"><meta charset="utf-8"><title>Kiểm tra nhãn OCR</title>'
            '<style>body{font:16px system-ui;max-width:1250px;margin:25px auto;background:#f5f5f5}'
            'section{background:white;padding:20px;margin:24px 0;border:1px solid #ccc;border-radius:8px}'
            '.source{width:100%;height:auto}.grid{display:grid;grid-template-columns:1fr 1fr;gap:18px;margin-top:16px}'
            '.grid img{max-width:100%;max-height:170px;object-fit:contain;background:#fafafa}'
            'h2{margin-top:0}</style><body>' + summary +
            '<p>Nhãn được đọc trực tiếp từ ảnh, chưa có danh sách đáp án hoặc mã người viết để đối chiếu.</p>'
            + ''.join(cards) + '</body></html>')
    output = args.data_dir / ("results.html" if evaluation else "review.html")
    output.write_text(page, encoding="utf-8")
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
