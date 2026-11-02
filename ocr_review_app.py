r"""Local Gradio demo: upload, review/adjust crops, then run VietOCR.

Run from the project root:
    Demo\.venv\Scripts\python.exe Demo\ocr_review_app.py

Open http://127.0.0.1:7860 in a browser. The default model is the original
VietOCR vgg_transformer. Pass --weights PATH after fine-tuning is available.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from functools import lru_cache
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
from PIL import Image

from prepare_dataset import FIELDS, SCALE, blue_handwriting_only, dark_handwriting_only, estimate_alignment, read_image


BASE_DIR = Path(__file__).resolve().parent
TEMPLATE = BASE_DIR / "phieu_mau.png"
MODEL_CONFIG = BASE_DIR / "models" / "vgg_transformer.yml"
PRETRAINED_WEIGHTS = BASE_DIR / "models" / "pretrained_vgg_transformer.pth"
FIELD_LABELS = {
    "ho_ten": "Họ và tên",
    "ngay_sinh": "Ngày sinh",
    "dan_toc": "Dân tộc",
    "ton_giao": "Tôn giáo",
}
FIELD_ORDER = tuple(FIELDS)


def bgr_to_pil(image: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


def pil_to_bgr(image: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)


def get_model_config(weights: str | None):
    from vietocr.tool.config import Cfg

    config = Cfg.load_config_from_file(MODEL_CONFIG)
    config["device"] = "cpu"
    if weights:
        path = Path(weights).expanduser().resolve()
        if not path.is_file():
            raise ValueError(f"Không tìm thấy trọng số model: {path}")
        config["weights"] = str(path)
    else:
        # A clone can run without committing the 145 MB pretrained file.
        # VietOCR downloads the URL from the config on first inference.
        if PRETRAINED_WEIGHTS.is_file():
            config["weights"] = str(PRETRAINED_WEIGHTS)
    return config


@lru_cache(maxsize=2)
def get_predictor(weights: str | None):
    from vietocr.tool.predictor import Predictor

    return Predictor(get_model_config(weights))


def prepare_form(image_path: str) -> dict:
    if not image_path:
        raise ValueError("Hãy tải ảnh phiếu lên trước.")
    photo = read_image(Path(image_path))
    template = read_image(TEMPLATE)
    crops: dict[str, Image.Image | None] = {}
    methods: dict[str, str] = {}
    status = ""

    try:
        transform, inliers = estimate_alignment(template, photo)
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
        overlay = aligned.copy()
        for field, (x1, y1, x2, y2) in FIELDS.items():
            a, b, c, d = (v * SCALE for v in (x1, y1, x2, y2))
            region = aligned[b:d, a:c]
            cleaned, has_blue = blue_handwriting_only(region)
            has_dark = False
            if not has_blue:
                cleaned, has_dark = dark_handwriting_only(region, scaled_template[b:d, a:c])
            crops[field] = bgr_to_pil(cleaned)
            methods[field] = (
                "tự động · tách mực xanh" if has_blue else
                "tự động · trừ nền mẫu" if has_dark else "tự động · ảnh gốc"
            )
            cv2.rectangle(overlay, (a, b), (c, d), (0, 35, 230), 3)
            cv2.putText(
                overlay, field, (a + 6, max(28, b - 8)), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0, 35, 230), 2, cv2.LINE_AA,
            )
        status = f"Đã căn phiếu ({inliers} điểm khớp). Xem kỹ 4 ảnh crop rồi xác nhận từng trường."
    except (ValueError, np.linalg.LinAlgError) as exc:
        # The user can still crop each field manually from the uploaded image.
        aligned = photo
        overlay = photo.copy()
        for field in FIELD_ORDER:
            crops[field] = None
            methods[field] = "chưa có crop"
        status = f"Căn phiếu tự động chưa thành công: {exc}. Hãy sửa crop tay cho từng trường."

    return {
        "form_name": Path(image_path).name,
        "aligned": bgr_to_pil(aligned),
        "overlay": bgr_to_pil(overlay),
        "crops": crops,
        "methods": methods,
        "predictions": {},
        "status": status,
    }


def manual_crop(editor_value: dict | None, prepared: dict, field: str) -> dict:
    if not prepared or field not in FIELD_ORDER:
        raise ValueError("Hãy chọn trường cần sửa crop.")
    if not editor_value or editor_value.get("composite") is None:
        raise ValueError("Hãy dùng công cụ Crop trong khung chỉnh ảnh rồi bấm Apply.")
    selected = editor_value["composite"]
    if not isinstance(selected, Image.Image):
        selected = Image.fromarray(np.asarray(selected))
    selected = selected.convert("RGB")
    full = prepared["aligned"]
    if selected.width < 45 or selected.height < 20:
        raise ValueError("Vùng crop quá nhỏ; hãy chọn trọn dòng chữ.")
    if selected.width >= full.width * 0.95 and selected.height >= full.height * 0.95:
        raise ValueError("Ảnh vẫn gần bằng toàn phiếu. Hãy kéo khung Crop quanh đúng trường.")
    cleaned, has_blue = blue_handwriting_only(pil_to_bgr(selected))
    updated = dict(prepared)
    updated["crops"] = dict(prepared["crops"])
    updated["methods"] = dict(prepared["methods"])
    updated["predictions"] = dict(prepared["predictions"])
    updated["crops"][field] = bgr_to_pil(cleaned)
    updated["methods"][field] = "crop tay · tách mực xanh" if has_blue else "crop tay · ảnh gốc"
    updated["predictions"].pop(field, None)
    return updated


def recognize(prepared: dict, weights: str | None) -> dict[str, str]:
    if not prepared:
        raise ValueError("Hãy tải phiếu và kiểm tra crop trước.")
    for field in FIELD_ORDER:
        if prepared["crops"].get(field) is None:
            raise ValueError(f"Chưa có crop cho trường {FIELD_LABELS[field]}.")
    predictor = get_predictor(weights)
    return {
        field: " ".join(predictor.predict(prepared["crops"][field]).split())
        for field in FIELD_ORDER
    }


def result_payload(prepared: dict, edited: dict[str, str], weights: str | None) -> dict:
    if not prepared or set(prepared["predictions"]) != set(FIELD_ORDER):
        raise ValueError("Hãy chạy VietOCR trên bốn crop đã xác nhận trước.")
    fields = {}
    for field in FIELD_ORDER:
        value = " ".join((edited.get(field) or "").split())
        fields[field] = {
            "ocr_raw": prepared["predictions"][field],
            "confirmed": value,
            "crop_method": prepared["methods"][field],
        }
    return {
        "form": prepared["form_name"],
        "model": "VietOCR vgg_transformer · fine-tuned" if weights else "VietOCR vgg_transformer · gốc",
        "fields": fields,
    }


def build_app(weights: str | None = None) -> gr.Blocks:
    model_note = (
        f"Đang dùng trọng số fine-tune: `{Path(weights).name}`."
        if weights else "Đang dùng VietOCR gốc (chưa fine-tune)."
    )
    with gr.Blocks(title="Demo OCR phiếu viết tay · VietOCR") as app:
        gr.Markdown(
            "# OCR phiếu viết tay tiếng Việt\n"
            "Tải một phiếu lên → xem và xác nhận 4 crop → chạy VietOCR → sửa chữ nếu cần → tải JSON. "
            + model_note
        )
        prepared_state = gr.State(None)
        active_field = gr.State(None)

        with gr.Row():
            upload = gr.Image(
                type="filepath", label="Ảnh phiếu", sources=["upload"],
                height=300, buttons=["fullscreen"],
            )
            with gr.Column():
                gr.Markdown("**Có thể thử ngay với hai phiếu mẫu:**")
                gr.Examples(
                    examples=[[str(BASE_DIR / "phieu_da_viet.png")], [str(BASE_DIR / "phieudaviet1.png")]],
                    inputs=upload,
                )
                prepare_button = gr.Button("1. Căn phiếu và cắt tự động", variant="primary")
                status = gr.Markdown("Tải ảnh rồi bấm nút cắt tự động.")

        overlay = gr.Image(label="Toàn phiếu và các vùng tự cắt", interactive=False, height=330, buttons=["fullscreen"])
        gr.Markdown("## Kiểm tra từng vùng\nĐánh dấu **Crop đúng** cho cả bốn trường trước khi OCR. Nếu sai, bấm **Sửa crop** ở đúng trường đó.")

        crop_images = {}
        confirms = {}
        edit_buttons = {}
        for pair in (("ho_ten", "ngay_sinh"), ("dan_toc", "ton_giao")):
            with gr.Row():
                for field in pair:
                    with gr.Column():
                        crop_images[field] = gr.Image(label=FIELD_LABELS[field], interactive=False, height=170)
                        with gr.Row():
                            confirms[field] = gr.Checkbox(label="Crop đúng", value=False)
                            edit_buttons[field] = gr.Button("Sửa crop", size="sm")

        edit_note = gr.Markdown(visible=False)
        editor = gr.ImageEditor(
            type="pil", image_mode="RGB", label="Kéo khung Crop trên ảnh phiếu, rồi bấm Apply trong công cụ",
            transforms=["crop"], brush=False, eraser=False, layers=False,
            height=480, visible=False, buttons=["fullscreen"],
        )
        apply_button = gr.Button("Dùng crop vừa chọn", visible=False)

        ocr_button = gr.Button("2. Xác nhận crop và chạy VietOCR", variant="primary")
        gr.Markdown("## Kết quả nhận dạng\nNếu chữ sai, sửa trực tiếp trong ô. Muốn đổi vùng ảnh, bấm **Sửa crop** rồi chạy OCR lại.")
        text_boxes = {}
        for pair in (("ho_ten", "ngay_sinh"), ("dan_toc", "ton_giao")):
            with gr.Row():
                for field in pair:
                    text_boxes[field] = gr.Textbox(label=FIELD_LABELS[field], interactive=True)
        export_button = gr.Button("3. Xem và tải JSON")
        json_view = gr.JSON(label="Kết quả cuối cùng")
        json_file = gr.File(label="Tải file JSON", interactive=False)

        def on_prepare(image_path):
            try:
                state = prepare_form(image_path)
            except ValueError as exc:
                raise gr.Error(str(exc)) from exc
            result = {
                prepared_state: state,
                active_field: None,
                overlay: state["overlay"],
                status: state["status"],
                editor: gr.update(value=None, visible=False),
                edit_note: gr.update(value="", visible=False),
                apply_button: gr.update(visible=False),
                json_view: None,
                json_file: None,
            }
            for field in FIELD_ORDER:
                result[crop_images[field]] = state["crops"][field]
                result[confirms[field]] = False
                result[text_boxes[field]] = ""
            return result

        prepare_button.click(
            on_prepare, inputs=upload,
            outputs=[prepared_state, active_field, overlay, status, editor, edit_note, apply_button,
                     json_view, json_file, *crop_images.values(), *confirms.values(), *text_boxes.values()],
        )

        for field in FIELD_ORDER:
            def begin_edit(state, selected=field):
                if not state:
                    raise gr.Error("Hãy cắt tự động hoặc tải phiếu trước.")
                return (
                    selected,
                    gr.update(value=state["aligned"], visible=True),
                    gr.update(value=f"Đang sửa **{FIELD_LABELS[selected]}**. Kéo vùng trên ảnh dưới đây, bấm Apply của công cụ rồi bấm «Dùng crop vừa chọn».", visible=True),
                    gr.update(visible=True),
                )

            edit_buttons[field].click(
                begin_edit, inputs=prepared_state,
                outputs=[active_field, editor, edit_note, apply_button],
            )

        def on_apply(editor_value, state, selected, *current_confirmations):
            try:
                updated = manual_crop(editor_value, state, selected)
            except ValueError as exc:
                raise gr.Error(str(exc)) from exc
            values = dict(zip(FIELD_ORDER, current_confirmations))
            values[selected] = False
            result = {
                prepared_state: updated,
                status: f"Đã thay crop **{FIELD_LABELS[selected]}**. Xem lại ảnh, tích Crop đúng rồi chạy OCR.",
                editor: gr.update(value=None, visible=False),
                edit_note: gr.update(value="", visible=False),
                apply_button: gr.update(visible=False),
                json_view: None,
                json_file: None,
                text_boxes[selected]: "",
            }
            for field in FIELD_ORDER:
                result[crop_images[field]] = updated["crops"][field]
                result[confirms[field]] = values[field]
            return result

        apply_button.click(
            on_apply,
            inputs=[editor, prepared_state, active_field, *confirms.values()],
            outputs=[prepared_state, status, editor, edit_note, apply_button, json_view, json_file,
                     *crop_images.values(), *confirms.values(), *text_boxes.values()],
        )

        def on_ocr(state, *confirmation_values):
            if not all(confirmation_values):
                raise gr.Error("Hãy xác nhận Crop đúng cho cả bốn trường trước khi OCR.")
            try:
                predictions = recognize(state, weights)
            except ValueError as exc:
                raise gr.Error(str(exc)) from exc
            except Exception as exc:
                raise gr.Error(f"Không chạy được VietOCR: {exc}") from exc
            updated = dict(state)
            updated["predictions"] = predictions
            result = {
                prepared_state: updated,
                status: "Đã chạy VietOCR. Kiểm tra và sửa chữ trong bốn ô bên dưới trước khi xuất JSON.",
                json_view: None,
                json_file: None,
            }
            for field in FIELD_ORDER:
                result[text_boxes[field]] = predictions[field]
            return result

        ocr_button.click(
            on_ocr, inputs=[prepared_state, *confirms.values()],
            outputs=[prepared_state, status, json_view, json_file, *text_boxes.values()],
        )

        def on_export(state, *texts):
            try:
                payload = result_payload(state, dict(zip(FIELD_ORDER, texts)), weights)
            except ValueError as exc:
                raise gr.Error(str(exc)) from exc
            output_dir = Path(tempfile.mkdtemp(prefix="vietocr_demo_"))
            output_file = output_dir / "result.json"
            output_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            return payload, str(output_file)

        export_button.click(
            on_export, inputs=[prepared_state, *text_boxes.values()], outputs=[json_view, json_file],
        )

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", help="Path to fine-tuned VietOCR weights; omit for pretrained baseline")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    weights = os.path.abspath(args.weights) if args.weights else None
    if weights and not Path(weights).is_file():
        parser.error(f"Không tìm thấy file trọng số: {weights}")
    app = build_app(weights)
    app.launch(server_name="127.0.0.1", server_port=args.port, share=False, show_error=True)


if __name__ == "__main__":
    main()
