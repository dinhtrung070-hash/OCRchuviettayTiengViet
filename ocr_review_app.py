r"""Local Gradio demo: upload, review/adjust crops, then run VietOCR.

Run from the project root:
    Demo\.venv\Scripts\python.exe Demo\ocr_review_app.py

Open http://127.0.0.1:7860 in a browser. The default model is the original
VietOCR vgg_transformer. Pass --weights PATH after fine-tuning is available.
"""

from __future__ import annotations

import argparse
import datetime
import json
import hashlib
import os
import re
import tempfile
from functools import lru_cache
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
from PIL import Image

from field_inference import normalize_value, predict_field
from field_bundle import load_bundle, resolve_model, category_policy
from handwriting_preprocess import PREPROCESS_VERSION, clean_handwriting
from prepare_dataset import (FIELDS, LEGACY_FIELDS, SCALE, detect_layout,
                             extract_handwriting, blue_handwriting_only, read_image)


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
EXAMPLE_PATHS = [BASE_DIR / "phieu_da_viet.png", BASE_DIR / "phieudaviet1.png"]
for local_form in (BASE_DIR / "forms" / "dataset_20261102" / "1.jpg",
                   BASE_DIR / "forms" / "dataset_20261102" / "50.jpg"):
    if local_form.is_file():
        EXAMPLE_PATHS.append(local_form)


def bgr_to_pil(image: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


def pil_to_bgr(image: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)


def get_model_config(weights: str | None):
    from vietocr.tool.config import Cfg

    config = Cfg.load_config_from_file(MODEL_CONFIG)
    config["device"] = "cpu"
    # The full VietOCR state dict includes all CNN weights; ImageNet weights
    # would only trigger an unnecessary second download on a fresh machine.
    config["cnn"]["pretrained"] = False
    config["form_structured"] = True
    config["form_decoder"] = "greedy"
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


@lru_cache(maxsize=4)
def get_predictor(weights: str | None):
    import torch
    from model_runtime import Predictor

    torch.set_num_threads(4)
    config = get_model_config(weights)
    predictor = Predictor(config)
    source = Path(config["weights"])
    predictor.form_metadata = {"weights": source.name if source.is_file() else config["weights"],
                               "sha256": hashlib.sha256(source.read_bytes()).hexdigest() if source.is_file() else None}
    return predictor


def get_field_predictor(weights: str | None, field: str):
    resolved, checksum = resolve_model(weights, field)
    predictor = get_predictor(resolved)
    if checksum and predictor.form_metadata["sha256"] != checksum:
        raise ValueError(f"Checkpoint cho {FIELD_LABELS[field]} không khớp checksum bundle")
    return predictor


def model_identity(weights: str | None) -> dict:
    if weights and Path(weights).suffix.lower() == ".json":
        data = load_bundle(str(Path(weights).resolve()))
        identity = {"name": data["name"], "base": data["base"], "fields": data["fields"],
                    "preprocess_version": data["preprocess_version"]}
        if "initializations" in data:
            identity["initializations"] = data["initializations"]
        if "category_policies" in data:
            identity["category_policies"] = data["category_policies"]
        return identity
    return get_predictor(weights).form_metadata


def prepare_form(image_path: str) -> dict:
    if not image_path:
        raise ValueError("Hãy tải ảnh phiếu lên trước.")
    photo = read_image(Path(image_path))
    template = read_image(TEMPLATE)
    crops: dict[str, Image.Image | None] = {}
    raw_crops: dict[str, Image.Image | None] = {}
    methods: dict[str, str] = {}
    status = ""

    try:
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
        overlay = aligned.copy()
        field_boxes = LEGACY_FIELDS if layout == "legacy" else FIELDS
        for field, (x1, y1, x2, y2) in field_boxes.items():
            a, b, c, d = (v * SCALE for v in (x1, y1, x2, y2))
            region = aligned[b:d, a:c]
            cleaned, ink = extract_handwriting(region, scaled_template[b:d, a:c], layout, field)
            raw_crops[field] = bgr_to_pil(region)
            crops[field] = bgr_to_pil(cleaned)
            methods[field] = (
                "tự động · lọc chấm, giữ nét chữ" if layout == "current" else
                "tự động · tách mực xanh" if ink == "blue" else "tự động · ảnh gốc"
            )
            cv2.rectangle(overlay, (a, b), (c, d), (0, 35, 230), 3)
            cv2.putText(
                overlay, field, (a + 6, max(28, b - 8)), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0, 35, 230), 2, cv2.LINE_AA,
            )
        layout_name = "mẫu mới" if layout == "current" else "mẫu cũ"
        status = f"Đã căn phiếu theo {layout_name}. Xem kỹ 4 ảnh crop rồi xác nhận từng trường."
    except (ValueError, np.linalg.LinAlgError) as exc:
        # The user can still crop each field manually from the uploaded image.
        aligned = photo
        layout = "unknown"
        overlay = photo.copy()
        for field in FIELD_ORDER:
            crops[field] = None
            raw_crops[field] = None
            methods[field] = "chưa có crop"
        status = f"Căn phiếu tự động chưa thành công: {exc}. Hãy sửa crop tay cho từng trường."

    return {
        "form_name": Path(image_path).name,
        "aligned": bgr_to_pil(aligned),
        "overlay": bgr_to_pil(overlay),
        "crops": crops,
        "raw_crops": raw_crops,
        "clean_crops": dict(crops),
        "methods": methods,
        "clean_methods": dict(methods),
        "layout": layout,
        "preprocess_version": PREPROCESS_VERSION,
        "predictions": {},
        "prediction_details": {},
        "model_name": "",
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
    if prepared.get("layout") == "legacy":
        cleaned, _ = blue_handwriting_only(pil_to_bgr(selected))
    else:
        cleaned, _ = clean_handwriting(pil_to_bgr(selected), mode="dots")
    updated = dict(prepared)
    updated["crops"] = dict(prepared["crops"])
    updated["methods"] = dict(prepared["methods"])
    updated["raw_crops"] = dict(prepared["raw_crops"])
    updated["clean_crops"] = dict(prepared["clean_crops"])
    updated["clean_methods"] = dict(prepared["clean_methods"])
    updated["predictions"] = dict(prepared["predictions"])
    updated["prediction_details"] = dict(prepared.get("prediction_details", {}))
    updated["raw_crops"][field] = selected
    updated["crops"][field] = bgr_to_pil(cleaned)
    updated["clean_crops"][field] = bgr_to_pil(cleaned)
    updated["methods"][field] = "crop tay · lọc dòng chấm"
    updated["clean_methods"][field] = updated["methods"][field]
    updated["predictions"].pop(field, None)
    updated["prediction_details"].pop(field, None)
    return updated


def select_crop_version(prepared: dict, field: str, cleaned: bool) -> dict:
    if not prepared or prepared["raw_crops"].get(field) is None:
        raise ValueError("Hãy cắt phiếu trước khi chọn ảnh.")
    updated = dict(prepared)
    for key in ("crops", "methods", "predictions", "prediction_details"):
        updated[key] = dict(prepared.get(key, {}))
    updated["crops"][field] = prepared["clean_crops" if cleaned else "raw_crops"][field]
    updated["methods"][field] = prepared["clean_methods"][field] if cleaned else "crop · ảnh gốc, giữ dòng chấm"
    updated["predictions"].pop(field, None)
    updated["prediction_details"].pop(field, None)
    return updated


def recognize(prepared: dict, weights: str | None) -> dict[str, str]:
    return {field: item["raw"] for field, item in recognize_detailed(prepared, weights).items()}


def recognize_detailed(prepared: dict, weights: str | None) -> dict[str, dict]:
    if not prepared:
        raise ValueError("Hãy tải phiếu và kiểm tra crop trước.")
    for field in FIELD_ORDER:
        if prepared["crops"].get(field) is None:
            raise ValueError(f"Chưa có crop cho trường {FIELD_LABELS[field]}.")
    results = {}
    for field in FIELD_ORDER:
        predictor = get_field_predictor(weights, field)
        results[field] = predict_field(predictor, prepared["crops"][field], field,
                                       category_policy=category_policy(weights, field))
        results[field]["model_metadata"] = predictor.form_metadata
    return results


def validate_field(field: str, value: str) -> tuple[bool, str]:
    if not value:
        return False, "Trường đang trống"
    if field == "ngay_sinh":
        parts = re.split(r"[./-]", value.replace(" ", ""))
        if len(parts) != 3 or not all(part.isdigit() for part in parts) or len(parts[2]) != 4:
            return False, "Ngày sinh cần có dạng ngày/tháng/năm"
        try:
            date = datetime.date(int(parts[2]), int(parts[1]), int(parts[0]))
            if date > datetime.date.today():
                return False, "Ngày sinh nằm trong tương lai"
        except ValueError:
            return False, "Ngày sinh không hợp lệ"
        return True, ""
    if any(char.isdigit() for char in value):
        return False, "Trường chữ có chứa chữ số"
    if field == "ho_ten" and len(value.split()) < 2:
        return False, "Họ tên cần ít nhất hai từ"
    return True, ""


def result_payload(prepared: dict, edited: dict[str, str], weights: str | None) -> dict:
    if not prepared or set(prepared["predictions"]) != set(FIELD_ORDER):
        raise ValueError("Hãy chạy VietOCR trên bốn crop đã xác nhận trước.")
    fields = {}
    for field in FIELD_ORDER:
        value = normalize_value(field, edited.get(field) or "")
        valid, message = validate_field(field, value)
        fields[field] = {
            "ocr_raw": prepared["predictions"][field],
            "formatted": normalize_value(field, prepared["predictions"][field]),
            "suggested_value": prepared.get("prediction_details", {}).get(field, {}).get("value", normalize_value(field, prepared["predictions"][field])),
            "confirmed": value,
            "crop_method": prepared["methods"][field],
            "valid": valid,
            "validation_message": message,
            "recognition": prepared.get("prediction_details", {}).get(field, {}),
            "needs_review": not valid or bool(prepared.get("prediction_details", {}).get(field, {}).get("warnings")),
        }
    return {
        "form": prepared["form_name"],
        "model": prepared.get("model_name") or ("VietOCR vgg_transformer · fine-tuned" if weights else "VietOCR vgg_transformer · gốc"),
        "model_metadata": prepared.get("model_metadata", {}),
        "all_fields_valid": all(item["valid"] for item in fields.values()),
        "validation_note": "Basic format checks only; validity does not guarantee correct transcription.",
        "preprocess_version": prepared.get("preprocess_version", "unknown"),
        "fields": fields,
    }


def build_app(weights: str | None = None, compare_weights: str | None = None,
              default_model: str = "baseline") -> gr.Blocks:
    current_label = (load_bundle(str(Path(compare_weights).resolve())).get("name", "Fine-tune hiện tại")
                     if compare_weights and Path(compare_weights).suffix.lower() == ".json" else "Fine-tune hiện tại")
    model_options = [("VietOCR gốc", "baseline"), (current_label, "fine_tuned")]
    previous_models = {}
    if compare_weights:
        for version, key in ((4, "previous_v4"), (3, "previous")):
            previous_path = BASE_DIR / "models" / f"extended1_v{version}_previous_bundle.json"
            if not previous_path.is_file() or Path(compare_weights).resolve() == previous_path.resolve():
                continue
            try:
                history = load_bundle(str(previous_path.resolve()))
            except (ValueError, OSError, KeyError, TypeError):
                continue
            if history.get("name") != current_label:
                previous_models[key] = str(previous_path.resolve())
                model_options.append((f"Fine-tune v{version} · đối chiếu", key))
    model_note = (
        "Có thể chọn VietOCR gốc hoặc bản fine-tune Extended 1 để so trên cùng ảnh crop."
        if compare_weights else
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
                gr.Markdown("**Có thể thử ngay với các phiếu mẫu trên máy:**")
                gr.Examples(
                    examples=[[str(path)] for path in EXAMPLE_PATHS],
                    inputs=upload,
                )
                prepare_button = gr.Button("1. Căn phiếu và cắt tự động", variant="primary")
                status = gr.Markdown("Tải ảnh rồi bấm nút cắt tự động.")

        overlay = gr.Image(label="Toàn phiếu và các vùng tự cắt", interactive=False, height=330, buttons=["fullscreen"])
        gr.Markdown("## Kiểm tra từng vùng\nĐánh dấu **Crop đúng** cho cả bốn trường trước khi OCR. Nếu sai, bấm **Sửa crop** ở đúng trường đó.")

        crop_images = {}
        clean_switches = {}
        confirms = {}
        edit_buttons = {}
        for pair in (("ho_ten", "ngay_sinh"), ("dan_toc", "ton_giao")):
            with gr.Row():
                for field in pair:
                    with gr.Column():
                        crop_images[field] = gr.Image(label=FIELD_LABELS[field], interactive=False, height=170)
                        clean_switches[field] = gr.Checkbox(label="Lọc dòng chấm", value=True,
                                                            info="Tắt để so với crop gốc nếu thấy mất nét.")
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

        model_choice = gr.Radio(
            choices=model_options,
            value=default_model, label="Model dùng cho lượt OCR này", visible=bool(compare_weights),
        )
        ocr_button = gr.Button("2. Xác nhận crop và chạy VietOCR", variant="primary")
        gr.Markdown("## Kết quả nhận dạng\nNếu chữ sai, sửa trực tiếp trong ô. Muốn đổi vùng ảnh, bấm **Sửa crop** rồi chạy OCR lại.")
        text_boxes = {}
        for pair in (("ho_ten", "ngay_sinh"), ("dan_toc", "ton_giao")):
            with gr.Row():
                for field in pair:
                    text_boxes[field] = gr.Textbox(label=FIELD_LABELS[field], interactive=True)
        recognition_note = gr.Markdown("")
        export_button = gr.Button("3. Xác nhận nội dung và tải JSON")
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
                recognition_note: "",
            }
            for field in FIELD_ORDER:
                result[crop_images[field]] = state["crops"][field]
                result[confirms[field]] = False
                result[text_boxes[field]] = ""
                result[clean_switches[field]] = True
            return result

        prepare_button.click(
            on_prepare, inputs=upload,
            outputs=[prepared_state, active_field, overlay, status, editor, edit_note, apply_button,
                     json_view, json_file, recognition_note, *crop_images.values(), *confirms.values(), *text_boxes.values(), *clean_switches.values()],
        )

        for field in FIELD_ORDER:
            def on_clean_switch(state, enabled, selected=field):
                try:
                    updated = select_crop_version(state, selected, enabled)
                except ValueError as exc:
                    raise gr.Error(str(exc)) from exc
                return {prepared_state: updated, crop_images[selected]: updated["crops"][selected],
                        confirms[selected]: False, text_boxes[selected]: "", json_view: None,
                        json_file: None, recognition_note: "",
                        status: f"Đã đổi ảnh {FIELD_LABELS[selected]}. Xem lại crop và xác nhận trước khi OCR."}

            clean_switches[field].input(
                on_clean_switch, inputs=[prepared_state, clean_switches[field]],
                outputs=[prepared_state, crop_images[field], confirms[field], text_boxes[field],
                         json_view, json_file, recognition_note, status],
            )

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
                clean_switches[selected]: True,
                recognition_note: "",
            }
            for field in FIELD_ORDER:
                result[crop_images[field]] = updated["crops"][field]
                result[confirms[field]] = values[field]
            return result

        apply_button.click(
            on_apply,
            inputs=[editor, prepared_state, active_field, *confirms.values()],
            outputs=[prepared_state, status, editor, edit_note, apply_button, json_view, json_file,
                     *crop_images.values(), *confirms.values(), *text_boxes.values(), *clean_switches.values(), recognition_note],
        )

        def on_ocr(state, selected_model, *confirmation_values):
            if not all(confirmation_values):
                raise gr.Error("Hãy xác nhận Crop đúng cho cả bốn trường trước khi OCR.")
            if selected_model in previous_models:
                selected_weights = previous_models[selected_model]
            elif selected_model == "fine_tuned":
                selected_weights = compare_weights or weights
            elif selected_model == "baseline":
                selected_weights = weights
            else:
                raise gr.Error("Lựa chọn model không hợp lệ.")
            try:
                details = recognize_detailed(state, selected_weights)
                predictions = {field: item["raw"] for field, item in details.items()}
            except ValueError as exc:
                raise gr.Error(str(exc)) from exc
            except Exception as exc:
                raise gr.Error(f"Không chạy được VietOCR: {exc}") from exc
            updated = dict(state)
            updated["predictions"] = predictions
            updated["prediction_details"] = details
            updated["model_metadata"] = model_identity(selected_weights)
            updated["model_name"] = (updated["model_metadata"].get("name", f"VietOCR vgg_transformer · {Path(selected_weights).name}") if selected_weights
                                     else "VietOCR vgg_transformer · gốc")
            notes = []
            for field, item in details.items():
                valid, message = validate_field(field, item["value"])
                warnings = item["warnings"] + ([message] if not valid else [])
                if warnings:
                    note = f"**{FIELD_LABELS[field]}**: " + "; ".join(warnings)
                    if item["suggestion"]:
                        note += f". Gợi ý để đối chiếu ảnh: **{item['suggestion']}**."
                    notes.append(note)
            result = {
                prepared_state: updated,
                status: f"Đã chạy {updated['model_name']}. Kiểm tra và sửa chữ trước khi xuất JSON.",
                json_view: None,
                json_file: None,
                recognition_note: "\n\n".join(notes) or "Hãy đối chiếu cả bốn kết quả với ảnh trước khi xuất JSON.",
            }
            for field in FIELD_ORDER:
                result[text_boxes[field]] = details[field]["value"]
            return result

        ocr_button.click(
            on_ocr, inputs=[prepared_state, model_choice, *confirms.values()],
            outputs=[prepared_state, status, json_view, json_file, recognition_note, *text_boxes.values()],
        )

        def on_export(state, *texts):
            try:
                payload = result_payload(state, dict(zip(FIELD_ORDER, texts)), weights)
            except ValueError as exc:
                raise gr.Error(str(exc)) from exc
            output_dir = Path(tempfile.mkdtemp(prefix="vietocr_demo_"))
            output_file = output_dir / "result.json"
            output_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            invalid = [FIELD_LABELS[key] for key, item in payload["fields"].items() if not item["valid"]]
            message = ("Đã xuất JSON. Cần kiểm tra lại: " + ", ".join(invalid) + ".") if invalid else "Đã xuất JSON; bốn trường hợp lệ theo kiểm tra cơ bản."
            return payload, str(output_file), message

        export_button.click(
            on_export, inputs=[prepared_state, *text_boxes.values()], outputs=[json_view, json_file, status],
        )

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", help="Path to VietOCR .pth weights or field-bundle .json")
    parser.add_argument("--compare-weights", help="Fine-tuned checkpoint/bundle selectable beside the original model")
    parser.add_argument("--default-model", choices=["baseline", "fine_tuned"], default="baseline")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    if args.weights and args.compare_weights:
        parser.error("Chỉ dùng một trong --weights hoặc --compare-weights")
    weights = os.path.abspath(args.weights) if args.weights else None
    compare_weights = os.path.abspath(args.compare_weights) if args.compare_weights else None
    if weights and not Path(weights).is_file():
        parser.error(f"Không tìm thấy file trọng số: {weights}")
    if compare_weights and not Path(compare_weights).is_file():
        parser.error(f"Không tìm thấy file trọng số: {compare_weights}")
    for artifact in (weights, compare_weights):
        if artifact and Path(artifact).suffix.lower() == ".json":
            try:
                load_bundle(artifact)
            except ValueError as exc:
                parser.error(str(exc))
    app = build_app(weights, compare_weights, args.default_model)
    app.launch(server_name="127.0.0.1", server_port=args.port, share=False, show_error=True)


if __name__ == "__main__":
    main()
