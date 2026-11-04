"""Exercise the running Gradio crop/review/OCR/export flow on local samples."""

from __future__ import annotations

import json
from pathlib import Path

from gradio_client import Client, handle_file


def expect_error(call, text: str) -> None:
    try:
        call()
    except Exception as exc:
        if text not in str(exc):
            raise
    else:
        raise AssertionError("Action should have been rejected")


def main() -> None:
    root = Path(__file__).resolve().parent
    client = Client("http://127.0.0.1:7861/", verbose=False)
    expect_error(lambda: client.predict("fine_tuned", True, True, True, True, api_name="/on_ocr"), "kiểm tra crop")
    cases = [root / "phieudaviet1.png", root / "forms/dataset_20261102/1.jpg", root / "forms/dataset_20261102/50.jpg"]
    records = []
    for path in cases:
        if not path.is_file():
            continue
        prepared = client.predict(handle_file(str(path)), api_name="/on_prepare")
        crops = prepared[7:11]
        assert len(crops) == 4 and all(Path(path).is_file() for path in crops)
        client.predict(False, api_name="/on_clean_switch")
        client.predict(True, api_name="/on_clean_switch")
        expect_error(lambda: client.predict("fine_tuned", False, True, True, True, api_name="/on_ocr"), "xác nhận")
        result = client.predict("fine_tuned", True, True, True, True, api_name="/on_ocr")
        values = result[-4:]
        payload, file_path, _ = client.predict(*values, api_name="/on_export")
        assert Path(file_path).is_file()
        assert payload["preprocess_version"] == "stroke_preserving_v2"
        assert "extended1_selected.pth" in payload["model"]
        selected = json.loads((root / "models/extended1_selected.json").read_text(encoding="utf-8"))
        assert payload["model_metadata"]["sha256"] == selected["sha256"]
        assert all("ocr_raw" in item and "recognition" in item for item in payload["fields"].values())
        if path.name == "phieudaviet1.png":
            assert values == ("Bằng Văn Vũ", "12/03/1991", "Tày", "Không"), values
        records.append({"form": path.name, "predictions": dict(zip(payload["fields"], values)), "payload": payload})
        print(path.name, values, flush=True)
    # Test manual crop and confirm that stale OCR can no longer be exported.
    client.predict(api_name="/begin_edit")
    editor = {"background": handle_file(crops[0]), "layers": [], "composite": handle_file(crops[0])}
    applied = client.predict(editor, True, True, True, True, api_name="/on_apply")
    expect_error(lambda: client.predict(*values, api_name="/on_export"), "chạy VietOCR")
    result = client.predict("baseline", True, True, True, True, api_name="/on_ocr")
    payload, _, _ = client.predict(*result[-4:], api_name="/on_export")
    assert payload["model"].endswith("gốc")
    (root / "models" / "smoke_v2.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print("PASS: upload, crop versions, confirmation gate, both models, manual crop, JSON export.")


if __name__ == "__main__":
    main()
