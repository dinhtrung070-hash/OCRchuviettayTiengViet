"""Exercise the running Gradio crop/review/OCR/export flow on local samples."""

from __future__ import annotations

import json
import csv
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
    cases = [root / "phieu_da_viet.png", root / "phieudaviet1.png", root / "forms/dataset_20261102/1.jpg", root / "forms/dataset_20261102/50.jpg"]
    with (root / "pilot_labels.csv").open(newline="", encoding="utf-8-sig") as handle:
        pilot = {row["form"]: row for row in csv.DictReader(handle)}
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
        active = root / "models/extended1_bundle.json"
        if active.is_file():
            selected = json.loads(active.read_text(encoding="utf-8"))
            assert payload["model"] == selected["name"]
            for field, item in payload["fields"].items():
                expected = selected["fields"].get(field, selected["base"])
                assert item["recognition"]["model_metadata"]["sha256"] == expected["sha256"]
                policy = selected.get("category_policies", {}).get(field)
                assert item["recognition"].get("category_policy") == policy
        else:
            selected = json.loads((root / "models/extended1_selected.json").read_text(encoding="utf-8"))
            assert payload["model_metadata"]["sha256"] == selected["sha256"]
        assert all("ocr_raw" in item and "recognition" in item for item in payload["fields"].values())
        if path.name in pilot:
            assert values == tuple(pilot[path.name][field] for field in payload["fields"]), values
        records.append({"form": path.name, "predictions": dict(zip(payload["fields"], values)), "payload": payload})
        print(path.name, values, flush=True)
        previous = root / "models/extended1_v3_previous_bundle.json"
        if previous.is_file() and path.name in ("1.jpg", "50.jpg"):
            old = json.loads(previous.read_text(encoding="utf-8"))
            result = client.predict("previous", True, True, True, True, api_name="/on_ocr")
            old_payload, _, _ = client.predict(*result[-4:], api_name="/on_export")
            assert old_payload["model"] == old["name"]
            for field, item in old_payload["fields"].items():
                expected = old["fields"].get(field, old["base"])
                assert item["recognition"]["model_metadata"]["sha256"] == expected["sha256"]
            records[-1]["previous_v3"] = {"predictions": dict(zip(old_payload["fields"], result[-4:])),
                                          "model_metadata": old_payload["model_metadata"]}
            print("v3 comparison", path.name, result[-4:], flush=True)
        previous_v4 = root / "models/extended1_v4_previous_bundle.json"
        if previous_v4.is_file() and path.name == "50.jpg" and selected.get("name") != "VietOCR Extended 1 v4":
            old = json.loads(previous_v4.read_text(encoding="utf-8"))
            result = client.predict("previous_v4", True, True, True, True, api_name="/on_ocr")
            old_payload, _, _ = client.predict(*result[-4:], api_name="/on_export")
            assert old_payload["model"] == old["name"]
            assert all("category_policy" not in item["recognition"] for item in old_payload["fields"].values())
            records[-1]["previous_v4"] = {"predictions": dict(zip(old_payload["fields"], result[-4:])),
                                          "model_metadata": old_payload["model_metadata"]}
            print("v4 comparison", path.name, result[-4:], flush=True)
    # Test manual crop and confirm that stale OCR can no longer be exported.
    client.predict(api_name="/begin_edit")
    editor = {"background": handle_file(crops[0]), "layers": [], "composite": handle_file(crops[0])}
    applied = client.predict(editor, True, True, True, True, api_name="/on_apply")
    expect_error(lambda: client.predict(*values, api_name="/on_export"), "chạy VietOCR")
    result = client.predict("baseline", True, True, True, True, api_name="/on_ocr")
    payload, _, _ = client.predict(*result[-4:], api_name="/on_export")
    assert payload["model"].endswith("gốc")
    report_name = "smoke_latest.json" if (root / "models/extended1_bundle.json").is_file() else "smoke_v2.json"
    (root / "models" / report_name).write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print("PASS: upload, crop versions, confirmation gate, model selection, manual crop, JSON export.")


if __name__ == "__main__":
    main()
