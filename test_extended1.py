"""Regression checks for stroke preservation, formatting and review/export state."""

import unittest
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from field_inference import normalize_value
from structured_decode import propose, valid_date
from handwriting_preprocess import remove_isolated_dots
from ocr_review_app import (BASE_DIR, prepare_form, result_payload, select_crop_version,
                            validate_field)


class Extended1Tests(unittest.TestCase):
    def test_dot_removal_keeps_crossing_stroke_and_accent(self):
        image = np.full((100, 240), 255, np.uint8)
        for x in range(20, 231, 10):
            cv2.circle(image, (x, 70), 2, 0, -1)
        cv2.line(image, (100, 30), (100, 82), 0, 4)
        cv2.line(image, (100, 40), (160, 40), 0, 4)
        cv2.circle(image, (132, 22), 2, 0, -1)
        cleaned, _ = remove_isolated_dots(image, None)
        self.assertEqual(int(cleaned[70, 100]), 0)
        self.assertEqual(int(cleaned[40, 130]), 0)
        self.assertEqual(int(cleaned[22, 132]), 0)
        self.assertEqual(int(cleaned[70, 220]), 255)

    def test_date_formatting_does_not_guess_digits(self):
        self.assertEqual(normalize_value("ngay_sinh", "9.7.2003"), "09/07/2003")
        self.assertEqual(normalize_value("ngay_sinh", "M/M/2002"), "M/M/2002")
        self.assertEqual(normalize_value("ngay_sinh", "30/02/2003"), "30/02/2003")
        self.assertFalse(validate_field("ngay_sinh", "30/02/2003")[0])
        self.assertFalse(validate_field("ngay_sinh", "01/01/2999")[0])

    def test_structured_decoder_never_uses_a_name_dictionary(self):
        self.assertIsNone(propose(None, None, None, "ho_ten", "Some Name", "Some Name", None))
        self.assertFalse(valid_date("31/02/2003"))
        self.assertFalse(valid_date("01/01/2999"))

    def test_switch_crop_invalidates_old_prediction(self):
        raw = Image.new("RGB", (100, 40), "white")
        clean = Image.new("RGB", (70, 35), "white")
        state = {"raw_crops": {"ho_ten": raw}, "clean_crops": {"ho_ten": clean},
                 "clean_methods": {"ho_ten": "clean"}, "methods": {"ho_ten": "clean"},
                 "crops": {"ho_ten": clean}, "predictions": {"ho_ten": "old"},
                 "prediction_details": {"ho_ten": {"raw": "old"}}}
        changed = select_crop_version(state, "ho_ten", False)
        self.assertIs(changed["crops"]["ho_ten"], raw)
        self.assertNotIn("ho_ten", changed["predictions"])
        self.assertEqual(state["predictions"]["ho_ten"], "old")

    def test_export_keeps_raw_text_and_user_correction(self):
        labels = {"ho_ten": "Nguyễn Văn Minh", "ngay_sinh": "9.7.2003", "dan_toc": "Kinh", "ton_giao": "Không"}
        state = {"form_name": "test.png", "predictions": labels,
                 "methods": dict.fromkeys(labels, "test"), "prediction_details": {},
                 "model_name": "test-model", "preprocess_version": "test"}
        edited = dict(labels, ngay_sinh="09/07/2003", ho_ten="Nguyễn Văn Bình")
        payload = result_payload(state, edited, None)
        self.assertEqual(payload["fields"]["ho_ten"]["ocr_raw"], "Nguyễn Văn Minh")
        self.assertEqual(payload["fields"]["ho_ten"]["confirmed"], "Nguyễn Văn Bình")
        self.assertEqual(payload["fields"]["ngay_sinh"]["formatted"], "09/07/2003")
        self.assertEqual(payload["model"], "test-model")

    def test_generated_data_matches_live_preprocessing(self):
        root = BASE_DIR / "prepared" / "extended1_v2"
        if not root.is_dir():
            self.skipTest("Local dataset is not included in the repository")
        for number in (1, 50):
            source = BASE_DIR / "forms" / "dataset_20261102" / f"{number}.jpg"
            live = prepare_form(str(source))
            for field, image in live["crops"].items():
                with Image.open(root / str(number) / f"{field}.png") as trained:
                    self.assertTrue(np.array_equal(np.asarray(image), np.asarray(trained.convert("RGB"))),
                                    f"Train/inference crop differs: {number}/{field}")

    def test_source_forms_are_disjoint_across_splits(self):
        root = BASE_DIR / "prepared" / "extended1_v2"
        if not root.is_dir():
            self.skipTest("Local dataset is not included in the repository")
        groups = {}
        for split in ("train", "valid", "test"):
            paths = [line.split("\t")[0] for line in (root / f"{split}_annotation.txt").read_text(encoding="utf-8").splitlines()]
            groups[split] = {Path(path).parts[0] for path in paths}
            self.assertTrue(all((root / path).is_file() for path in paths))
        self.assertFalse(groups["train"] & groups["valid"])
        self.assertFalse(groups["train"] & groups["test"])
        self.assertFalse(groups["valid"] & groups["test"])


if __name__ == "__main__":
    unittest.main()
