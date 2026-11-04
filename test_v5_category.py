"""Selective proposals must preserve raw OCR and abstain on weak evidence."""

import math
import unittest
import tempfile
import json
import hashlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from PIL import Image
from vietocr.model.vocab import Vocab

from field_inference import predict_field, text_distance
from structured_decode import propose
from v5_category_decode import propose_category
from field_bundle import category_policy, load_bundle
from ocr_review_app import BASE_DIR


POLICY = {"threshold": 0.35980080941540404, "odds": 2}


class CategoryPolicyTests(unittest.TestCase):
    def setUp(self):
        self.options = patch("v5_category_decode.vocabularies", return_value={"dan_toc": ["Tày", "Lào"]})
        self.options.start()
        self.addCleanup(self.options.stop)

    @staticmethod
    def scores(probabilities):
        return lambda model, memory, vocab, forms: [math.log(probabilities.get(form, .01)) for form in forms]

    def test_case_evidence_can_support_the_same_canonical_class(self):
        with patch("v5_category_decode.string_scores", side_effect=self.scores({"Tày": .2, "tày": .7, "Lào": .2})):
            result = propose_category(None, None, None, "dan_toc", "lày", "lày", text_distance, POLICY)
        self.assertEqual(result["value"], "Tày")
        self.assertEqual(result["source"], "field_vocabulary_v5")
        self.assertIn("xác nhận", result["warning"])

    def test_absolute_score_and_competing_class_can_both_require_abstention(self):
        for probabilities in ({"Tày": .3, "Lào": .01}, {"Tày": .7, "Lào": .6}):
            with self.subTest(probabilities=probabilities), patch("v5_category_decode.string_scores", side_effect=self.scores(probabilities)):
                result = propose_category(None, None, None, "dan_toc", "lày", "lày", text_distance, POLICY)
            self.assertIsNone(result)

    def test_known_schema_value_is_not_changed(self):
        with patch("v5_category_decode.string_scores") as scores:
            result = propose_category(None, None, None, "dan_toc", "tày", "tày", text_distance, POLICY)
        self.assertIsNone(result)
        scores.assert_not_called()

    def test_bundle_policy_is_isolated_and_cannot_target_names(self):
        workspace = BASE_DIR / "prepared"
        workspace.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=workspace) as temporary:
            root = Path(temporary).resolve()
            self.assertTrue(root.is_relative_to(BASE_DIR.resolve()))
            (root / "base.pth").write_bytes(b"test-checkpoint")
            base = {"format": "vietocr_field_bundle_v1", "base": {"weights": "base.pth",
                    "sha256": hashlib.sha256(b"test-checkpoint").hexdigest()}, "fields": {}}
            old_path, new_path = root / "old.json", root / "new.json"
            old_path.write_text(json.dumps(base), encoding="utf-8")
            new = dict(base, category_policies={"ton_giao": POLICY})
            new_path.write_text(json.dumps(new), encoding="utf-8")
            self.assertIsNone(category_policy(str(old_path), "ton_giao"))
            self.assertIsNone(category_policy(str(new_path), "ho_ten"))
            returned = category_policy(str(new_path), "ton_giao")
            returned["threshold"] = .99
            self.assertEqual(category_policy(str(new_path), "ton_giao"), POLICY)
            for index, policies in enumerate(({"ho_ten": POLICY}, {"ton_giao": {"threshold": float("nan"), "odds": 2}},
                                               {"ton_giao": {"threshold": True, "odds": 2}})):
                invalid = root / f"bad{index}.json"
                invalid.write_text(json.dumps(dict(base, category_policies=policies)), encoding="utf-8")
                with self.assertRaises(ValueError):
                    load_bundle(str(invalid))

    def test_legacy_default_and_names_never_dispatch_to_the_new_policy(self):
        with patch("v5_category_decode.propose_category") as helper:
            self.assertIsNone(propose(None, None, None, "dan_toc", "Tày", "Tày", text_distance))
            self.assertIsNone(propose(None, None, None, "ho_ten", "Một Họ Tên", "Một Họ Tên", text_distance, POLICY))
        helper.assert_not_called()

    def test_structured_proposal_preserves_raw_text_and_records_its_policy(self):
        vocab = Vocab("làyT")
        emitted = [vocab.c2i[c] for c in "lày"] + [vocab.eos]

        class Transformer:
            def forward_encoder(self, features):
                return features

            def forward_decoder(self, tokens, memory):
                length = tokens.shape[0]
                logits = torch.full((1, length, len(vocab)), -15.)
                logits[0, -1, emitted[length-1]] = 15.
                return logits, memory

        model = SimpleNamespace(eval=lambda: None, cnn=lambda picture: torch.zeros((1, 1, 1)), transformer=Transformer())
        predictor = SimpleNamespace(model=model, vocab=vocab,
                                    config={"dataset": {"image_height": 32, "image_min_width": 32, "image_max_width": 512}, "device": "cpu"})
        proposal = {"value": "Tày", "score": .7, "source": "field_vocabulary_v5", "warning": "Cần xác nhận lại với ảnh"}
        with patch("structured_decode.propose", return_value=proposal) as correction:
            result = predict_field(predictor, Image.new("RGB", (96, 32), "white"), "dan_toc",
                                   structured=True, category_policy=POLICY)
        self.assertEqual(result["raw"], "lày")
        self.assertEqual(result["value"], "Tày")
        self.assertEqual(result["category_policy"], POLICY)
        self.assertIn(proposal["warning"], result["warnings"])
        self.assertEqual(correction.call_args.kwargs["category_policy"], POLICY)
        with patch("structured_decode.propose") as correction:
            unstructured = predict_field(predictor, Image.new("RGB", (96, 32), "white"), "dan_toc", structured=False)
        correction.assert_not_called()
        self.assertEqual(unstructured["raw"], unstructured["value"])
        self.assertNotIn("category_policy", unstructured)


if __name__ == "__main__":
    unittest.main()
