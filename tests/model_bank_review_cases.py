"""Focused regression cases added while closing model-bank review threads."""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from tools.model_bank import (
    MANIFEST_FILENAME,
    BankAttestationError,
    BankParseError,
    dumps_manifest,
    load_model_bank,
    load_unattested_checkpoint,
)

FIXTURE_ROOT = Path(__file__).parents[1] / "tools" / "fixtures" / "model_bank"
VALID = FIXTURE_ROOT / "valid" / MANIFEST_FILENAME


class ModelBankReviewTests(unittest.TestCase):
    def test_unattested_checkpoint_rejects_invalid_utf8(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            broken = Path(tmp) / "not-utf8.json"
            broken.write_bytes(b"\xff\xfe not utf-8")
            with self.assertRaises(BankParseError) as caught:
                load_unattested_checkpoint(broken)
            self.assertIn("is not UTF-8", str(caught.exception))

    def test_overflowing_json_float_is_parse_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "overflow.json"
            checkpoint.write_text('{"weight": 1e400}', encoding="utf-8")
            with self.assertRaises(BankParseError) as caught:
                load_unattested_checkpoint(checkpoint)
            self.assertIn("unreadable JSON number", str(caught.exception))

    def test_required_identifiers_reject_whitespace_only_values(self) -> None:
        for field in (
            "id",
            "feature_map_id",
            "output_contract_id",
            "numeric_format",
        ):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as tmp:
                dest = Path(tmp) / "whitespace"
                shutil.copytree(VALID.parent, dest)
                manifest = dest / MANIFEST_FILENAME
                document = json.loads(manifest.read_text(encoding="utf-8"))
                document["models"][0][field] = "   "
                manifest.write_text(dumps_manifest(document), encoding="utf-8")
                with self.assertRaises(BankAttestationError) as caught:
                    load_model_bank(manifest)
                self.assertEqual(caught.exception.field, field)
                self.assertIn("non-whitespace", str(caught.exception))
