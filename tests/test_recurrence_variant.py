# SPDX-License-Identifier: MIT OR Apache-2.0
"""The committed exp-026 variant bundle ``dataset/merged_v2_rec_v1`` (GH #3).

Attestation, byte-identity of the hidden images with ``dataset/merged_v2``,
agreement between the bundle and the recorded ``comparison.json``, the
feed-forward consumers' refusal of a recurrent checkpoint, and the variant's
recurrent replay on the method fixture. The contract and tool tests live in
``tests/test_recurrence.py``.
"""

from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from tools.hamming_banks import bank_from_json
from tools.measure_hamming import main as hamming_main
from tools.model_bank import load_model_bank
from tools.q88_core import ParseError, encode_q88_hex, load_model, parse_mem
from tools.recurrence_bundle import UNCHANGED_MEM_IMAGES
from tools.recurrence_eval import FIXTURE_JSONL
from tools.recurrent_lif import (
    RECURRENCE_SCHEMA,
    RECURRENT_MEM_NAME,
    recurrence_from_model,
    recurrent_mem_words,
    require_feed_forward,
)
from tools.replay_core import REPO_ROOT
from tools.replay_frozen import main as replay_main

SHIPPED_DIR = REPO_ROOT / "dataset" / "merged_v2"
SHIPPED_MANIFEST = SHIPPED_DIR / "model_bank.json"
VARIANT_DIR = REPO_ROOT / "dataset" / "merged_v2_rec_v1"
VARIANT_MANIFEST = VARIANT_DIR / "model_bank.json"
RECORDED_COMPARISON = REPO_ROOT / "docs" / "recurrence" / "exp-026" / "comparison.json"


class TestRecordedVariantBundle(unittest.TestCase):
    """The committed ``dataset/merged_v2_rec_v1`` bundle and its comparison."""

    def test_bundle_attests_and_declares_recurrence(self):
        bank = load_model_bank(VARIANT_MANIFEST)
        entry = bank.select("merged_v2_rec_v1")
        model = json.loads(entry.checkpoint_bytes.decode("utf-8"))
        rec = recurrence_from_model(model, "variant")
        self.assertIsNotNone(rec)
        self.assertEqual(rec.nonzero(), 48)
        shipped = load_model_bank(SHIPPED_MANIFEST).select("merged_v2")
        self.assertEqual(
            model["variant"]["derived_from"]["checkpoint_digest"], shipped.checkpoint_digest
        )
        self.assertIsNotNone(entry.training_dataset_digest)

    def test_hidden_images_identical_to_shipped(self):
        for name in UNCHANGED_MEM_IMAGES:
            self.assertEqual(
                [e.text.upper() for e in parse_mem(VARIANT_DIR / name)],
                [e.text.upper() for e in parse_mem(SHIPPED_DIR / name)],
                name,
            )

    def test_output_and_recurrent_images_match_checkpoint(self):
        model = json.loads((VARIANT_DIR / "snn_model.json").read_text(encoding="utf-8"))
        rec = recurrence_from_model(model, "variant")
        self.assertEqual(
            [e.text.upper() for e in parse_mem(VARIANT_DIR / RECURRENT_MEM_NAME)],
            recurrent_mem_words(rec),
        )
        self.assertEqual(
            [e.text.upper() for e in parse_mem(VARIANT_DIR / "parameters_output_weights.mem")],
            [encode_q88_hex(w) for n in model["neurons"] for w in n["output_weights"]],
        )

    def test_recorded_comparison_matches_bundle(self):
        doc = json.loads(RECORDED_COMPARISON.read_text(encoding="utf-8"))
        model = json.loads((VARIANT_DIR / "snn_model.json").read_text(encoding="utf-8"))
        selected = doc["selection"]["selected_rec_arm"]
        self.assertEqual(doc["selection"]["selected_gain"], model["variant"]["gain"])
        self.assertEqual(
            doc["arms"][selected]["readout"]["weights_neuron_major"],
            [w for n in model["neurons"] for w in n["output_weights"]],
        )
        shipped = load_model_bank(SHIPPED_MANIFEST).select("merged_v2")
        self.assertEqual(doc["model"]["checkpoint_digest"], shipped.checkpoint_digest)
        self.assertEqual(doc["input"]["splits"]["test"]["n_ticks"], 117653)
        self.assertEqual(
            doc["input"]["splits"]["test"]["sessions"][0], "gpu-000170"
        )
        self.assertEqual(
            doc["input"]["splits"]["test"]["sessions"][-1], "gpu-000198"
        )

    def test_feed_forward_consumers_refuse_the_variant(self):
        # The contract: a checkpoint that declares recurrence is never
        # stepped feed-forward with the block dropped. The Hamming harness
        # is the one consumer that takes an arbitrary float checkpoint.
        variant = REPO_ROOT / "dataset" / "merged_v2_rec_v1"
        model = json.loads((variant / "snn_model.json").read_text(encoding="utf-8"))
        with self.assertRaises(ParseError):
            require_feed_forward(model, "variant")
        feed_forward = load_model(SHIPPED_DIR / "snn_model.json")
        self.assertIs(require_feed_forward(feed_forward), feed_forward)
        with self.assertRaises(ParseError):
            bank_from_json(variant / "snn_model.json", "float-json", 0.05)
        argv = [
            "--condition", "shipped-merged-v2", "--jsonl", str(FIXTURE_JSONL),
            "--float-json", str(variant / "snn_model.json"), "--mem-dir", str(variant),
        ]
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            code = hamming_main(argv)
        self.assertEqual(code, 2)

    def test_variant_replays_on_fixture(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(
                replay_main(["--bank-manifest", str(VARIANT_MANIFEST), "--out-dir", tmp]), 0
            )
            manifest = json.loads((Path(tmp) / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["stepper"]["recurrence"]["schema"], RECURRENCE_SCHEMA)


if __name__ == "__main__":
    unittest.main()
