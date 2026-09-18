"""Model-bank attestation tests (Linear RM-1327).

Run::

    pytest -q -s
    python3 -m unittest tests.test_model_bank -v
    python3 tools/verify_model_bank.py
    python3 tools/verify_model_bank.py --self-test

``pytest -q -s`` keeps the computed digest lines in the test output.
Standard library only; pytest is an optional runner.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path

from tools.model_bank import (
    FEATURE_MAP_LIVE_EXP_025,
    MANIFEST_FILENAME,
    NUMERIC_FORMAT_Q88,
    OUTPUT_CONTRACT_SUPERVISOR_V3_RM1150,
    REPO_ROOT,
    BankAttestationError,
    BankParseError,
    digest_file,
    dumps_manifest,
    load_model_bank,
    load_shipped_merged_v2_bank,
    load_unattested_checkpoint,
    wrap_legacy_checkpoint,
)
from tools.verify_model_bank import main

FIXTURE_ROOT = REPO_ROOT / "tools" / "fixtures" / "model_bank"
VALID = FIXTURE_ROOT / "valid" / MANIFEST_FILENAME
TAMPERED = FIXTURE_ROOT / "tampered-checkpoint" / MANIFEST_FILENAME
UNSUPPORTED = FIXTURE_ROOT / "unsupported-version" / MANIFEST_FILENAME
SHIPPED_CHECKPOINT = REPO_ROOT / "dataset" / "merged_v2" / "snn_model.json"
FIXTURE_DIGEST = (
    "sha256:bd0133a6225aa377e9c68668b7abd4d69a1dc670877ad01be5629befa7d10111"
)
SHIPPED_DIGEST = (
    "sha256:cf3b7a47c5eb62d93b28480804e2ca82923db5bece1285bcbeba723437bfaf89"
)


class ModelBankTests(unittest.TestCase):
    def test_valid_fixture_attests_and_selects(self) -> None:
        bank = load_model_bank(VALID)
        entry = bank.select("fixture-ok")
        print(f"computed checkpoint digest: {entry.checkpoint_digest}")
        self.assertEqual(entry.checkpoint_digest, FIXTURE_DIGEST)
        self.assertEqual(bank.ids(), ("fixture-ok",))
        self.assertEqual(entry.feature_map_id, FEATURE_MAP_LIVE_EXP_025)
        self.assertEqual(
            entry.output_contract_id, OUTPUT_CONTRACT_SUPERVISOR_V3_RM1150
        )
        self.assertEqual(entry.numeric_format, NUMERIC_FORMAT_Q88)
        self.assertIsNone(entry.training_dataset_digest)
        self.assertEqual(
            f"sha256:{hashlib.sha256(entry.checkpoint_bytes).hexdigest()}",
            entry.checkpoint_digest,
        )
        with self.assertRaises(BankAttestationError) as caught:
            bank.select("ghost")
        self.assertEqual(caught.exception.field, "id")
        self.assertIn("ghost", str(caught.exception))

    def test_manifest_serialization_is_stable(self) -> None:
        raw_bytes = VALID.read_bytes()
        raw = raw_bytes.decode("utf-8")
        dumped = dumps_manifest(json.loads(raw))
        self.assertEqual(raw_bytes, dumped.encode("utf-8"))
        self.assertTrue(dumped.endswith("\n"))
        self.assertNotIn(b"\r", raw_bytes)

    def test_select_consumes_attested_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "toctou"
            shutil.copytree(VALID.parent, dest)
            bank = load_model_bank(dest / MANIFEST_FILENAME)
            loaded = bank.entries[0]
            consumed = loaded.checkpoint_bytes
            self.assertEqual(
                f"sha256:{hashlib.sha256(consumed).hexdigest()}",
                loaded.checkpoint_digest,
            )
            loaded.checkpoint.write_bytes(b"replaced-after-load\n")
            entry = bank.select("fixture-ok")
            self.assertEqual(entry.checkpoint_bytes, consumed)
            self.assertEqual(
                f"sha256:{hashlib.sha256(entry.checkpoint_bytes).hexdigest()}",
                entry.checkpoint_digest,
            )

    def test_nul_loader_paths_are_parse_errors(self) -> None:
        with self.assertRaises(BankParseError) as caught_manifest:
            load_model_bank("model_bank.json\x00")
        # resolve() runs first; pathlib raises ValueError for an embedded NUL.
        self.assertRegex(
            str(caught_manifest.exception), r"cannot read|cannot resolve"
        )
        with self.assertRaises(BankParseError) as caught_ckpt:
            load_unattested_checkpoint("snn_model.json\x00")
        self.assertIn("cannot read", str(caught_ckpt.exception))

    def test_deeply_nested_json_is_parse_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp)
            nested = '{"a":' * 10000 + "1" + "}" * 10000
            manifest = dest / MANIFEST_FILENAME
            manifest.write_text(nested, encoding="utf-8")
            try:
                load_model_bank(manifest)
            except BankParseError as exc:
                self.assertIn("JSON nesting exceeds parser limit", str(exc))
            except BankAttestationError as exc:
                # CPython 3.14+ json.loads can accept this depth; extra key
                # then fails attestation instead of parse.
                self.assertEqual(exc.field, "a")
            else:
                self.fail("deeply nested manifest was accepted")
            checkpoint = dest / "nested.json"
            checkpoint.write_text(nested, encoding="utf-8")
            try:
                load_unattested_checkpoint(checkpoint)
            except BankParseError as exc:
                self.assertIn(
                    "JSON nesting exceeds parser limit", str(exc)
                )
            else:
                self.fail("deeply nested checkpoint was accepted")

    def test_control_character_token_is_attestation_error(self) -> None:
        cases = (
            ("LF", "fixture\nok"),
            ("NEL", "fixture\u0085ok"),
            ("CSI", "fixture\u009bok"),
        )
        for name, token in cases:
            with self.subTest(name=name):
                with tempfile.TemporaryDirectory() as tmp:
                    dest = Path(tmp) / name
                    shutil.copytree(VALID.parent, dest)
                    document = json.loads(
                        (dest / MANIFEST_FILENAME).read_text(encoding="utf-8")
                    )
                    document["models"][0]["id"] = token
                    (dest / MANIFEST_FILENAME).write_text(
                        dumps_manifest(document), encoding="utf-8"
                    )
                    with self.assertRaises(BankAttestationError) as caught:
                        load_model_bank(dest / MANIFEST_FILENAME)
                    self.assertEqual(caught.exception.field, "id")
                    self.assertIn("control character", str(caught.exception))

    def test_unknown_control_key_is_escaped_on_cli(self) -> None:
        cases = (
            ("ESC", "extra\u001b[2J", "\\x1b"),
            ("LF", "extra\nforged-status", "\\n"),
        )
        for name, key, escaped in cases:
            with self.subTest(name=name):
                with tempfile.TemporaryDirectory() as tmp:
                    dest = Path(tmp) / name
                    shutil.copytree(VALID.parent, dest)
                    document = json.loads(
                        (dest / MANIFEST_FILENAME).read_text(encoding="utf-8")
                    )
                    document[key] = True
                    (dest / MANIFEST_FILENAME).write_text(
                        dumps_manifest(document), encoding="utf-8"
                    )
                    stderr = io.StringIO()
                    with redirect_stderr(stderr):
                        code = main([str(dest / MANIFEST_FILENAME)])
                    self.assertEqual(code, 1)
                    text = stderr.getvalue()
                    self.assertNotIn("\x1b", text)
                    self.assertNotIn("\nforged-status", text)
                    self.assertIn(escaped, text)
                    self.assertIn("unsupported extra field", text)

    def test_nonstandard_json_constants_are_parse_errors(self) -> None:
        for token in ("NaN", "Infinity", "-Infinity"):
            with self.subTest(token=token):
                with tempfile.TemporaryDirectory() as tmp:
                    dest = Path(tmp)
                    manifest = dest / MANIFEST_FILENAME
                    manifest.write_text(
                        f'{{"schema_version": {token}, "models": []}}',
                        encoding="utf-8",
                    )
                    with self.assertRaises(BankParseError) as caught:
                        load_model_bank(manifest)
                    self.assertIn("non-standard JSON constant", str(caught.exception))
                    self.assertIn(token, str(caught.exception))
                    checkpoint = dest / "const.json"
                    checkpoint.write_text(f'{{"v": {token}}}', encoding="utf-8")
                    with self.assertRaises(BankParseError) as caught_ckpt:
                        load_unattested_checkpoint(checkpoint)
                    self.assertIn(
                        "non-standard JSON constant", str(caught_ckpt.exception)
                    )

    def test_dumps_manifest_rejects_nonstandard_constants(self) -> None:
        for value in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(value=value):
                with self.assertRaises(BankParseError) as caught:
                    dumps_manifest({"v": value})
                self.assertIn("cannot serialize manifest", str(caught.exception))

    def test_dumps_manifest_rejects_non_json_values(self) -> None:
        with self.assertRaises(BankParseError) as caught:
            dumps_manifest({"v": {1}})
        self.assertIn("cannot serialize manifest", str(caught.exception))

    def test_dumps_manifest_rejects_excessive_nesting(self) -> None:
        value: object = 0
        for _ in range(10000):
            value = {"nested": value}
        try:
            dumps_manifest({"v": value})
        except BankParseError as exc:
            self.assertIn("cannot serialize manifest", str(exc))
        except RecursionError:
            self.fail("RecursionError leaked from dumps_manifest")

    def test_duplicate_json_object_key_is_parse_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "dupkeys"
            shutil.copytree(VALID.parent, dest)
            # First digest mismatches; second matches. CPython last-wins would
            # attest the matching digest and hide the conflicting key.
            poisoned = (
                "{\n"
                '  "models": [\n'
                "    {\n"
                '      "checkpoint": "checkpoints/ok.bin",\n'
                '      "checkpoint_digest":'
                ' "sha256:' + ("0" * 64) + '",\n'
                '      "checkpoint_digest": "' + FIXTURE_DIGEST + '",\n'
                '      "feature_map_id":'
                f' "{FEATURE_MAP_LIVE_EXP_025}",\n'
                '      "id": "fixture-ok",\n'
                '      "numeric_format":'
                f' "{NUMERIC_FORMAT_Q88}",\n'
                '      "output_contract_id":'
                f' "{OUTPUT_CONTRACT_SUPERVISOR_V3_RM1150}"\n'
                "    }\n"
                "  ],\n"
                '  "schema_version": 1\n'
                "}\n"
            )
            (dest / MANIFEST_FILENAME).write_text(poisoned, encoding="utf-8")
            with self.assertRaises(BankParseError) as caught:
                load_model_bank(dest / MANIFEST_FILENAME)
            self.assertIn("duplicate object key", str(caught.exception))
            self.assertIn("checkpoint_digest", str(caught.exception))
            checkpoint = dest / "dup.json"
            checkpoint.write_text(
                '{"neurons": [], "neurons": []}\n', encoding="utf-8"
            )
            with self.assertRaises(BankParseError) as caught_ckpt:
                load_unattested_checkpoint(checkpoint)
            self.assertIn("duplicate object key", str(caught_ckpt.exception))
            self.assertIn("neurons", str(caught_ckpt.exception))

    def test_oversized_json_integer_is_parse_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp)
            manifest = dest / MANIFEST_FILENAME
            manifest.write_text(
                '{"schema_version": 1, "models": [1' + "0" * 5000 + "]}",
                encoding="utf-8",
            )
            with self.assertRaises(BankParseError) as caught:
                load_model_bank(manifest)
            self.assertIn("unreadable JSON number", str(caught.exception))
            checkpoint = dest / "broken.json"
            checkpoint.write_text(
                '{"neurons": [1' + "0" * 5000 + "]}", encoding="utf-8"
            )
            with self.assertRaises(BankParseError) as caught_ckpt:
                load_unattested_checkpoint(checkpoint)
            self.assertIn("unreadable JSON number", str(caught_ckpt.exception))

    def test_digest_file_hashes_checkpoint_bytes(self) -> None:
        ckpt = VALID.parent / "checkpoints" / "ok.bin"
        self.assertEqual(digest_file(ckpt), FIXTURE_DIGEST)
        self.assertEqual(digest_file(SHIPPED_CHECKPOINT), SHIPPED_DIGEST)

    def test_digest_file_missing_is_attestation_error(self) -> None:
        missing = VALID.parent / "checkpoints" / "missing.bin"
        with self.assertRaises(BankAttestationError) as caught:
            digest_file(missing)
        self.assertEqual(caught.exception.field, "checkpoint")
        self.assertIsNone(caught.exception.entry)
        self.assertIn("cannot read file", str(caught.exception))

    def test_digest_file_nul_path_is_attestation_error(self) -> None:
        with self.assertRaises(BankAttestationError) as caught:
            digest_file(Path("ok.bin\x00"))
        self.assertEqual(caught.exception.field, "checkpoint")
        self.assertIn("cannot read file", str(caught.exception))

    def test_public_names_resolve(self) -> None:
        import tools.model_bank as model_bank

        for name in model_bank.__all__:
            self.assertTrue(hasattr(model_bank, name), name)

    def test_digest_file_unreadable_is_attestation_error(self) -> None:
        if os.name == "nt" or not hasattr(os, "geteuid"):
            self.skipTest("chmod(0) does not make files unreadable here")
        if os.geteuid() == 0:
            self.skipTest("root can read chmod 0 files")
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "unreadable.bin"
            dest.write_bytes(b"ok\n")
            mode = dest.stat().st_mode
            try:
                dest.chmod(0)
                with self.assertRaises(BankAttestationError) as caught:
                    digest_file(dest)
            finally:
                dest.chmod(mode)
            self.assertEqual(caught.exception.field, "checkpoint")
            self.assertIn("cannot read file", str(caught.exception))

    def test_unreadable_checkpoint_is_attestation_error(self) -> None:
        if os.name == "nt" or not hasattr(os, "geteuid"):
            self.skipTest("chmod(0) does not make files unreadable here")
        if os.geteuid() == 0:
            self.skipTest("root can read chmod 0 files")
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "unreadable"
            shutil.copytree(VALID.parent, dest)
            ckpt = dest / "checkpoints" / "ok.bin"
            mode = ckpt.stat().st_mode
            try:
                ckpt.chmod(0)
                with self.assertRaises(BankAttestationError) as caught:
                    load_model_bank(dest / MANIFEST_FILENAME)
            finally:
                ckpt.chmod(mode)
            self.assertEqual(caught.exception.entry, "fixture-ok")
            self.assertEqual(caught.exception.field, "checkpoint")
            self.assertIn("cannot read file", str(caught.exception))

    def test_nul_checkpoint_path_is_attestation_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "nul"
            shutil.copytree(VALID.parent, dest)
            document = json.loads((dest / MANIFEST_FILENAME).read_text(encoding="utf-8"))
            document["models"][0]["checkpoint"] = "check\x00points/ok.bin"
            (dest / MANIFEST_FILENAME).write_text(
                dumps_manifest(document), encoding="utf-8"
            )
            with self.assertRaises(BankAttestationError) as caught:
                load_model_bank(dest / MANIFEST_FILENAME)
            self.assertEqual(caught.exception.entry, "fixture-ok")
            self.assertEqual(caught.exception.field, "checkpoint")
            self.assertRegex(
                str(caught.exception),
                r"NUL|control character",
            )

    def test_manifest_symlink_loads_resolved_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            dest = tmp_path / "real"
            shutil.copytree(VALID.parent, dest)
            other = tmp_path / "other"
            other.mkdir()
            link = other / MANIFEST_FILENAME
            link.symlink_to(dest / MANIFEST_FILENAME)
            bank = load_model_bank(link)
            entry = bank.select("fixture-ok")
            self.assertEqual(entry.checkpoint_digest, FIXTURE_DIGEST)
            self.assertEqual(bank.manifest_path, (dest / MANIFEST_FILENAME).resolve())
            self.assertEqual(bank.root, dest.resolve())
            self.assertEqual(entry.checkpoint.parent, (dest / "checkpoints").resolve())
            self.assertEqual(
                f"sha256:{hashlib.sha256(entry.checkpoint_bytes).hexdigest()}",
                FIXTURE_DIGEST,
            )

    def test_symlink_loop_checkpoint_is_attestation_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "loop"
            shutil.copytree(VALID.parent, dest)
            loop = dest / "loop.bin"
            loop.symlink_to(loop)
            document = json.loads((dest / MANIFEST_FILENAME).read_text(encoding="utf-8"))
            document["models"][0]["checkpoint"] = "loop.bin"
            (dest / MANIFEST_FILENAME).write_text(
                dumps_manifest(document), encoding="utf-8"
            )
            with self.assertRaises(BankAttestationError) as caught:
                load_model_bank(dest / MANIFEST_FILENAME)
            self.assertEqual(caught.exception.entry, "fixture-ok")
            self.assertEqual(caught.exception.field, "checkpoint")
            # Python 3.11 raises during resolve(); 3.13+ may report missing file.
            self.assertRegex(
                str(caught.exception),
                r"cannot resolve path|missing file",
            )

    def test_unattested_checkpoint_rejects_invalid_utf8(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            broken = Path(tmp) / "not-utf8.json"
            broken.write_bytes(b"\xff\xfe not utf-8")
            with self.assertRaises(BankParseError) as caught:
                load_unattested_checkpoint(broken)
            self.assertIn("is not UTF-8", str(caught.exception))

    def test_path_independent_copy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "moved"
            shutil.copytree(VALID.parent, dest)
            moved = load_model_bank(dest / MANIFEST_FILENAME).select("fixture-ok")
            original = load_model_bank(VALID).select("fixture-ok")
            print(f"original digest: {original.checkpoint_digest}")
            print(f"moved digest:    {moved.checkpoint_digest}")
            self.assertEqual(moved.checkpoint_digest, original.checkpoint_digest)
            self.assertNotEqual(moved.checkpoint, original.checkpoint)

    def test_tampered_checkpoint_names_entry_and_field(self) -> None:
        with self.assertRaises(BankAttestationError) as caught:
            load_model_bank(TAMPERED)
        err = caught.exception
        print(f"tampered error: {err}")
        self.assertEqual(err.entry, "fixture-ok")
        self.assertEqual(err.field, "checkpoint_digest")
        self.assertIn("fixture-ok", str(err))
        self.assertIn("checkpoint_digest", str(err))
        self.assertIn("mismatch", str(err))
        self.assertIn("computed sha256:", str(err))

    def test_unsupported_schema_version(self) -> None:
        with self.assertRaises(BankAttestationError) as caught:
            load_model_bank(UNSUPPORTED)
        err = caught.exception
        self.assertEqual(err.field, "schema_version")
        self.assertIn("unsupported version 2", str(err))

    def test_empty_models_array_is_attestation_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "empty"
            shutil.copytree(VALID.parent, dest)
            document = json.loads((dest / MANIFEST_FILENAME).read_text(encoding="utf-8"))
            document["models"] = []
            (dest / MANIFEST_FILENAME).write_text(
                dumps_manifest(document), encoding="utf-8"
            )
            with self.assertRaises(BankAttestationError) as caught:
                load_model_bank(dest / MANIFEST_FILENAME)
            self.assertIsNone(caught.exception.entry)
            self.assertEqual(caught.exception.field, "models")
            self.assertIn("at least one entry", str(caught.exception))

    def test_missing_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "missing"
            shutil.copytree(VALID.parent, dest)
            (dest / "checkpoints" / "ok.bin").unlink()
            with self.assertRaises(BankAttestationError) as caught:
                load_model_bank(dest / MANIFEST_FILENAME)
            self.assertEqual(caught.exception.entry, "fixture-ok")
            self.assertEqual(caught.exception.field, "checkpoint")
            self.assertIn("missing file", str(caught.exception))

    def test_duplicate_model_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "dup"
            shutil.copytree(VALID.parent, dest)
            document = json.loads((dest / MANIFEST_FILENAME).read_text(encoding="utf-8"))
            document["models"].append(dict(document["models"][0]))
            (dest / MANIFEST_FILENAME).write_text(
                dumps_manifest(document), encoding="utf-8"
            )
            with self.assertRaises(BankAttestationError) as caught:
                load_model_bank(dest / MANIFEST_FILENAME)
            self.assertEqual(caught.exception.entry, "fixture-ok")
            self.assertEqual(caught.exception.field, "id")
            self.assertIn("duplicate model ID", str(caught.exception))

    def test_missing_required_contract_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "nocontract"
            shutil.copytree(VALID.parent, dest)
            document = json.loads((dest / MANIFEST_FILENAME).read_text(encoding="utf-8"))
            del document["models"][0]["output_contract_id"]
            (dest / MANIFEST_FILENAME).write_text(
                dumps_manifest(document), encoding="utf-8"
            )
            with self.assertRaises(BankAttestationError) as caught:
                load_model_bank(dest / MANIFEST_FILENAME)
            self.assertEqual(caught.exception.entry, "fixture-ok")
            self.assertEqual(caught.exception.field, "output_contract_id")
            self.assertIn("missing required contract ID", str(caught.exception))

    def test_shipped_merged_v2_bank_and_legacy_wrapper(self) -> None:
        bank = load_shipped_merged_v2_bank()
        entry = bank.select("merged_v2")
        print(f"shipped computed digest: {entry.checkpoint_digest}")
        self.assertEqual(entry.checkpoint_digest, SHIPPED_DIGEST)
        wrapped = wrap_legacy_checkpoint(
            SHIPPED_CHECKPOINT,
            model_id="merged_v2",
            feature_map_id=FEATURE_MAP_LIVE_EXP_025,
            output_contract_id=OUTPUT_CONTRACT_SUPERVISOR_V3_RM1150,
            numeric_format=NUMERIC_FORMAT_Q88,
            training_dataset_digest=entry.training_dataset_digest,
        )
        wrapped_entry = wrapped.select("merged_v2")
        print(f"wrapped computed digest: {wrapped_entry.checkpoint_digest}")
        self.assertEqual(wrapped_entry.checkpoint_digest, entry.checkpoint_digest)
        self.assertEqual(entry.feature_map_id, FEATURE_MAP_LIVE_EXP_025)
        self.assertEqual(
            entry.output_contract_id, OUTPUT_CONTRACT_SUPERVISOR_V3_RM1150
        )
        document = load_unattested_checkpoint(SHIPPED_CHECKPOINT)
        self.assertEqual(
            document["legal_columns"],
            [
                "mem_util_pct",
                "power_w",
                "gpu_temp_c",
                "sm_clock_mhz",
                "mem_clock_mhz",
            ],
        )


if __name__ == "__main__":
    unittest.main()
