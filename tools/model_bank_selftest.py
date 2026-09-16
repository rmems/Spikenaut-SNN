"""Self-test for the model-bank attester: prove the checker can actually fail.

A loader that cannot fail is worthless. Each numbered scenario asserts
``model_bank`` rejects a specific regression named by Linear RM-1327, and that
a valid bundle still attests after it is copied to a new path.

Scratch files are written to a temp directory; nothing here ever writes into
``dataset/`` or the checked-in fixtures.

Run via the CLI::

    python3 tools/verify_model_bank.py --self-test
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path

try:  # package import: `python3 -m tools.verify_model_bank`
    from .model_bank import (
        FEATURE_MAP_LIVE_EXP_025,
        MANIFEST_FILENAME,
        NUMERIC_FORMAT_Q88,
        OUTPUT_CONTRACT_SUPERVISOR_V3_RM1150,
        REPO_ROOT,
        BankAttestationError,
        dumps_manifest,
        load_model_bank,
        load_shipped_merged_v2_bank,
        load_unattested_checkpoint,
        wrap_legacy_checkpoint,
    )
    from .q88_core import SelfTestFailure
except ImportError:  # direct script: `python3 tools/verify_model_bank.py`
    from model_bank import (
        FEATURE_MAP_LIVE_EXP_025,
        MANIFEST_FILENAME,
        NUMERIC_FORMAT_Q88,
        OUTPUT_CONTRACT_SUPERVISOR_V3_RM1150,
        REPO_ROOT,
        BankAttestationError,
        dumps_manifest,
        load_model_bank,
        load_shipped_merged_v2_bank,
        load_unattested_checkpoint,
        wrap_legacy_checkpoint,
    )
    from q88_core import SelfTestFailure

FIXTURE_ROOT = REPO_ROOT / "tools" / "fixtures" / "model_bank"
VALID_MANIFEST = FIXTURE_ROOT / "valid" / MANIFEST_FILENAME
TAMPERED_MANIFEST = FIXTURE_ROOT / "tampered-checkpoint" / MANIFEST_FILENAME
UNSUPPORTED_MANIFEST = FIXTURE_ROOT / "unsupported-version" / MANIFEST_FILENAME

SELF_TEST_SECTIONS = 10
EXPECTED_REQUIRE_CALLS = 16
EXPECTED_GUARD_CHECKS = 6


def _require(condition: bool, message: str) -> None:
    """Assert a self-test invariant. Survives ``python -O``."""
    _require.calls += 1
    if not condition:
        raise SelfTestFailure(message)


_require.calls = 0


def _bundle_manifest(bundle: Path) -> Path:
    return bundle / MANIFEST_FILENAME


def _raises_attestation(
    fn: Callable[[], object],
    *,
    field: str,
    entry: str | None = None,
) -> int:
    """``fn`` must fail naming ``field`` (and ``entry`` when given). One guard."""
    try:
        fn()
    except BankAttestationError as exc:
        _assert_error_names(exc, field=field, entry=entry)
        return 1
    raise SelfTestFailure(
        f"expected BankAttestationError naming field {field!r}"
    )


def _assert_error_names(
    exc: BankAttestationError, *, field: str, entry: str | None
) -> None:
    text = str(exc)
    if exc.field != field:
        raise SelfTestFailure(
            f"error must name field {field!r}, got {text!r}"
        )
    if field not in text:
        raise SelfTestFailure(
            f"error must name field {field!r}, got {text!r}"
        )
    if entry is None:
        return
    if entry not in text:
        raise SelfTestFailure(
            f"error must name entry {entry!r}, got {text!r}"
        )


def valid_fixture_attests(stream) -> int:
    print("1. valid fixture attests and select returns it", file=stream)
    bank = load_model_bank(VALID_MANIFEST)
    _require(len(bank) == 1, "valid fixture is one entry")
    entry = bank.select("fixture-ok")
    print(f"   computed digest: {entry.checkpoint_digest}", file=stream)
    _require(
        entry.checkpoint_digest.startswith("sha256:"),
        "digest uses the sha256: prefix",
    )
    _require(
        entry.feature_map_id == FEATURE_MAP_LIVE_EXP_025,
        "feature map is the live exp-025 identifier",
    )
    _require(
        entry.output_contract_id == OUTPUT_CONTRACT_SUPERVISOR_V3_RM1150,
        "output contract names RM-1150, and is not reimplemented here",
    )
    _require(
        entry.numeric_format == NUMERIC_FORMAT_Q88,
        "numeric format is q8.8-fixed-point",
    )
    _require(
        bank.ids() == ("fixture-ok",),
        "selection surface is only attested ids",
    )
    _require(
        f"sha256:{hashlib.sha256(entry.checkpoint_bytes).hexdigest()}"
        == entry.checkpoint_digest,
        "select returns the attested checkpoint bytes",
    )
    return 7


def serialization_is_stable(stream) -> int:
    print("2. golden manifest matches dumps_manifest", file=stream)
    raw_bytes = VALID_MANIFEST.read_bytes()
    raw = raw_bytes.decode("utf-8")
    document = json.loads(raw)
    dumped = dumps_manifest(document)
    _require(
        raw_bytes == dumped.encode("utf-8"),
        "golden valid manifest is the stable serialization",
    )
    _require(dumped.endswith("\n"), "stable dump has a trailing newline")
    _require(b"\r" not in raw_bytes, "golden fixture is LF-only on disk")
    return 3


def copied_bundle_still_attests(stream, tmp: Path) -> int:
    print("3. moving a valid bundle without changing bytes still attests", file=stream)
    dest = tmp / "moved-valid"
    shutil.copytree(VALID_MANIFEST.parent, dest)
    bank = load_model_bank(_bundle_manifest(dest))
    entry = bank.select("fixture-ok")
    original = load_model_bank(VALID_MANIFEST).select("fixture-ok")
    print(f"   original digest: {original.checkpoint_digest}", file=stream)
    print(f"   moved digest:    {entry.checkpoint_digest}", file=stream)
    _require(
        entry.checkpoint_digest == original.checkpoint_digest,
        "digest is path-independent",
    )
    _require(
        entry.checkpoint != original.checkpoint,
        "the moved checkpoint is a different path",
    )
    return 2


def tampered_checkpoint_rejected(stream) -> int:
    print("4. tampered checkpoint is a digest mismatch", file=stream)
    return _raises_attestation(
        lambda: load_model_bank(TAMPERED_MANIFEST),
        entry="fixture-ok",
        field="checkpoint_digest",
    )


def unsupported_version_rejected(stream) -> int:
    print("5. unsupported schema version is rejected", file=stream)
    return _raises_attestation(
        lambda: load_model_bank(UNSUPPORTED_MANIFEST),
        field="schema_version",
    )


def missing_file_rejected(stream, tmp: Path) -> int:
    print("6. missing checkpoint file is rejected", file=stream)
    dest = tmp / "missing-file"
    shutil.copytree(VALID_MANIFEST.parent, dest)
    (dest / "checkpoints" / "ok.bin").unlink()
    return _raises_attestation(
        lambda: load_model_bank(_bundle_manifest(dest)),
        entry="fixture-ok",
        field="checkpoint",
    )


def duplicate_id_rejected(stream, tmp: Path) -> int:
    print("7. duplicate model ID is rejected", file=stream)
    dest = tmp / "duplicate-id"
    shutil.copytree(VALID_MANIFEST.parent, dest)
    document = json.loads(_bundle_manifest(dest).read_text(encoding="utf-8"))
    document["models"].append(dict(document["models"][0]))
    _bundle_manifest(dest).write_text(
        dumps_manifest(document), encoding="utf-8"
    )
    return _raises_attestation(
        lambda: load_model_bank(_bundle_manifest(dest)),
        entry="fixture-ok",
        field="id",
    )


def missing_contract_id_rejected(stream, tmp: Path) -> int:
    print("8. missing required contract ID is rejected", file=stream)
    dest = tmp / "missing-contract"
    shutil.copytree(VALID_MANIFEST.parent, dest)
    document = json.loads(_bundle_manifest(dest).read_text(encoding="utf-8"))
    del document["models"][0]["output_contract_id"]
    _bundle_manifest(dest).write_text(
        dumps_manifest(document), encoding="utf-8"
    )
    return _raises_attestation(
        lambda: load_model_bank(_bundle_manifest(dest)),
        entry="fixture-ok",
        field="output_contract_id",
    )


def shipped_and_legacy_paths(stream) -> int:
    print("9. shipped merged_v2 bank and legacy wrappers still work", file=stream)
    bank = load_shipped_merged_v2_bank()
    entry = bank.select("merged_v2")
    print(f"   shipped digest: {entry.checkpoint_digest}", file=stream)
    _require(len(bank) == 1, "shipped bank is one entry")
    _require(
        entry.output_contract_id == OUTPUT_CONTRACT_SUPERVISOR_V3_RM1150,
        "shipped entry names the RM-1150 contract",
    )
    wrapped = wrap_legacy_checkpoint(
        REPO_ROOT / "dataset" / "merged_v2" / "snn_model.json",
        model_id="merged_v2",
        feature_map_id=FEATURE_MAP_LIVE_EXP_025,
        output_contract_id=OUTPUT_CONTRACT_SUPERVISOR_V3_RM1150,
        numeric_format=NUMERIC_FORMAT_Q88,
        training_dataset_digest=entry.training_dataset_digest,
    )
    wrapped_entry = wrapped.select("merged_v2")
    print(f"   wrapped digest: {wrapped_entry.checkpoint_digest}", file=stream)
    _require(
        wrapped_entry.checkpoint_digest == entry.checkpoint_digest,
        "legacy wrapper computes the same digest as the shipped manifest",
    )
    document = load_unattested_checkpoint(
        REPO_ROOT / "dataset" / "merged_v2" / "snn_model.json"
    )
    _require(
        isinstance(document, dict) and "neurons" in document,
        "unattested checkpoint still loads as JSON",
    )
    return 4


def unknown_id_not_selectable(stream) -> int:
    print("10. selection cannot return an unattested id", file=stream)
    bank = load_model_bank(VALID_MANIFEST)
    return _raises_attestation(
        lambda: bank.select("no-such-model"),
        entry="no-such-model",
        field="id",
    )


def _suite_is_fully_accounted_for(sections_run: int, scenario_checks: int) -> int:
    scenario_requires = _require.calls
    _require(
        sections_run == SELF_TEST_SECTIONS,
        f"self-test ran {sections_run} numbered sections, expected "
        f"{SELF_TEST_SECTIONS} -- a scenario was dropped",
    )
    _require(
        scenario_requires == EXPECTED_REQUIRE_CALLS,
        f"scenarios made {scenario_requires} _require calls, expected "
        f"{EXPECTED_REQUIRE_CALLS}",
    )
    _require(
        scenario_checks - scenario_requires == EXPECTED_GUARD_CHECKS,
        f"scenarios reported {scenario_checks} assertions against "
        f"{scenario_requires} _require calls, leaving "
        f"{scenario_checks - scenario_requires} try/except guards, expected "
        f"{EXPECTED_GUARD_CHECKS}",
    )
    return 3


def self_test(stream=sys.stdout) -> bool:
    """Prove this checker actually rejects real regressions."""
    print("model-bank attester self-test", file=stream)
    print("", file=stream)
    _require.calls = 0
    checks = 0
    sections_run = 0

    with tempfile.TemporaryDirectory(prefix="model-bank-selftest-") as tmpdir:
        tmp = Path(tmpdir)
        ordered = (
            valid_fixture_attests,
            serialization_is_stable,
            lambda stream: copied_bundle_still_attests(stream, tmp),
            tampered_checkpoint_rejected,
            unsupported_version_rejected,
            lambda stream: missing_file_rejected(stream, tmp),
            lambda stream: duplicate_id_rejected(stream, tmp),
            lambda stream: missing_contract_id_rejected(stream, tmp),
            shipped_and_legacy_paths,
            unknown_id_not_selectable,
        )
        for scenario in ordered:
            checks += scenario(stream)
            sections_run += 1

    checks += _suite_is_fully_accounted_for(sections_run, checks)

    print(
        f"SELF-TEST PASSED: {checks} assertions across {sections_run} sections.\n"
        f"The loader attests the golden fixture and the shipped merged_v2 bank;\n"
        f"rejects a missing file, digest mismatch, duplicate model ID,\n"
        f"unsupported schema version, and missing required contract ID; and a\n"
        f"copied valid bundle still attests.",
        file=stream,
    )
    return True
