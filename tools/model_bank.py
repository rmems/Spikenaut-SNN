# SPDX-License-Identifier: MIT OR Apache-2.0
"""Manifest-backed model-bank loader (Linear RM-1327).

A model bank can only support reproducible replay when every selectable
checkpoint is bound to the metadata that explains its input map and output
semantics. This module loads a JSON manifest, attests each entry, and will
hand a caller only entries that passed.

Minimum attestation per entry
-----------------------------
* ``id`` — unique within the bank
* ``checkpoint`` — path relative to the manifest directory
* ``checkpoint_digest`` — ``sha256:`` + 64 lowercase hex of the checkpoint
  **file bytes** (not the path, not a directory walk)
* ``feature_map_id`` — input-map identifier
* ``output_contract_id`` — output-row / action-map identifier; the shipped
  value names the RM-1150 supervisor-v3 decision contract. This loader does
  not implement that decision.
* ``numeric_format`` — e.g. ``q8.8-fixed-point``
* ``training_dataset_digest`` — optional, same ``sha256:`` form

Bank loading rejects a missing checkpoint, a digest mismatch, a duplicate
model ID, an unsupported ``schema_version``, and a missing required contract
ID. Errors name the bad entry and field.

Verification is path-independent: moving a valid bundle without changing
bytes still passes, because paths are relative to the manifest and the
digest covers file contents only.

Stable serialization
--------------------
``dumps_manifest`` is the contract:

* UTF-8, LF newlines, exactly one trailing newline
* ``json.dumps(..., indent=2, sort_keys=True, ensure_ascii=True)``
* object keys sorted recursively; array order is significant

The golden document is ``tools/fixtures/model_bank/valid/model_bank.json``.

Compatibility
-------------
``SnnModel::load_default`` and ``verify_q88`` remain the single-checkpoint
path. ``wrap_legacy_checkpoint`` and ``load_unattested_checkpoint`` wrap that
layout so existing callers do not have to write a manifest before they can
keep loading one file. Prefer ``load_model_bank`` when selecting a model.

Standard library only.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NoReturn

REPO_ROOT = Path(__file__).resolve().parent.parent
SHIPPED_MANIFEST = REPO_ROOT / "dataset" / "merged_v2" / "model_bank.json"
SHIPPED_CHECKPOINT = REPO_ROOT / "dataset" / "merged_v2" / "snn_model.json"

SCHEMA_VERSION = 1
SUPPORTED_SCHEMA_VERSIONS = frozenset({SCHEMA_VERSION})

DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")

# Canonical identifiers this repository ships. Unknown-but-well-formed IDs
# still attest — the bank is not a registry — but missing/empty values do not.
FEATURE_MAP_LIVE_EXP_025 = "spikenaut.feature-map.live-exp-025.v1"
OUTPUT_CONTRACT_SUPERVISOR_V3_RM1150 = (
    "spikenaut.output-contract.supervisor-v3.rm-1150"
)
NUMERIC_FORMAT_Q88 = "q8.8-fixed-point"
HUB_V3_JSONL_DIGEST = (
    "sha256:26d7d7442605a32b11330f53f55e621750f31e775465209991f73f94a6c72c09"
)

MANIFEST_KEYS = frozenset({"schema_version", "models"})
ENTRY_REQUIRED = (
    "id",
    "checkpoint",
    "checkpoint_digest",
    "feature_map_id",
    "output_contract_id",
    "numeric_format",
)
ENTRY_OPTIONAL = ("training_dataset_digest",)
ENTRY_KEYS = frozenset(ENTRY_REQUIRED + ENTRY_OPTIONAL)


class BankError(Exception):
    """A model-bank manifest could not be used."""


class BankParseError(BankError):
    """The manifest could not be read as the schema this loader expects.

    Mapped to process exit code 2: nothing was attested.
    """


class BankAttestationError(BankError):
    """An entry or field failed attestation.

    Mapped to process exit code 1: the bank was read and rejected.
    """

    def __init__(
        self,
        message: str,
        *,
        entry: str | None = None,
        field: str | None = None,
    ) -> None:
        super().__init__(message)
        self.entry = entry
        self.field = field


def dumps_manifest(document: Mapping[str, Any]) -> str:
    """Stable serialization of a model-bank document.

    See the module docstring. The golden fixture must round-trip through this
    function unchanged.
    """
    return json.dumps(
        document,
        indent=2,
        sort_keys=True,
        ensure_ascii=True,
    ) + "\n"


def digest_file(path: Path) -> str:
    """``sha256:`` digest of the file's bytes.

    The path itself is not hashed, so copying the file to a new directory
    without changing contents yields the same digest.
    """
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            hasher.update(chunk)
    return f"sha256:{hasher.hexdigest()}"


@dataclass(frozen=True)
class AttestedEntry:
    """One bank entry that passed attestation."""

    id: str
    checkpoint: Path
    checkpoint_relative: str
    checkpoint_digest: str
    feature_map_id: str
    output_contract_id: str
    numeric_format: str
    training_dataset_digest: str | None


@dataclass(frozen=True)
class ModelBank:
    """A fully attested bank. Every entry in ``entries`` passed."""

    manifest_path: Path | None
    root: Path
    entries: tuple[AttestedEntry, ...]

    def ids(self) -> tuple[str, ...]:
        """Model IDs in manifest order."""
        return tuple(entry.id for entry in self.entries)

    def select(self, model_id: str) -> AttestedEntry:
        """Return one entry, re-affirming its digest at selection time.

        Only entries that passed load-time attestation are visible. An unknown
        ID is not a silent miss: it names the field.
        """
        for entry in self.entries:
            if entry.id == model_id:
                _reaffirm(entry)
                return entry
        raise BankAttestationError(
            f"model-bank field 'id': not an attested entry: {model_id!r}",
            field="id",
        )

    def __iter__(self) -> Iterator[AttestedEntry]:
        return iter(self.entries)

    def __len__(self) -> int:
        return len(self.entries)


def load_model_bank(manifest_path: Path | str) -> ModelBank:
    """Load and attest every entry in ``manifest_path``.

    # Errors

    * :class:`BankParseError` if the file cannot be read or is not JSON
    * :class:`BankAttestationError` if any entry fails attestation
    """
    path = Path(manifest_path)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise BankParseError(
            f"model-bank: cannot read manifest {path}: {exc}"
        ) from exc
    except UnicodeDecodeError as exc:
        raise BankParseError(
            f"model-bank: manifest {path} is not UTF-8: {exc}"
        ) from exc
    try:
        document = json.loads(text)
    except json.JSONDecodeError as exc:
        raise BankParseError(
            f"model-bank: invalid JSON in {path}: {exc}"
        ) from exc
    if not isinstance(document, dict):
        raise BankParseError(
            f"model-bank: top level must be an object, got {type(document).__name__}"
        )
    root = path.parent.resolve()
    entries = _attest_document(document, root)
    return ModelBank(manifest_path=path.resolve(), root=root, entries=entries)


def load_shipped_merged_v2_bank() -> ModelBank:
    """Attest the historical single ``merged_v2`` checkpoint as a one-entry bank."""
    return load_model_bank(SHIPPED_MANIFEST)


def wrap_legacy_checkpoint(
    checkpoint_path: Path | str,
    *,
    model_id: str,
    feature_map_id: str,
    output_contract_id: str,
    numeric_format: str,
    training_dataset_digest: str | None = None,
) -> ModelBank:
    """Compatibility wrapper: one existing checkpoint file as a one-entry bank.

    Computes the digest from the file bytes. Callers that already have a lone
    ``snn_model.json`` can migrate without writing a manifest first. The
    required contract identifiers still have to be supplied — wrapping does
    not invent an output map.
    """
    path = Path(checkpoint_path)
    entry_id = _require_token(model_id, entry=model_id, field="id")
    relative = path.name
    _require_relative_checkpoint(relative, entry=entry_id)
    if not path.is_file():
        _fail(
            entry=entry_id,
            field="checkpoint",
            message=f"missing file {path.as_posix()!r}",
        )
    resolved = path.resolve()
    digest = digest_file(resolved)
    feature = _require_token(
        feature_map_id, entry=entry_id, field="feature_map_id"
    )
    contract = _require_contract_id(output_contract_id, entry=entry_id)
    fmt = _require_token(
        numeric_format, entry=entry_id, field="numeric_format"
    )
    dataset = None
    if training_dataset_digest is not None:
        dataset = _require_digest(
            training_dataset_digest,
            entry=entry_id,
            field="training_dataset_digest",
        )
    attested = AttestedEntry(
        id=entry_id,
        checkpoint=resolved,
        checkpoint_relative=relative,
        checkpoint_digest=digest,
        feature_map_id=feature,
        output_contract_id=contract,
        numeric_format=fmt,
        training_dataset_digest=dataset,
    )
    return ModelBank(
        manifest_path=None,
        root=resolved.parent,
        entries=(attested,),
    )


def load_unattested_checkpoint(path: Path | str) -> Any:
    """Read a standalone ``snn_model.json`` without bank attestation.

    This is the pre-RM-1327 single-checkpoint path. It does not verify
    digest, feature-map, or output-contract metadata. Prefer
    :func:`load_model_bank` when selecting among models.
    """
    checkpoint = Path(path)
    try:
        text = checkpoint.read_text(encoding="utf-8")
    except OSError as exc:
        raise BankParseError(
            f"model-bank: cannot read checkpoint {checkpoint}: {exc}"
        ) from exc
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise BankParseError(
            f"model-bank: invalid JSON in {checkpoint}: {exc}"
        ) from exc


def _attest_document(
    document: dict[str, Any], root: Path
) -> tuple[AttestedEntry, ...]:
    extra = set(document) - MANIFEST_KEYS
    if extra:
        names = ", ".join(sorted(extra))
        _fail(
            field=sorted(extra)[0],
            message=f"unsupported extra field(s): {names}",
        )
    version = document.get("schema_version", _MISSING)
    if version is _MISSING:
        _fail(field="schema_version", message="missing required field")
    if type(version) is not int:
        _fail(
            field="schema_version",
            message=f"must be an integer, got {type(version).__name__}",
        )
    if version not in SUPPORTED_SCHEMA_VERSIONS:
        supported = ", ".join(str(v) for v in sorted(SUPPORTED_SCHEMA_VERSIONS))
        _fail(
            field="schema_version",
            message=f"unsupported version {version} (supported: {supported})",
        )
    models = document.get("models", _MISSING)
    if models is _MISSING:
        _fail(field="models", message="missing required field")
    if not isinstance(models, list):
        _fail(
            field="models",
            message=f"must be an array, got {type(models).__name__}",
        )
    attested: list[AttestedEntry] = []
    seen: dict[str, int] = {}
    for index, raw in enumerate(models):
        entry = _attest_entry(raw, index, root, seen)
        seen[entry.id] = index
        attested.append(entry)
    return tuple(attested)


def _attest_entry(
    raw: Any,
    index: int,
    root: Path,
    seen: Mapping[str, int],
) -> AttestedEntry:
    label = f"[{index}]"
    if not isinstance(raw, dict):
        _fail(
            entry=label,
            field="id",
            message=f"entry must be an object, got {type(raw).__name__}",
        )
    extra = set(raw) - ENTRY_KEYS
    if extra:
        names = ", ".join(sorted(extra))
        _fail(
            entry=label,
            field="id",
            message=f"unsupported extra field(s): {names}",
        )
    model_id = _require_token(raw.get("id", _MISSING), entry=label, field="id")
    if model_id in seen:
        _fail(
            entry=model_id,
            field="id",
            message="duplicate model ID",
        )
    relative = _require_token(
        raw.get("checkpoint", _MISSING), entry=model_id, field="checkpoint"
    )
    _require_relative_checkpoint(relative, entry=model_id)
    declared = _require_digest(
        raw.get("checkpoint_digest", _MISSING),
        entry=model_id,
        field="checkpoint_digest",
    )
    checkpoint = (root / relative).resolve()
    try:
        checkpoint.relative_to(root)
    except ValueError:
        _fail(
            entry=model_id,
            field="checkpoint",
            message=f"path {relative!r} escapes the bundle root",
        )
    if not checkpoint.is_file():
        _fail(
            entry=model_id,
            field="checkpoint",
            message=f"missing file {relative!r}",
        )
    computed = digest_file(checkpoint)
    if computed != declared:
        _fail(
            entry=model_id,
            field="checkpoint_digest",
            message=(
                f"mismatch (declared {declared}, computed {computed})"
            ),
        )
    feature = _require_token(
        raw.get("feature_map_id", _MISSING),
        entry=model_id,
        field="feature_map_id",
    )
    contract = _require_contract_id(
        raw.get("output_contract_id", _MISSING),
        entry=model_id,
    )
    fmt = _require_token(
        raw.get("numeric_format", _MISSING),
        entry=model_id,
        field="numeric_format",
    )
    dataset_raw = raw.get("training_dataset_digest", _MISSING)
    dataset = None
    if dataset_raw is not _MISSING:
        dataset = _require_digest(
            dataset_raw, entry=model_id, field="training_dataset_digest"
        )
    return AttestedEntry(
        id=model_id,
        checkpoint=checkpoint,
        checkpoint_relative=relative,
        checkpoint_digest=computed,
        feature_map_id=feature,
        output_contract_id=contract,
        numeric_format=fmt,
        training_dataset_digest=dataset,
    )


def _reaffirm(entry: AttestedEntry) -> None:
    """Selection-time check: the attested file is still the attested bytes."""
    if not entry.checkpoint.is_file():
        _fail(
            entry=entry.id,
            field="checkpoint",
            message=f"missing file {entry.checkpoint_relative!r}",
        )
    computed = digest_file(entry.checkpoint)
    if computed != entry.checkpoint_digest:
        _fail(
            entry=entry.id,
            field="checkpoint_digest",
            message=(
                f"mismatch (declared {entry.checkpoint_digest}, "
                f"computed {computed})"
            ),
        )


def _require_relative_checkpoint(relative: str, *, entry: str) -> None:
    path = Path(relative)
    if path.is_absolute() or path.anchor:
        _fail(
            entry=entry,
            field="checkpoint",
            message=f"must be a relative path, got {relative!r}",
        )
    if any(part == ".." for part in path.parts):
        _fail(
            entry=entry,
            field="checkpoint",
            message=f"must not contain '..', got {relative!r}",
        )
    if path.as_posix() != relative.replace("\\", "/") or "\\" in relative:
        # Windows separators would make the same bundle path-dependent.
        if "\\" in relative:
            _fail(
                entry=entry,
                field="checkpoint",
                message=f"must use POSIX separators, got {relative!r}",
            )


def _require_contract_id(value: Any, *, entry: str) -> str:
    if value is _MISSING or value is None or value == "":
        _fail(
            entry=entry,
            field="output_contract_id",
            message="missing required contract ID",
        )
    return _require_token(value, entry=entry, field="output_contract_id")


def _require_token(value: Any, *, entry: str, field: str) -> str:
    if value is _MISSING:
        _fail(entry=entry, field=field, message="missing required field")
    if not isinstance(value, str) or not value:
        _fail(
            entry=entry,
            field=field,
            message=f"must be a non-empty string, got {value!r}",
        )
    return value


def _require_digest(value: Any, *, entry: str, field: str) -> str:
    token = _require_token(value, entry=entry, field=field)
    if DIGEST_RE.fullmatch(token) is None:
        _fail(
            entry=entry,
            field=field,
            message=(
                "must be 'sha256:' plus 64 lowercase hex characters, "
                f"got {token!r}"
            ),
        )
    return token


def _fail(*, field: str, message: str, entry: str | None = None) -> NoReturn:
    if entry is None:
        text = f"model-bank field {field!r}: {message}"
    else:
        text = f"model-bank entry {entry!r} field {field!r}: {message}"
    raise BankAttestationError(text, entry=entry, field=field)


class _Missing:
    """Sentinel for a JSON key that was not present."""


_MISSING = _Missing()
