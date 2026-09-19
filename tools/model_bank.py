# SPDX-License-Identifier: MIT OR Apache-2.0
"""Manifest-backed model-bank loader (Linear RM-1327).

A model bank can only support reproducible replay when every selectable
checkpoint is bound to the metadata that explains its input map and output
semantics. This module loads a JSON manifest, attests each entry, and will
hand a caller only entries that passed.

Minimum attestation per entry
-----------------------------
* ``id`` -- unique within the bank
* ``checkpoint`` -- path relative to the manifest directory
* ``checkpoint_digest`` -- ``sha256:`` + 64 lowercase hex of the checkpoint
  **file bytes** (not the path, not a directory walk)
* ``feature_map_id`` -- input-map identifier
* ``output_contract_id`` -- output-row / action-map identifier; the shipped
  value names the RM-1150 supervisor-v3 decision contract. This loader does
  not implement that decision.
* ``numeric_format`` -- consumption-grid identifier (the shipped bank uses
  ``q8.8-fixed-point``, matching ``config.json`` ``weight_format``). The
  checkpoint file may still be float JSON; the digest covers those bytes,
  and Q8.8 encoding stays on the ``verify_q88`` path.
* ``training_dataset_digest`` -- optional, same ``sha256:`` form

Bank loading rejects a missing checkpoint, a digest mismatch, a duplicate
model ID, an empty ``models`` array, an unsupported ``schema_version``, a
missing required contract ID, a JSON object with a duplicate key, and the
non-standard constants ``NaN`` / ``Infinity`` / ``-Infinity``. Duplicate
keys would otherwise last-win under ``json.loads`` and attest a different
value than another parser. Errors name the bad entry and field.

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

Layout
------
* ``model_bank_const`` -- shipped identifiers, schema keys, bundle paths
* ``model_bank_errors`` -- parse/attestation exceptions
* ``model_bank_json`` -- stable dump and fail-closed parse
* ``model_bank_io`` -- checkpoint digest, resolve, consume
* ``model_bank_fields`` -- token, digest, schema, extra-key checks
* this module -- attested types, load, select, wrap

Standard library only.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:  # package import: `python3 -m tools.verify_model_bank`
    from .model_bank_const import (
        DIGEST_RE,
        ENTRY_KEYS,
        ENTRY_OPTIONAL,
        ENTRY_REQUIRED,
        FEATURE_MAP_LIVE_EXP_025,
        HUB_V3_JSONL_DIGEST,
        MANIFEST_FILENAME,
        MANIFEST_KEYS,
        MSG_MISSING_FIELD,
        NUMERIC_FORMAT_Q88,
        OUTPUT_CONTRACT_SUPERVISOR_V3_RM1150,
        REPO_ROOT,
        SCHEMA_VERSION,
        SHIPPED_CHECKPOINT,
        SHIPPED_MANIFEST,
        SUPPORTED_SCHEMA_VERSIONS,
    )
    from .model_bank_errors import (
        BankAttestationError,
        BankError,
        BankParseError,
        _MISSING,
        _fail,
    )
    from .model_bank_fields import (
        _optional_digest,
        _reject_unknown_keys,
        _require_contract_id,
        _require_digest,
        _require_schema_version,
        _require_token,
    )
    from .model_bank_io import (
        _consume_checkpoint,
        _legacy_checkpoint,
        _require_checkpoint_file,
        _require_relative_checkpoint,
        _resolved_checkpoint,
        digest_file,
    )
    from .model_bank_json import _parse_json, _read_utf8, dumps_manifest
except ImportError:  # direct script: `python3 tools/verify_model_bank.py`
    from model_bank_const import (
        DIGEST_RE,
        ENTRY_KEYS,
        ENTRY_OPTIONAL,
        ENTRY_REQUIRED,
        FEATURE_MAP_LIVE_EXP_025,
        HUB_V3_JSONL_DIGEST,
        MANIFEST_FILENAME,
        MANIFEST_KEYS,
        MSG_MISSING_FIELD,
        NUMERIC_FORMAT_Q88,
        OUTPUT_CONTRACT_SUPERVISOR_V3_RM1150,
        REPO_ROOT,
        SCHEMA_VERSION,
        SHIPPED_CHECKPOINT,
        SHIPPED_MANIFEST,
        SUPPORTED_SCHEMA_VERSIONS,
    )
    from model_bank_errors import (
        BankAttestationError,
        BankError,
        BankParseError,
        _MISSING,
        _fail,
    )
    from model_bank_fields import (
        _optional_digest,
        _reject_unknown_keys,
        _require_contract_id,
        _require_digest,
        _require_schema_version,
        _require_token,
    )
    from model_bank_io import (
        _consume_checkpoint,
        _legacy_checkpoint,
        _require_checkpoint_file,
        _require_relative_checkpoint,
        _resolved_checkpoint,
        digest_file,
    )
    from model_bank_json import _parse_json, _read_utf8, dumps_manifest

__all__ = (
    "DIGEST_RE",
    "ENTRY_KEYS",
    "ENTRY_OPTIONAL",
    "ENTRY_REQUIRED",
    "FEATURE_MAP_LIVE_EXP_025",
    "HUB_V3_JSONL_DIGEST",
    "MANIFEST_FILENAME",
    "MANIFEST_KEYS",
    "MSG_MISSING_FIELD",
    "NUMERIC_FORMAT_Q88",
    "OUTPUT_CONTRACT_SUPERVISOR_V3_RM1150",
    "REPO_ROOT",
    "SCHEMA_VERSION",
    "SHIPPED_CHECKPOINT",
    "SHIPPED_MANIFEST",
    "SUPPORTED_SCHEMA_VERSIONS",
    "AttestedEntry",
    "BankAttestationError",
    "BankError",
    "BankParseError",
    "ModelBank",
    "digest_file",
    "dumps_manifest",
    "load_model_bank",
    "load_shipped_merged_v2_bank",
    "load_unattested_checkpoint",
    "wrap_legacy_checkpoint",
)


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
    checkpoint_bytes: bytes


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
        """Return one load-attested entry, consuming those checkpoint bytes.

        Only entries that passed load-time attestation are visible. An unknown
        ID is not a silent miss: it names the field. The returned
        ``checkpoint_bytes`` are the bytes hashed at load time, so a later
        replacement of the file cannot change what the caller consumes.
        """
        for entry in self.entries:
            if entry.id == model_id:
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

    Checkpoint paths are relative to the resolved manifest file. A symlink
    to the JSON still loads the bundle next to the opened file rather than
    next to the link.

    # Errors

    * :class:`BankParseError` if the file cannot be read or is not JSON
    * :class:`BankAttestationError` if any entry fails attestation
    """
    path = Path(manifest_path)
    try:
        # Resolve the file first so the bytes we parse and the bundle root
        # that owns checkpoint paths come from the same opened target.
        resolved_manifest = path.resolve()
    except (ValueError, RuntimeError, OSError) as exc:
        raise BankParseError(
            f"model-bank: cannot resolve manifest {path}: {exc}"
        ) from exc
    root = resolved_manifest.parent
    document = _parse_json(
        _read_utf8(resolved_manifest, kind="manifest"), source=resolved_manifest
    )
    if not isinstance(document, dict):
        raise BankParseError(
            f"model-bank: top level must be an object, got {type(document).__name__}"
        )
    entries = _attest_document(document, root)
    return ModelBank(
        manifest_path=resolved_manifest, root=root, entries=entries
    )


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
    required contract identifiers still have to be supplied -- wrapping does
    not invent an output map.
    """
    path = Path(checkpoint_path)
    entry_id = _require_token(model_id, entry=model_id, field="id")
    resolved, relative, digest, payload = _legacy_checkpoint(path, entry=entry_id)
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
        checkpoint_bytes=payload,
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
    return _parse_json(
        _read_utf8(checkpoint, kind="checkpoint"), source=checkpoint
    )


def _attest_document(
    document: dict[str, Any], root: Path
) -> tuple[AttestedEntry, ...]:
    _reject_unknown_keys(document, MANIFEST_KEYS)
    _require_schema_version(document.get("schema_version", _MISSING))
    models = document.get("models", _MISSING)
    if models is _MISSING:
        _fail(field="models", message=MSG_MISSING_FIELD)
    if not isinstance(models, list):
        _fail(
            field="models",
            message=f"must be an array, got {type(models).__name__}",
        )
    if not models:
        _fail(field="models", message="must contain at least one entry")
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
    _reject_unknown_keys(raw, ENTRY_KEYS, entry=label)
    model_id = _require_token(raw.get("id", _MISSING), entry=label, field="id")
    if model_id in seen:
        _fail(entry=model_id, field="id", message="duplicate model ID")
    relative, checkpoint, computed, payload = _attest_checkpoint(
        raw, model_id, root
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
    dataset = _optional_digest(raw, model_id)
    return AttestedEntry(
        id=model_id,
        checkpoint=checkpoint,
        checkpoint_relative=relative,
        checkpoint_digest=computed,
        feature_map_id=feature,
        output_contract_id=contract,
        numeric_format=fmt,
        training_dataset_digest=dataset,
        checkpoint_bytes=payload,
    )


def _attest_checkpoint(
    raw: Mapping[str, Any], model_id: str, root: Path
) -> tuple[str, Path, str, bytes]:
    relative = _require_token(
        raw.get("checkpoint", _MISSING), entry=model_id, field="checkpoint"
    )
    _require_relative_checkpoint(relative, entry=model_id)
    declared = _require_digest(
        raw.get("checkpoint_digest", _MISSING),
        entry=model_id,
        field="checkpoint_digest",
    )
    checkpoint = _resolved_checkpoint(root, relative, entry=model_id)
    _require_checkpoint_file(checkpoint, entry=model_id, relative=relative)
    computed, payload = _consume_checkpoint(
        checkpoint, entry=model_id, relative=relative
    )
    if computed != declared:
        _fail(
            entry=model_id,
            field="checkpoint_digest",
            message=f"mismatch (declared {declared}, computed {computed})",
        )
    return relative, checkpoint, computed, payload
