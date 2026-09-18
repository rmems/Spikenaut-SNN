# SPDX-License-Identifier: MIT OR Apache-2.0
"""Model-bank field checks: tokens, digests, schema version, extra keys.

These helpers name the bad entry and field. They do not open checkpoints.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Mapping
from typing import Any

try:  # package import: `python3 -m tools.verify_model_bank`
    from .model_bank_const import (
        DIGEST_RE,
        MSG_MISSING_FIELD,
        SUPPORTED_SCHEMA_VERSIONS,
    )
    from .model_bank_errors import _MISSING, _escape_error_text, _fail
except ImportError:  # direct script: `python3 tools/verify_model_bank.py`
    from model_bank_const import (
        DIGEST_RE,
        MSG_MISSING_FIELD,
        SUPPORTED_SCHEMA_VERSIONS,
    )
    from model_bank_errors import _MISSING, _escape_error_text, _fail


def _require_schema_version(version: Any) -> None:
    if version is _MISSING:
        _fail(field="schema_version", message=MSG_MISSING_FIELD)
    if not _is_int_version(version):
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


def _is_int_version(version: Any) -> bool:
    return isinstance(version, int) and not isinstance(version, bool)


def _reject_unknown_keys(
    mapping: Mapping[str, Any],
    allowed: frozenset[str],
    *,
    entry: str | None = None,
) -> None:
    extra = set(mapping) - allowed
    if extra:
        names = ", ".join(sorted(_escape_error_text(name) for name in extra))
        _fail(
            entry=entry,
            field=min(extra),
            message=f"unsupported extra field(s): {names}",
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
        _fail(entry=entry, field=field, message=MSG_MISSING_FIELD)
    if not isinstance(value, str) or not value:
        _fail(
            entry=entry,
            field=field,
            message=f"must be a non-empty string, got {value!r}",
        )
    if _contains_control(value):
        _fail(
            entry=entry,
            field=field,
            message=f"must not contain a control character, got {value!r}",
        )
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as exc:
        _fail(
            entry=entry,
            field=field,
            message=f"must be well-formed UTF-8, got {value!r}: {exc}",
        )
    return value


def _contains_control(value: str) -> bool:
    """True if any character is Unicode category Cc (C0, DEL, or C1)."""
    return any(unicodedata.category(character) == "Cc" for character in value)


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


def _optional_digest(raw: Mapping[str, Any], model_id: str) -> str | None:
    dataset_raw = raw.get("training_dataset_digest", _MISSING)
    if dataset_raw is _MISSING:
        return None
    return _require_digest(
        dataset_raw, entry=model_id, field="training_dataset_digest"
    )
