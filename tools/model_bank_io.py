# SPDX-License-Identifier: MIT OR Apache-2.0
"""Model-bank checkpoint I/O: digest, resolve, and consume file bytes.

Paths are relative to the resolved manifest. The digest covers file
contents only, so moving a valid bundle without changing bytes still
passes.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

try:  # package import: `python3 -m tools.verify_model_bank`
    from .model_bank_errors import BankAttestationError, _fail
except ImportError:  # direct script: `python3 tools/verify_model_bank.py`
    from model_bank_errors import BankAttestationError, _fail


def digest_file(path: Path) -> str:
    """``sha256:`` digest of the file's bytes.

    The path itself is not hashed, so copying the file to a new directory
    without changing contents yields the same digest. I/O failures become
    :class:`BankAttestationError` so callers never see a raw ``OSError``.
    """
    try:
        digest, _payload = _read_checkpoint_bytes(path)
    except (OSError, ValueError) as exc:
        raise BankAttestationError(
            f"model-bank field 'checkpoint': cannot read file "
            f"{path.as_posix()!r}: {exc}",
            field="checkpoint",
        ) from exc
    return digest


def _hash_bytes(data: bytes) -> str:
    return f"sha256:{hashlib.sha256(data).hexdigest()}"


def _read_checkpoint_bytes(path: Path) -> tuple[str, bytes]:
    with path.open("rb") as handle:
        data = handle.read()
    return _hash_bytes(data), data


def _consume_checkpoint(
    path: Path, *, entry: str, relative: str
) -> tuple[str, bytes]:
    try:
        return _read_checkpoint_bytes(path)
    except (OSError, ValueError) as exc:
        _fail(
            entry=entry,
            field="checkpoint",
            message=f"cannot read file {relative!r}: {exc}",
        )


def _legacy_checkpoint(
    path: Path, *, entry: str
) -> tuple[Path, str, str, bytes]:
    relative = path.name
    _require_relative_checkpoint(relative, entry=entry)
    resolved = _resolve_existing_path(path, entry=entry)
    digest, payload = _consume_checkpoint(
        resolved, entry=entry, relative=relative
    )
    return resolved, relative, digest, payload


def _resolve_existing_path(path: Path, *, entry: str) -> Path:
    try:
        exists = path.is_file()
        resolved = path.resolve() if exists else path
    except (ValueError, RuntimeError, OSError) as exc:
        _fail(
            entry=entry,
            field="checkpoint",
            message=f"cannot resolve path {path.as_posix()!r}: {exc}",
        )
    if not exists:
        _fail(
            entry=entry,
            field="checkpoint",
            message=f"missing file {path.as_posix()!r}",
        )
    return resolved


def _resolved_checkpoint(root: Path, relative: str, *, entry: str) -> Path:
    try:
        checkpoint = (root / relative).resolve()
    except (ValueError, RuntimeError, OSError) as exc:
        _fail(
            entry=entry,
            field="checkpoint",
            message=f"cannot resolve path {relative!r}: {exc}",
        )
    try:
        checkpoint.relative_to(root)
    except ValueError:
        _fail(
            entry=entry,
            field="checkpoint",
            message=f"path {relative!r} escapes the bundle root",
        )
    return checkpoint


def _require_checkpoint_file(
    path: Path, *, entry: str, relative: str
) -> None:
    try:
        present = path.is_file()
    except (OSError, ValueError) as exc:
        _fail(
            entry=entry,
            field="checkpoint",
            message=f"cannot read file {relative!r}: {exc}",
        )
    if not present:
        _fail(
            entry=entry,
            field="checkpoint",
            message=f"missing file {relative!r}",
        )


def _require_relative_checkpoint(relative: str, *, entry: str) -> None:
    reason = _unsafe_relative_reason(relative)
    if reason is not None:
        _fail(entry=entry, field="checkpoint", message=reason)


def _unsafe_relative_reason(relative: str) -> str | None:
    path = Path(relative)
    if path.is_absolute():
        return f"must be a relative path, got {relative!r}"
    if path.anchor:
        return f"must be a relative path, got {relative!r}"
    if ".." in path.parts:
        return f"must not contain '..', got {relative!r}"
    if "\\" in relative:
        return f"must use POSIX separators, got {relative!r}"
    if "\x00" in relative:
        return f"must not contain NUL, got {relative!r}"
    return None
