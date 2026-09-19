# SPDX-License-Identifier: MIT OR Apache-2.0
"""Model-bank error types and attestation-failure helper.

Kept import-cycle-free so JSON parsing, path I/O, and field checks can
all raise the same contract exceptions.
"""

from __future__ import annotations

from typing import NoReturn


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


class _DuplicateJsonKeyError(ValueError):
    """JSON object repeated a key. CPython ``json.loads`` would last-win."""

    def __init__(self, key: str) -> None:
        super().__init__(f"duplicate object key {key!r}")
        self.key = key


class _NonstandardJsonConstantError(ValueError):
    """CPython ``json.loads`` would accept NaN/Infinity by default."""

    def __init__(self, token: str) -> None:
        super().__init__(f"non-standard JSON constant {token!r}")
        self.token = token


class _Missing:
    """Sentinel for a JSON key that was not present."""


_MISSING = _Missing()


def _escape_error_text(value: str) -> str:
    """ASCII-only form of an untrusted string for error messages."""
    return value.encode("unicode_escape").decode("ascii")


def _fail(*, field: str, message: str, entry: str | None = None) -> NoReturn:
    if entry is None:
        text = f"model-bank field {field!r}: {message}"
    else:
        text = f"model-bank entry {entry!r} field {field!r}: {message}"
    raise BankAttestationError(text, entry=entry, field=field)
