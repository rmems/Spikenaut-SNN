# SPDX-License-Identifier: MIT OR Apache-2.0
"""Model-bank JSON contract: stable dump and fail-closed parse.

``dumps_manifest`` is the on-disk serialization. ``_parse_json`` maps
CPython decoder edges (duplicate keys, NaN/Infinity, oversized ints,
deep nesting, invalid UTF-8) onto ``BankParseError``.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

try:  # package import: `python3 -m tools.verify_model_bank`
    from .model_bank_errors import (
        BankParseError,
        _DuplicateJsonKeyError,
        _NonstandardJsonConstantError,
    )
except ImportError:  # direct script: `python3 tools/verify_model_bank.py`
    from model_bank_errors import (
        BankParseError,
        _DuplicateJsonKeyError,
        _NonstandardJsonConstantError,
    )


def dumps_manifest(document: Mapping[str, Any]) -> str:
    """Stable serialization of a model-bank document.

    See the ``model_bank`` module docstring. The golden fixture must
    round-trip through this function unchanged. Non-JSON values, including
    ``NaN`` / ``Infinity``, are rejected so the serializer cannot emit JSON
    the loader would refuse.
    """
    try:
        return json.dumps(
            document,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        ) + "\n"
    except (TypeError, ValueError, RecursionError) as exc:
        raise BankParseError(
            f"model-bank: cannot serialize manifest: {exc}"
        ) from exc


def _read_utf8(path: Path, *, kind: str) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise BankParseError(
            f"model-bank: {kind} {path} is not UTF-8: {exc}"
        ) from exc
    except (OSError, ValueError) as exc:
        # pathlib raises ValueError for an embedded NUL in the path.
        raise BankParseError(
            f"model-bank: cannot read {kind} {path}: {exc}"
        ) from exc


def _reject_duplicate_object_keys(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    seen: dict[str, Any] = {}
    for key, value in pairs:
        if key in seen:
            raise _DuplicateJsonKeyError(key)
        seen[key] = value
    return seen


def _reject_nonstandard_json_constant(token: str) -> None:
    raise _NonstandardJsonConstantError(token)


# CPython 3.11 ``json.loads`` RecursionError starts around this depth.
# CPython 3.14 can decode past it, so the named bound keeps both fail-closed.
MAX_JSON_NESTING = 1000


def _advance_json_string(character: str, escaped: bool) -> tuple[bool, bool]:
    """Return (still_in_string, next_escaped) for one character."""
    if escaped:
        return True, False
    if character == "\\":
        return True, True
    return character != '"', False


def _apply_nesting_delimiter(character: str, depth: int) -> int:
    if character in "{[":
        return depth + 1
    if character in "}]" and depth:
        return depth - 1
    return depth


def _json_nesting_depth(text: str) -> int:
    """Nesting of objects/arrays, ignoring text inside JSON strings."""
    depth = 0
    deepest = 0
    in_string = False
    escaped = False
    for character in text:
        if in_string:
            in_string, escaped = _advance_json_string(character, escaped)
            continue
        if character == '"':
            in_string = True
            continue
        depth = _apply_nesting_delimiter(character, depth)
        if depth > deepest:
            deepest = depth
    return deepest


def _refuse_excessive_json_nesting(text: str, *, source: Path) -> None:
    if _json_nesting_depth(text) >= MAX_JSON_NESTING:
        raise BankParseError(
            f"model-bank: JSON nesting exceeds parser limit in {source}"
        )


def _parse_json(text: str, *, source: Path) -> Any:
    _refuse_excessive_json_nesting(text, source=source)
    try:
        return json.loads(
            text,
            object_pairs_hook=_reject_duplicate_object_keys,
            parse_constant=_reject_nonstandard_json_constant,
        )
    except json.JSONDecodeError as exc:
        raise BankParseError(
            f"model-bank: invalid JSON in {source}: {exc}"
        ) from exc
    except _DuplicateJsonKeyError as exc:
        raise BankParseError(
            f"model-bank: duplicate object key {exc.key!r} in {source}"
        ) from exc
    except _NonstandardJsonConstantError as exc:
        raise BankParseError(
            f"model-bank: non-standard JSON constant {exc.token!r} in {source}"
        ) from exc
    except RecursionError as exc:
        # CPython 3.11 json.loads raises RecursionError around ~1000 nested
        # objects/arrays; it is not JSONDecodeError.
        raise BankParseError(
            f"model-bank: JSON nesting exceeds parser limit in {source}: {exc}"
        ) from exc
    except ValueError as exc:
        # CPython raises a bare ValueError for an integer literal longer than
        # sys.get_int_max_str_digits() (4300 by default on 3.11+). It is not a
        # JSONDecodeError, so it would escape the CLI as a traceback.
        raise BankParseError(
            f"model-bank: unreadable JSON number in {source}: {exc}"
        ) from exc
