"""v3 state_telemetry encoder and episode holdout selection."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

try:
    from .hamming_const import (
        EMBARGO_EPS,
        FORBIDDEN_SENSORS,
        FROZEN_MINMAX,
        LIVE_COLUMNS,
        N_INPUTS,
        TEST_EP_HI,
        TEST_EP_LO,
        TRAIN_EP_HI,
        TRAIN_EP_LO,
        VAL_EP_HI,
        VAL_EP_LO,
        f32,
    )
    from .q88_core import ParseError, read_utf8_text
except ImportError:
    from hamming_const import (
        EMBARGO_EPS,
        FORBIDDEN_SENSORS,
        FROZEN_MINMAX,
        LIVE_COLUMNS,
        N_INPUTS,
        TEST_EP_HI,
        TEST_EP_LO,
        TRAIN_EP_HI,
        TRAIN_EP_LO,
        VAL_EP_HI,
        VAL_EP_LO,
        f32,
    )
    from q88_core import ParseError, read_utf8_text

EPISODE_RE = re.compile(r"^gpu-(\d{6})$")


def frozen_unit01(column: str, value: float | None) -> float:
    """``(x - lo) / (hi - lo)`` then clamp to ``[0, 1]``.

    ``None`` / missing is **0**, not an imputed neighbour or a refit
    minmax. Distill: ``T=0 stays 0`` -- ``gpu_temp_c == 0`` encodes as 0
    on this protocol (it is not dropped as a dropout tick).
    """
    lo, hi = FROZEN_MINMAX[column]
    if value is None:
        return 0.0
    span = f32(hi) - f32(lo)
    if span == 0.0:
        return 0.0
    scaled = (f32(value) - f32(lo)) / span
    return f32(min(1.0, max(0.0, scaled)))


def encode_record(record: dict) -> list[float]:
    """16-wide analog stimulus from one v3 ``state_telemetry`` row.

    Axons 0..4 are the five live columns after frozen minmax; axons 5..15
    stay 0 as unused width -- not fake channels, not first-differences.
    """
    stim = [0.0] * N_INPUTS
    for i, column in enumerate(LIVE_COLUMNS):
        stim[i] = frozen_unit01(column, _live_number(record, column))
    return stim


def _live_number(record: dict, column: str) -> float | None:
    if column not in record:
        return None
    raw = record[column]
    if raw is None:
        return None
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise ParseError(
            f"{column}: expected a finite number or null, got "
            f"{type(raw).__name__} {raw!r}"
        )
    try:
        number = float(raw)
    except (OverflowError, ValueError) as exc:
        raise ParseError(
            f"{column}: {raw!r} is not representable as a float ({exc})"
        ) from exc
    if not math.isfinite(number):
        raise ParseError(f"{column}: expected a finite number, got {raw!r}")
    return number


def episode_index(episode_id: object) -> int | None:
    """Parse ``gpu-000138`` -> 138. Does not invent an index from row order."""
    if not isinstance(episode_id, str):
        return None
    match = EPISODE_RE.fullmatch(episode_id)
    if match is None:
        return None
    return int(match.group(1))


def episode_split(index: int | None) -> str | None:
    if index is None:
        return None
    if index in EMBARGO_EPS:
        return None
    if TRAIN_EP_LO <= index <= TRAIN_EP_HI:
        return "train"
    if VAL_EP_LO <= index <= VAL_EP_HI:
        return "val"
    if TEST_EP_LO <= index <= TEST_EP_HI:
        return "test"
    return None


def is_state_telemetry(record: dict) -> bool:
    return any(column in record for column in LIVE_COLUMNS)


def is_forbidden_derived(record: dict) -> bool:
    return any(name in record and record[name] is not None for name in FORBIDDEN_SENSORS)


@dataclass(frozen=True)
class Sample:
    """One holdout tick after the encoder has accepted the row."""

    episode_id: str
    episode: int | None
    stim: list[float]
    source_line: int


def load_jsonl(path: Path) -> list[tuple[int, dict]]:
    """Read a JSONL file; each non-blank line must be a JSON object.

    Returns ``(file_lineno, record)`` so later refusals cite the true
    file line, not the compacted index after blank-line skips.
    """
    if not path.is_file():
        raise ParseError(f"missing file: {path}")
    text = read_utf8_text(path)
    records: list[tuple[int, dict]] = []
    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ParseError(
                f"{path.name}:{lineno}: not a JSON object ({exc})"
            ) from exc
        if not isinstance(parsed, dict):
            raise ParseError(
                f"{path.name}:{lineno}: expected a JSON object, got "
                f"{type(parsed).__name__}"
            )
        records.append((lineno, parsed))
    return records


def _as_row(item: object, fallback_lineno: int) -> tuple[int, dict]:
    if isinstance(item, tuple) and len(item) == 2:
        lineno, record = item
        if isinstance(lineno, int) and isinstance(record, dict):
            return lineno, record
    if isinstance(item, dict):
        return fallback_lineno, item
    raise ParseError(
        f"internal row {fallback_lineno}: expected a JSON object, got "
        f"{type(item).__name__}"
    )


def _refuse_non_v3(record: dict, lineno: int) -> None:
    if is_forbidden_derived(record):
        raise ParseError(
            f"line {lineno}: refusing *_derived / tick_rate sensors "
            "(closed form of tick_rate). Legal live columns: "
            + ", ".join(LIVE_COLUMNS)
        )
    if not is_state_telemetry(record):
        raise ParseError(
            f"line {lineno}: not v3 state_telemetry (expected "
            + ", ".join(LIVE_COLUMNS)
            + ")"
        )


def _require_episode_id(record: dict, lineno: int) -> tuple[str, int]:
    raw_id = record.get("episode_id")
    index = episode_index(raw_id)
    if index is None:
        raise ParseError(
            f"line {lineno}: v3 state_telemetry row has missing or "
            f"malformed episode_id (got {raw_id!r}; expected gpu-######). "
            "Refusing to silently drop the row."
        )
    if not isinstance(raw_id, str):
        raise ParseError(f"line {lineno}: episode_id must be a string")
    return raw_id, index


def _note_episode_boundary(
    index: int,
    raw_id: str,
    keep: bool,
    prev: int | None,
    seen: set[int],
    split: str,
) -> int:
    if index == prev:
        return index
    if keep:
        if index in seen:
            raise ParseError(
                f"episode_id {raw_id} is not contiguous in file order "
                f"for split {split}. Temporal reset needs grouped "
                "episodes; refusing interleaved rows."
            )
        seen.add(index)
    return index


def _require_split_rows(
    split: str,
    had_episode: bool,
    out: list[Sample],
    allow_unsplit: bool,
) -> None:
    if split != "all" and not had_episode:
        raise ParseError(
            f"NOTHING WAS MEASURED: 0 records for split={split!r}. "
            "The JSONL is empty; this is not a missing episode_id."
        )
    if split == "test" and not out and not allow_unsplit:
        raise ParseError(
            "No test episodes (gpu-000170..198) in this JSONL. "
            "exp-024 Hamming is k=none/k=4 on the test split; "
            "refusing to evaluate the train split."
        )


def select_samples(
    records: list,
    split: str,
    *,
    allow_unsplit: bool = False,
) -> list[Sample]:
    """Keep rows for ``split`` in file order; refuse interleaved episodes.

    ``split='all'`` (method fixture) keeps every v3 row and does not require
    ``gpu-000170..198``. ``split='test'`` is the exp-024 holdout and errors
    if no test episodes exist -- never falls back to train.

    A v3 row with a missing or malformed ``episode_id`` errors instead of
    being silently dropped, matching Distill ``filter_split``.
    """
    if split not in {"train", "val", "test", "all"}:
        raise ParseError(f"unknown split {split!r} (expected train|val|test|all)")

    out: list[Sample] = []
    seen: set[int] = set()
    prev: int | None = None
    had_episode = False

    for fallback, item in enumerate(records, start=1):
        lineno, record = _as_row(item, fallback)
        _refuse_non_v3(record, lineno)
        raw_id, index = _require_episode_id(record, lineno)
        had_episode = True
        keep = split == "all" or episode_split(index) == split
        prev = _note_episode_boundary(index, raw_id, keep, prev, seen, split)
        if keep:
            out.append(
                Sample(
                    episode_id=raw_id,
                    episode=index,
                    stim=encode_record(record),
                    source_line=lineno,
                )
            )

    _require_split_rows(split, had_episode, out, allow_unsplit)
    return out
