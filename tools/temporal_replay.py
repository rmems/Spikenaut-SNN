"""Prepare single-lane replay frames or SpikeStream events from one live capture."""

import argparse
import json
from pathlib import Path
import sys

from .hamming_const import FROZEN_MINMAX, LIVE_COLUMNS, f32
from .model_bank_errors import BankParseError
from .model_bank_json import _parse_json
from .q88_core import ParseError, Q88RangeError, as_finite_float, decode_q88, encode_q88
from .replay_inputs import _read_input_bytes, _utf8_or_refuse

DELTA = 1 / 32


def five(row, field):
    values = row.get(field)
    if not isinstance(values, list) or len(values) != len(LIVE_COLUMNS):
        raise ParseError(f"{field} needs five measured values in live-lane order")
    return [f32(as_finite_float(value, field)) for value in values]


def unit_values(raw):
    unit = []
    for name, value in zip(LIVE_COLUMNS, raw, strict=True):
        lo, hi = FROZEN_MINMAX[name]
        # Match the bridge's binary32 subtraction and division.
        scaled = f32(f32(value - lo) / f32(hi - lo))
        unit.append(max(0.0, min(1.0, scaled)))
    return unit


def sample_tick(row, previous):
    if not isinstance(row, dict) or "event" in row:
        raise ParseError("expected a successful capture sample")
    tick = row.get("tick")
    if isinstance(tick, bool) or not isinstance(tick, int) or tick < 0:
        raise ParseError("tick must be a nonnegative integer")
    if previous is not None and tick != previous + 1:
        raise ParseError("tick gap or reset; supply one contiguous capture")
    return tick


def logged_frame(row):
    stim = five(row, "stim")
    gain = as_finite_float(row.get("gain"), "gain")
    if gain < 0 or sum(value != 0 for value in stim) > 1:
        raise ParseError("expected a nonnegative gain and single-lane baseline")
    if any(value != decode_q88(encode_q88(value)) for value in stim):
        raise ParseError("baseline stim must contain the recorded Q8.8 wire values")
    return stim, gain


def delta_frame(raw, anchors):
    current = unit_values(raw)
    stim = [0.0] * len(current)
    if not anchors:
        anchors.extend(current)
        return stim
    changes = [abs(value - ref) for value, ref in zip(current, anchors, strict=True)]
    lane = max(range(len(changes)), key=changes.__getitem__)
    if changes[lane] >= DELTA:
        stim[lane] = 1.0
        anchors[lane] = current[lane]
    return stim


def frames(source: Path, encoding: str):
    """Validate the complete capture before its frames can reach a replay file."""
    result = []
    prior_tick = None
    anchors = []
    payload = _read_input_bytes(source, "live capture")
    lines = _utf8_or_refuse(payload, source, "live capture").splitlines()
    for line_no, line in enumerate(lines, 1):
        try:
            row = _parse_json(line, source=source)
            tick = sample_tick(row, prior_tick)
            raw = five(row, "raw")
            stim, gain = logged_frame(row)
            if encoding == "delta":
                stim, gain = delta_frame(raw, anchors), 1.0
            result.append((tick, stim, gain))
            prior_tick = tick
        except (ParseError, BankParseError, Q88RangeError, OverflowError) as exc:
            raise ParseError(f"line {line_no}: {exc}") from exc
    if not result:
        raise ParseError("empty capture")
    return result


def main(argv=None):
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("capture", type=Path)
    cli.add_argument("--encoding", choices=("logged", "delta"), required=True)
    cli.add_argument("--format", choices=("replay", "events"), default="replay")
    args = cli.parse_args(argv)
    try:
        rows = frames(args.capture, args.encoding)
        for tick, stim, gain in rows:
            if args.format == "events":
                for axon, value in enumerate(stim):
                    if value:
                        print(tick, axon, sep="\t")
            else:
                # replay_log searches for the exact compact key '"stim":['.
                print(json.dumps({"tick": tick, "stim": stim, "gain": gain}, separators=(",", ":")))
    except (ParseError, OSError, UnicodeError) as exc:
        print(f"Temporal replay refused: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
