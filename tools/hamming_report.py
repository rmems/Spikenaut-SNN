"""Render a float-vs-Q8.8 Hamming measurement.

Deciding the numbers and describing them are different jobs. Nothing here
applies a Hamming pass/fail threshold -- issue #39 publishes the measurement
and the protocol; the tolerance is blocked on #20.
"""

from __future__ import annotations

import sys
from pathlib import Path

try:  # package import: `python3 -m tools.measure_hamming`
    from .hamming_core import EXP024_CLAIMED, FROZEN_LINEAGE, Measurement
    from .q88_core import REPO_ROOT
except ImportError:  # direct script: `python3 tools/measure_hamming.py`
    from hamming_core import EXP024_CLAIMED, FROZEN_LINEAGE, Measurement
    from q88_core import REPO_ROOT


def _stream_safe(text: str, stream) -> str:
    encoding = getattr(stream, "encoding", None) or "utf-8"
    try:
        text.encode(encoding)
    except LookupError:
        return text.encode("utf-8", "backslashreplace").decode("utf-8")
    except UnicodeEncodeError:
        return text.encode(encoding, "backslashreplace").decode(encoding)
    return text


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _k_line(prefix: str, result) -> str:
    return (
        f"  {prefix:<8} Hamming {result.pct:7.3f}%   "
        f"mean bits {result.mean_bits:.4f}   "
        f"({result.disagree_ticks}/{result.n_ticks} ticks disagree)"
    )


def _protocol_lines(measurement: Measurement) -> list[str]:
    proto = measurement.protocol
    return [
        "float-vs-Q8.8 Hamming holdout  (issue #39)",
        f"repo root: {_rel(REPO_ROOT)}",
        "",
        "PROTOCOL",
        f"  condition : {proto.condition}",
        f"  weights   : {proto.weights_label}",
        f"  float JSON: {_rel(proto.float_json)}",
        f"  mem dir   : {_rel(proto.mem_dir)}",
        f"  encoder   : {proto.encoder}",
        f"  split     : {proto.split}",
        f"  episodes  : {proto.episodes}",
        f"  ticks     : {proto.n_ticks}",
        f"  seed      : {proto.seed}",
        f"  I_DRIVE   : {proto.i_drive}  (Dale I bias on neurons 12-15)",
        f"  stepper   : {proto.stepper}",
        f"  compared  : {proto.compared}",
        (
            f"  hidden json<->mem : {measurement.hidden_json_mem_mismatches}/"
            f"{measurement.hidden_compared}"
        ),
    ]


def _results_lines(measurement: Measurement) -> list[str]:
    lines = [
        "RESULTS  (measurement, not a pass/fail gate; tolerance blocked on #20)",
        _k_line("k=none", measurement.k_none),
        _k_line("k=4", measurement.k_4),
    ]
    if measurement.notes:
        lines.append("")
        lines.append("NOTES")
        lines.extend(f"  {note}" for note in measurement.notes)
    return lines


def _exp024_reference_lines() -> list[str]:
    claimed = EXP024_CLAIMED
    return [
        "exp-024 reference (scratch weights + v3 JSONL are not in this repo)",
        f"  weights : {claimed['weights']}",
        f"  encoder : legal 5-ch train-scaled; frozen minmax lineage {FROZEN_LINEAGE}",
        f"  split   : {claimed['split']} (n={claimed['n_ticks']})",
        f"  seed    : {claimed['seed']} / {claimed['epochs']} ep",
        (
            f"  claimed : k=none {claimed['k_none_pct']}% / "
            f"{claimed['k_none_bits']} bits; "
            f"k=4 {claimed['k_4_pct']}% / "
            f"{claimed['k_4_bits']} bits; "
            f"hidden json<->mem {claimed['hidden_json_mem_mismatches']}/256"
        ),
        "  Reproduce with external paths, never by overwriting dataset/merged_v2:",
        "    python3 tools/measure_hamming.py --condition exp-024 \\",
        "        --jsonl PATH/state_telemetry.jsonl \\",
        "        --float-json PATH/snn_model.json --mem-dir PATH/",
    ]


def _condition_tail(measurement: Measurement) -> list[str]:
    proto = measurement.protocol
    if proto.condition == "method-fixture":
        return [
            "This default run is the in-repo method fixture. It proves the",
            "harness can score Hamming; it does not reproduce exp-024's",
            "figures (those weights and the 117653-tick JSONL are not shipped).",
        ]
    if proto.condition == "shipped-merged-v2":
        return [
            "This is the shipped merged_v2 ramp, labeled as a different",
            "condition. It is not the exp-023 PASS Distill knobs scratch.",
        ]
    if proto.condition != "exp-024":
        return []
    return _exp024_delta_lines(measurement)


def _exp024_comparable(measurement: Measurement) -> bool:
    """True only for the recorded exp-024 test holdout (n=117653)."""
    proto = measurement.protocol
    claimed = EXP024_CLAIMED
    if proto.split != "test":
        return False
    if proto.n_ticks != claimed["n_ticks"]:
        return False
    if measurement.k_none.n_ticks != claimed["n_ticks"]:
        return False
    episodes = proto.episodes
    return "gpu-000170" in episodes and "gpu-000198" in episodes


def _exp024_delta_lines(measurement: Measurement) -> list[str]:
    if not _exp024_comparable(measurement):
        return [
            "exp-024 deltas are N/A (not comparable): this run is not the",
            "recorded holdout (test gpu-000170..198, n=117653).",
        ]
    claimed = EXP024_CLAIMED
    delta_none = measurement.k_none.pct - claimed["k_none_pct"]
    delta_4 = measurement.k_4.pct - claimed["k_4_pct"]
    return [
        "exp-024 comparison against the claimed figures:",
        (
            f"  k=none delta_pct={delta_none:+.3f}  "
            f"delta_bits={measurement.k_none.mean_bits - claimed['k_none_bits']:+.4f}"
        ),
        (
            f"  k=4    delta_pct={delta_4:+.3f}  "
            f"delta_bits={measurement.k_4.mean_bits - claimed['k_4_bits']:+.4f}"
        ),
    ]


def render(measurement: Measurement, stream=sys.stdout) -> list[str]:
    """Format the protocol + numbers. Does not decide a Hamming gate."""
    lines = _protocol_lines(measurement)
    lines.append("")
    lines.extend(_results_lines(measurement))
    lines.append("")
    lines.extend(_exp024_reference_lines())
    tail = _condition_tail(measurement)
    if tail:
        lines.append("")
        lines.extend(tail)
    lines.append("")
    lines.append(
        "OK: measurement published. No Hamming tolerance is applied (#20)."
    )
    return lines


def report(measurement: Measurement, stream=sys.stdout) -> bool:
    """Print the measurement. Returns False only for an empty/incomplete run.

    A completed measurement with any Hamming -- including 100% -- is a
    successful publish. Empty ticks are already rejected in
    ``measure()``; this is the last backstop.
    """
    if measurement.k_none.n_ticks == 0 or measurement.k_4.n_ticks == 0:
        print(
            "FAILED: NOTHING WAS MEASURED: a K condition scored 0 ticks. "
            "An unguarded 0.0% Hamming would be a clean-looking lie.",
            file=stream,
        )
        return False
    text = "\n".join(render(measurement))
    print(_stream_safe(text, stream), file=stream)
    return True
