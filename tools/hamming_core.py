"""Float-vs-Q8.8 Hamming holdout core: encoder, keep-LIF, K-WTA, report data.

Shared by ``measure_hamming`` and ``hamming_selftest``. This module is the
only place in this repository that *advances* a LIF membrane. That is a
deliberate, documented exception for issue #39:

* the stepper lives in this standard-library Python harness
* it is **not** a claim that the Rust crate runs spikes
  (``src/lib.rs`` still: feeding a spike train through the graph is its
  own ticket; ``Neuron::membrane_potential`` is decoded and never advanced)

The discrete step matches SynapticDistill ``scripts/spikenaut_train.jl``
``tick!(learn=false)``:

    input = W @ stim          # analog current, not Poisson
    input[12:16] += I_DRIVE   # Dale I bias; 0.05 on the exp-024 protocol
    v = decay * v + input     # decay is a KEEP factor (Rust leak = 1 - keep)
    spikes = v >= threshold
    optional K-WTA among firers
    v[spikes] = 0

K-WTA is the Distill mixed E/I quota (``I_WTA_MAX=2``, ``E_WTA_MIN=2``,
12 excitatory + 4 inhibitory), applied only to neurons that already
crossed threshold. Health eval is ``k=none`` and never calls it; the
Hamming report still measures both because that is what exp-024 published.

Standard library only.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

try:
    from .hamming_banks import (
        bank_from_json,
        bank_from_mem,
        hidden_json_mem_mismatches,
    )
    from .hamming_const import (
        CONDITION_EXP024,
        CONDITION_METHOD_FIXTURE,
        CONDITION_SHIPPED,
        EXP024_TEST_N_TICKS,
        FIXTURE_DIR,
        FROZEN_LINEAGE,
        I_DRIVE_EXP024,
        LIVE_COLUMNS,
        N_EXC,
        SHIPPED_DIR,
        UNUSED_AXONS,
        f32,
    )
    from .hamming_encode import (
        Sample,
        encode_record,
        episode_index,
        episode_split,
        frozen_unit01,
        load_jsonl,
        select_samples,
    )
    from .hamming_lif import LifBank, apply_kwta, hamming_bits, keep_lif_step
    from .q88_core import ParseError, encode_q88_hex
except ImportError:
    from hamming_banks import (
        bank_from_json,
        bank_from_mem,
        hidden_json_mem_mismatches,
    )
    from hamming_const import (
        CONDITION_EXP024,
        CONDITION_METHOD_FIXTURE,
        CONDITION_SHIPPED,
        EXP024_TEST_N_TICKS,
        FIXTURE_DIR,
        FROZEN_LINEAGE,
        I_DRIVE_EXP024,
        LIVE_COLUMNS,
        N_EXC,
        SHIPPED_DIR,
        UNUSED_AXONS,
        f32,
    )
    from hamming_encode import (
        Sample,
        encode_record,
        episode_index,
        episode_split,
        frozen_unit01,
        load_jsonl,
        select_samples,
    )
    from hamming_lif import LifBank, apply_kwta, hamming_bits, keep_lif_step
    from q88_core import ParseError, encode_q88_hex

# Re-export the public harness surface so callers keep importing hamming_core.
__all__ = (
    "CONDITION_EXP024",
    "CONDITION_METHOD_FIXTURE",
    "CONDITION_SHIPPED",
    "EXP024_CLAIMED",
    "FIXTURE_DIR",
    "FROZEN_LINEAGE",
    "I_DRIVE_EXP024",
    "LifBank",
    "LIVE_COLUMNS",
    "Measurement",
    "N_EXC",
    "ParseError",
    "Protocol",
    "SHIPPED_DIR",
    "Sample",
    "apply_kwta",
    "bank_from_json",
    "bank_from_mem",
    "encode_q88_hex",
    "encode_record",
    "hidden_json_mem_mismatches",
    "episode_index",
    "episode_split",
    "f32",
    "frozen_unit01",
    "keep_lif_step",
    "load_expected",
    "load_jsonl",
    "measure",
    "measure_method_fixture",
    "method_fixture_paths",
    "pin_matches",
    "select_samples",
)

# Historical exp-024 figures. Published here as the named measurement;
# reproducing them needs the scratch weights + v3 JSONL, which this
# repository does not ship. The harness must not pretend merged_v2 is
# that bank.
EXP024_CLAIMED = {
    "k_none_pct": 13.187,
    "k_none_bits": 0.1608,
    "k_4_pct": 56.188,
    "k_4_bits": 1.697,
    "hidden_json_mem_mismatches": 0,
    "n_ticks": EXP024_TEST_N_TICKS,
    "seed": 123,
    "epochs": 5,
    "weights": "exp-023 PASS Distill knobs scratch (seed 123 / 5 ep)",
    "split": "v3 state_telemetry test episodes gpu-000170..198",
}


@dataclass(frozen=True)
class KResult:
    """Hamming for one K-WTA setting."""

    k: int | None
    n_ticks: int
    disagree_ticks: int
    mean_bits: float
    pct: float

    @property
    def label(self) -> str:
        return "none" if self.k is None else str(self.k)


@dataclass(frozen=True)
class Protocol:
    """Everything a published Hamming number must carry."""

    condition: str
    weights_label: str
    float_json: Path
    mem_dir: Path
    encoder: str
    split: str
    episodes: str
    seed: str
    n_ticks: int
    i_drive: float
    stepper: str
    compared: str


@dataclass
class Measurement:
    """One completed holdout comparison. Never a pass/fail on the %. """

    protocol: Protocol
    k_none: KResult
    k_4: KResult
    hidden_json_mem_mismatches: int
    hidden_compared: int
    notes: list[str] = field(default_factory=list)


def _empty_k(k: int | None) -> KResult:
    return KResult(k=k, n_ticks=0, disagree_ticks=0, mean_bits=0.0, pct=0.0)


def _k_result(k: int | None, bits_per_tick: list[int]) -> KResult:
    n = len(bits_per_tick)
    if n == 0:
        return _empty_k(k)
    disagree = sum(1 for b in bits_per_tick if b > 0)
    mean_bits = sum(bits_per_tick) / n
    return KResult(
        k=k,
        n_ticks=n,
        disagree_ticks=disagree,
        mean_bits=mean_bits,
        pct=100.0 * disagree / n,
    )


def run_pair(
    float_bank: LifBank,
    q88_bank: LifBank,
    samples: list[Sample],
    k: int | None,
) -> KResult:
    """Run both banks on ``samples`` and score per-tick Hamming."""
    left = float_bank.copy_params()
    right = q88_bank.copy_params()
    left.reset()
    right.reset()
    bits: list[int] = []
    prev: str | None = None
    for sample in samples:
        if sample.episode_id != prev:
            left.reset()
            right.reset()
            prev = sample.episode_id
        a = keep_lif_step(left, sample.stim, k)
        b = keep_lif_step(right, sample.stim, k)
        bits.append(hamming_bits(a, b))
    return _k_result(k, bits)


def _episode_span(samples: list[Sample]) -> str:
    ids = []
    seen: set[str] = set()
    for sample in samples:
        if sample.episode_id not in seen:
            seen.add(sample.episode_id)
            ids.append(sample.episode_id)
    if not ids:
        return "(none)"
    if len(ids) == 1:
        return ids[0]
    return f"{ids[0]}..{ids[-1]} ({len(ids)} episodes)"


def _refuse_exp024_on_shipped(condition: str, mem_dir: Path) -> None:
    if condition == CONDITION_EXP024 and mem_dir.resolve() == SHIPPED_DIR.resolve():
        raise ParseError(
            "condition exp-024 refuses dataset/merged_v2: that is the shipped "
            "ramp, not the exp-023 PASS Distill knobs scratch. Pass the "
            "scratch --mem-dir, or use --condition shipped-merged-v2 to "
            "label a different condition."
        )


def _unused_axon_notes(samples: list[Sample]) -> list[str]:
    unused = [
        axon
        for sample in samples
        for axon in UNUSED_AXONS
        if sample.stim[axon] != 0.0
    ]
    if not unused:
        return []
    return [f"UNUSED-AXON LEAK: {len(unused)} non-zero values on axons 5-15"]


def _protocol_for(ctx: dict, samples: list[Sample]) -> Protocol:
    return Protocol(
        condition=ctx["condition"],
        weights_label=ctx["weights_label"],
        float_json=ctx["float_json"],
        mem_dir=ctx["mem_dir"],
        encoder=(
            f"legal 5-ch train-scaled; frozen minmax lineage {FROZEN_LINEAGE}; "
            "axons 0-4 = "
            + ", ".join(LIVE_COLUMNS)
            + "; unused axons 5-15 = 0"
        ),
        split=ctx["split"],
        episodes=_episode_span(samples),
        seed=ctx["seed"],
        n_ticks=len(samples),
        i_drive=ctx["i_drive"],
        stepper=(
            "Python keep-LIF in tools/hamming_core.py "
            "(v = decay*v + W@stim; decay is KEEP). "
            "Not the Rust crate."
        ),
        compared="float bank (snn_model.json as written, f32) vs Q8.8-decoded .mem",
    )


def _require_samples(samples: list[Sample], split: str) -> None:
    """Empty holdout is a hard failure -- never a clean-looking 0.0%."""
    if not samples:
        raise ParseError(
            f"NOTHING WAS MEASURED: 0 ticks after split={split!r}. "
            "An unguarded 0.0% Hamming would be a clean-looking lie."
        )


def measure(
    *,
    float_json: Path,
    mem_dir: Path,
    jsonl: Path,
    split: str,
    condition: str,
    seed: str,
    i_drive: float,
    weights_label: str,
) -> Measurement:
    """Compare float-JSON vs Q8.8-decoded banks on a holdout JSONL."""
    _refuse_exp024_on_shipped(condition, mem_dir)
    samples = select_samples(load_jsonl(jsonl), split)
    _require_samples(samples, split)
    float_bank, model = bank_from_json(float_json, "float-json", i_drive)
    q88_bank = bank_from_mem(mem_dir, "q88-decoded", i_drive)
    mismatches, compared = hidden_json_mem_mismatches(model, mem_dir)
    return Measurement(
        protocol=_protocol_for(
            {
                "condition": condition,
                "weights_label": weights_label,
                "float_json": float_json,
                "mem_dir": mem_dir,
                "split": split,
                "seed": seed,
                "i_drive": i_drive,
            },
            samples,
        ),
        k_none=run_pair(float_bank, q88_bank, samples, None),
        k_4=run_pair(float_bank, q88_bank, samples, 4),
        hidden_json_mem_mismatches=mismatches,
        hidden_compared=compared,
        notes=_unused_axon_notes(samples),
    )


def method_fixture_paths() -> dict[str, Path]:
    return {
        "float_json": FIXTURE_DIR / "snn_model.json",
        "mem_dir": FIXTURE_DIR,
        "jsonl": FIXTURE_DIR / "holdout.jsonl",
        "expect": FIXTURE_DIR / "expected.json",
    }


def measure_method_fixture() -> Measurement:
    paths = method_fixture_paths()
    return measure(
        float_json=paths["float_json"],
        mem_dir=paths["mem_dir"],
        jsonl=paths["jsonl"],
        split="all",
        condition=CONDITION_METHOD_FIXTURE,
        seed="n/a (analog current; Poisson unused when learn=false)",
        i_drive=0.0,
        weights_label=(
            "in-repo method fixture tools/fixtures/hamming_method "
            "(NOT exp-023 scratch, NOT shipped merged_v2 ramp)"
        ),
    )


def load_expected(path: Path) -> dict:
    if not path.is_file():
        raise ParseError(f"missing expected pin: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ParseError(f"{path.name}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ParseError(f"{path.name}: expected a JSON object")
    return payload


def _pin_close(failures: list[str], name: str, got: float, want: float) -> None:
    if abs(got - want) > 1e-9:
        failures.append(f"{name}: got {got}, pinned {want}")


def pin_matches(measurement: Measurement, expected: dict) -> list[str]:
    """How the measurement disagrees with a *method* pin, if at all.

    This is a harness-regression check (the fixture still yields the same
    Hamming). It is not a tolerance on exp-024 and must not be used as one.
    """
    failures: list[str] = []
    _pin_close(failures, "k_none.pct", measurement.k_none.pct, float(expected["k_none_pct"]))
    _pin_close(
        failures, "k_none.bits", measurement.k_none.mean_bits, float(expected["k_none_bits"])
    )
    _pin_close(failures, "k_4.pct", measurement.k_4.pct, float(expected["k_4_pct"]))
    _pin_close(failures, "k_4.bits", measurement.k_4.mean_bits, float(expected["k_4_bits"]))
    if measurement.k_none.n_ticks != int(expected["n_ticks"]):
        failures.append(
            f"n_ticks: got {measurement.k_none.n_ticks}, "
            f"pinned {expected['n_ticks']}"
        )
    if measurement.hidden_json_mem_mismatches != int(
        expected["hidden_json_mem_mismatches"]
    ):
        failures.append(
            "hidden json<->mem: got "
            f"{measurement.hidden_json_mem_mismatches}, pinned "
            f"{expected['hidden_json_mem_mismatches']}"
        )
    return failures
