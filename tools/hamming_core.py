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
import math
import re
import struct
from dataclasses import dataclass, field
from pathlib import Path

try:  # package import: `python3 -m tools.measure_hamming`
    from .q88_core import (
        N_INPUTS,
        N_NEURONS,
        ParseError,
        Q88RangeError,
        SelfTestFailure,
        as_finite_float,
        decode_q88,
        encode_q88_hex,
        load_model,
        parse_mem,
    )
except ImportError:  # direct script: `python3 tools/measure_hamming.py`
    from q88_core import (
        N_INPUTS,
        N_NEURONS,
        ParseError,
        Q88RangeError,
        SelfTestFailure,
        as_finite_float,
        decode_q88,
        encode_q88_hex,
        load_model,
        parse_mem,
    )

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURE_DIR = REPO_ROOT / "tools" / "fixtures" / "hamming_method"
SHIPPED_DIR = REPO_ROOT / "dataset" / "merged_v2"

# Legal 5-ch train-scaled encoder (exp-008 / exp-024). Order is the contract.
LIVE_COLUMNS: tuple[str, ...] = (
    "mem_util_pct",
    "power_w",
    "gpu_temp_c",
    "sm_clock_mhz",
    "mem_clock_mhz",
)
N_LIVE_AXONS = len(LIVE_COLUMNS)

# Frozen minmax from v3 state_telemetry train, sha lineage 74acdd0f.
# Do not refit on val/test. Copied from SynapticDistill.jl FROZEN_MINMAX.
FROZEN_MINMAX: dict[str, tuple[float, float]] = {
    "mem_util_pct": (0.0, 75.0),
    "power_w": (8.527000427246094, 302.8450012207031),
    "gpu_temp_c": (0.0, 69.0),
    "sm_clock_mhz": (180.0, 2910.0),
    "mem_clock_mhz": (405.0, 14801.0),
}
FROZEN_LINEAGE = "74acdd0f"

# Episode holdout. Session key is episode_id (ts_utc is 100% null on v3).
TRAIN_EP_LO, TRAIN_EP_HI = 0, 138
VAL_EP_LO, VAL_EP_HI = 140, 168
TEST_EP_LO, TEST_EP_HI = 170, 198
EMBARGO_EPS = frozenset((139, 169))
EPISODE_RE = re.compile(r"^gpu-(\d{6})$")
EXP024_TEST_N_TICKS = 117653

# Distill Dale / K-WTA / I-drive. 0-based: excitatory 0..11, inhibitory 12..15.
N_EXC = 12
INHIB_ROWS = tuple(range(N_EXC, N_NEURONS))
I_DRIVE_EXP024 = 0.05
I_WTA_MAX = 2
E_WTA_MIN = 2

CONDITION_METHOD_FIXTURE = "method-fixture"
CONDITION_EXP024 = "exp-024"
CONDITION_SHIPPED = "shipped-merged-v2"

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

FORBIDDEN_SENSORS = (
    "hashrate_mh_derived",
    "power_w_derived",
    "gpu_temp_c_derived",
    "reward_hint_derived",
    "tick_rate",
    "fan_speed_pct",
    "vddcr_gfx_v",
    "vram_temp_c",
    "step_idx",
)


def f32(value: float) -> float:
    """Snap ``value`` onto IEEE-754 binary32, matching Julia ``Float32``.

    The Distill sidecar does every LIF update in Float32. Python's default
    float is binary64; leaving the extra bits in would invent a third
    arithmetic that neither bank used.
    """
    return struct.unpack("=f", struct.pack("=f", float(value)))[0]


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
    number = float(raw)
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


def load_jsonl(path: Path) -> list[dict]:
    """Read a JSONL file; each non-blank line must be a JSON object."""
    if not path.is_file():
        raise ParseError(f"missing file: {path}")
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ParseError(f"{path.name}: not valid UTF-8 ({exc})") from exc
    except OSError as exc:
        raise ParseError(f"{path}: cannot be read ({exc})") from exc

    records: list[dict] = []
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
        records.append(parsed)
    return records


def select_samples(
    records: list[dict],
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

    for lineno, record in enumerate(records, start=1):
        if is_forbidden_derived(record) and not is_state_telemetry(record):
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

        raw_id = record.get("episode_id")
        index = episode_index(raw_id)
        if index is None:
            raise ParseError(
                f"line {lineno}: v3 state_telemetry row has missing or "
                f"malformed episode_id (got {raw_id!r}; expected gpu-######). "
                "Refusing to silently drop the row."
            )
        had_episode = True
        if not isinstance(raw_id, str):
            raise ParseError(f"line {lineno}: episode_id must be a string")

        keep = split == "all" or episode_split(index) == split
        if index != prev:
            if keep:
                if index in seen:
                    raise ParseError(
                        f"episode_id {raw_id} is not contiguous in file order "
                        f"for split {split}. Temporal reset needs grouped "
                        "episodes; refusing interleaved rows."
                    )
                seen.add(index)
            prev = index
        if keep:
            out.append(
                Sample(
                    episode_id=raw_id,
                    episode=index,
                    stim=encode_record(record),
                    source_line=lineno,
                )
            )

    if split != "all" and not had_episode:
        raise ParseError(
            "No episode_id on records; cannot select "
            f"{split} gpu-######. Refusing to evaluate the wrong split."
        )
    if split == "test" and not out and not allow_unsplit:
        raise ParseError(
            "No test episodes (gpu-000170..198) in this JSONL. "
            "exp-024 Hamming is k=none/k=4 on the test split; "
            "refusing to evaluate the train split."
        )
    return out


@dataclass
class LifBank:
    """One 16-unit keep-LIF population (float or Q8.8-decoded parameters)."""

    name: str
    weights: list[list[float]]  # [neuron][channel]
    decay: list[float]  # KEEP factors
    threshold: list[float]
    i_drive: float
    v: list[float] = field(default_factory=lambda: [0.0] * N_NEURONS)

    def reset(self) -> None:
        self.v = [0.0] * N_NEURONS

    def copy_params(self) -> LifBank:
        return LifBank(
            name=self.name,
            weights=[row[:] for row in self.weights],
            decay=list(self.decay),
            threshold=list(self.threshold),
            i_drive=self.i_drive,
        )


def apply_kwta(
    spikes: list[bool],
    voltage: list[float],
    k: int,
    *,
    n_exc: int = N_EXC,
    i_wta_max: int = I_WTA_MAX,
    e_wta_min: int = E_WTA_MIN,
) -> list[bool]:
    """Keep the ``k`` highest-``v`` firers with Distill's mixed E/I quota.

    Losers stay above threshold in ``voltage``; only the returned mask is
    silenced. Membrane of losers is left intact so they can compete next
    tick -- the caller resets winners only.
    """
    if k <= 0:
        return [False] * len(spikes)
    cand = [i for i, fired in enumerate(spikes) if fired]
    if not cand:
        return list(spikes)
    e_c = [i for i in cand if i < n_exc]
    i_c = [i for i in cand if i >= n_exc]
    e_c.sort(key=lambda i: voltage[i], reverse=True)
    i_c.sort(key=lambda i: voltage[i], reverse=True)
    take_i = min(i_wta_max, len(i_c), k)
    take_e = min(len(e_c), max(e_wta_min, k - take_i), k)
    take_e = min(take_e, k - take_i)
    leftover = k - take_e - take_i
    extra_e = min(leftover, max(0, len(e_c) - take_e))
    take_e += extra_e
    leftover -= extra_e
    take_i += min(leftover, max(0, min(i_wta_max, len(i_c)) - take_i))
    kept = [False] * len(spikes)
    for i in e_c[:take_e]:
        kept[i] = True
    for i in i_c[:take_i]:
        kept[i] = True
    return kept


def keep_lif_step(bank: LifBank, stim: list[float], k: int | None) -> list[bool]:
    """Advance one keep-LIF tick. Used only by this harness.

    ``stim`` is the 16-wide analog encoder output. Poisson pre-spikes exist
    in Distill only for STDP; ``learn=false`` (this measurement) injects
    analog current ``W @ stim``.
    """
    if len(stim) != N_INPUTS:
        raise ParseError(f"stimulus width {len(stim)}, expected {N_INPUTS}")
    inputs = []
    for n in range(N_NEURONS):
        acc = 0.0
        row = bank.weights[n]
        for c, s in enumerate(stim):
            acc = f32(acc + f32(row[c] * s))
        if n in INHIB_ROWS and bank.i_drive != 0.0:
            acc = f32(acc + f32(bank.i_drive))
        inputs.append(acc)

    for n in range(N_NEURONS):
        bank.v[n] = f32(f32(bank.decay[n] * bank.v[n]) + inputs[n])

    spikes = [bank.v[n] >= bank.threshold[n] for n in range(N_NEURONS)]
    if k is not None:
        spikes = apply_kwta(spikes, bank.v, k)
    for n, fired in enumerate(spikes):
        if fired:
            bank.v[n] = 0.0
    return spikes


def hamming_bits(left: list[bool], right: list[bool]) -> int:
    return sum(a != b for a, b in zip(left, right, strict=True))


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


def hidden_json_mem_mismatches(model: dict, mem_dir: Path) -> tuple[int, int]:
    """Count hidden-weight slots whose Q8.8 encoding disagrees with ``.mem``.

    This is the encoding-half check exp-024 reported as ``0/256``. It is
    not a Hamming number.
    """
    weights_path = mem_dir / "parameters_weights.mem"
    entries = parse_mem(weights_path)
    neurons = model["neurons"]
    mismatches = 0
    compared = 0
    expected = N_NEURONS * N_INPUTS
    if len(entries) != expected:
        raise ParseError(
            f"{weights_path.name}: {len(entries)} words, expected {expected}"
        )
    for i, neuron in enumerate(neurons):
        for j, weight in enumerate(neuron["weights"]):
            value = as_finite_float(weight, f"neurons[{i}].weights[{j}]")
            try:
                want = encode_q88_hex(value)
            except Q88RangeError as exc:
                raise ParseError(str(exc)) from exc
            got = entries[i * N_INPUTS + j].text.upper()
            compared += 1
            if want != got:
                mismatches += 1
    return mismatches, compared


def bank_from_json(path: Path, name: str, i_drive: float) -> tuple[LifBank, dict]:
    model = load_model(path)
    neurons = model["neurons"]
    weights = []
    decay = []
    threshold = []
    for i, neuron in enumerate(neurons):
        decay.append(
            f32(as_finite_float(neuron["decay_rate"], f"neurons[{i}].decay_rate"))
        )
        threshold.append(
            f32(as_finite_float(neuron["threshold"], f"neurons[{i}].threshold"))
        )
        weights.append(
            [
                f32(as_finite_float(w, f"neurons[{i}].weights[{j}]"))
                for j, w in enumerate(neuron["weights"])
            ]
        )
    return (
        LifBank(
            name=name,
            weights=weights,
            decay=decay,
            threshold=threshold,
            i_drive=f32(i_drive),
        ),
        model,
    )


def bank_from_mem(mem_dir: Path, name: str, i_drive: float) -> LifBank:
    """Decode thresholds, keep-factors and hidden weights from ``.mem``."""
    thresh = parse_mem(mem_dir / "parameters.mem")
    decay = parse_mem(mem_dir / "parameters_decay.mem")
    hidden = parse_mem(mem_dir / "parameters_weights.mem")
    if len(thresh) != N_NEURONS:
        raise ParseError(
            f"parameters.mem: {len(thresh)} words, expected {N_NEURONS}"
        )
    if len(decay) != N_NEURONS:
        raise ParseError(
            f"parameters_decay.mem: {len(decay)} words, expected {N_NEURONS}"
        )
    if len(hidden) != N_NEURONS * N_INPUTS:
        raise ParseError(
            f"parameters_weights.mem: {len(hidden)} words, expected "
            f"{N_NEURONS * N_INPUTS}"
        )
    weights = []
    for n in range(N_NEURONS):
        row = [
            f32(decode_q88(hidden[n * N_INPUTS + c].word)) for c in range(N_INPUTS)
        ]
        weights.append(row)
    return LifBank(
        name=name,
        weights=weights,
        decay=[f32(decode_q88(e.word)) for e in decay],
        threshold=[f32(decode_q88(e.word)) for e in thresh],
        i_drive=f32(i_drive),
    )


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
    """Compare float-JSON vs Q8.8-decoded banks on a holdout JSONL.

    Completing this function is a *measurement*, not a verdict on whether
    the Hamming is acceptable. An empty sample list is a hard failure --
    reporting 0.0% having read nothing would be believed.
    """
    if condition == CONDITION_EXP024 and mem_dir.resolve() == SHIPPED_DIR.resolve():
        raise ParseError(
            "condition exp-024 refuses dataset/merged_v2: that is the shipped "
            "ramp, not the exp-023 PASS Distill knobs scratch. Pass the "
            "scratch --mem-dir, or use --condition shipped-merged-v2 to "
            "label a different condition."
        )

    records = load_jsonl(jsonl)
    samples = select_samples(records, split)
    if not samples:
        raise ParseError(
            f"NOTHING WAS MEASURED: 0 ticks after split={split!r}. "
            "An unguarded 0.0% Hamming would be a clean-looking lie."
        )

    float_bank, model = bank_from_json(float_json, "float-json", i_drive)
    q88_bank = bank_from_mem(mem_dir, "q88-decoded", i_drive)
    mismatches, compared = hidden_json_mem_mismatches(model, mem_dir)

    unused = [c for sample in samples for c in range(N_LIVE_AXONS, N_INPUTS) if sample.stim[c] != 0.0]
    notes = []
    if unused:
        notes.append(
            f"UNUSED-AXON LEAK: {len(unused)} non-zero values on axons 5-15"
        )

    k_none = run_pair(float_bank, q88_bank, samples, None)
    k_4 = run_pair(float_bank, q88_bank, samples, 4)

    protocol = Protocol(
        condition=condition,
        weights_label=weights_label,
        float_json=float_json,
        mem_dir=mem_dir,
        encoder=(
            f"legal 5-ch train-scaled; frozen minmax lineage {FROZEN_LINEAGE}; "
            "axons 0-4 = "
            + ", ".join(LIVE_COLUMNS)
            + "; unused axons 5-15 = 0"
        ),
        split=split,
        episodes=_episode_span(samples),
        seed=seed,
        n_ticks=len(samples),
        i_drive=i_drive,
        stepper=(
            "Python keep-LIF in tools/hamming_core.py "
            "(v = decay*v + W@stim; decay is KEEP). "
            "Not the Rust crate."
        ),
        compared="float bank (snn_model.json as written, f32) vs Q8.8-decoded .mem",
    )
    return Measurement(
        protocol=protocol,
        k_none=k_none,
        k_4=k_4,
        hidden_json_mem_mismatches=mismatches,
        hidden_compared=compared,
        notes=notes,
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


def pin_matches(measurement: Measurement, expected: dict) -> list[str]:
    """How the measurement disagrees with a *method* pin, if at all.

    This is a harness-regression check (the fixture still yields the same
    Hamming). It is not a tolerance on exp-024 and must not be used as one.
    """
    failures: list[str] = []

    def _close(name: str, got: float, want: float, tol: float = 1e-9) -> None:
        if abs(got - want) > tol:
            failures.append(f"{name}: got {got}, pinned {want}")

    _close("k_none.pct", measurement.k_none.pct, float(expected["k_none_pct"]))
    _close("k_none.bits", measurement.k_none.mean_bits, float(expected["k_none_bits"]))
    _close("k_4.pct", measurement.k_4.pct, float(expected["k_4_pct"]))
    _close("k_4.bits", measurement.k_4.mean_bits, float(expected["k_4_bits"]))
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
