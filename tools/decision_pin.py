"""Golden pin for the output-row decision contract (Linear RM-1328).

``decision_core`` is the Python reference. ``src/decision.rs`` is the Rust
one. This module names the cases both languages must agree on -- normal,
tie, all-equal, non-finite, malformed, a confidence abstention, and the
shipped checkpoint's Distill ordering -- and writes them to
``tools/fixtures/decision/expected.json``.

``tests/decision.rs`` reads the same file. A change to either implementation,
or to a case, fails one of the two jobs.

Standard library only. ASCII hyphens only (cp1252-safe).
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

try:
    from .decision_core import (
        CONFIDENCE_FORMULA,
        CONTRACT_ID,
        OUTPUT_WIDTH,
        SHIPPED_VOCABULARY,
        SUPERVISOR_VOCABULARY,
        TIE_BREAK,
        DecisionConfig,
        DecisionError,
        decide,
        decision_as_dict,
        error_as_dict,
        replay_tick,
        score_readout,
    )
    from .hamming_const import N_NEURONS, REPO_ROOT
    from .q88_core import (
        MEM_OUTPUT,
        ParseError,
        Q88RangeError,
        as_finite_float,
        decode_q88,
        encode_q88_hex,
        parse_mem,
        read_utf8_text,
    )
except ImportError:
    from decision_core import (
        CONFIDENCE_FORMULA,
        CONTRACT_ID,
        OUTPUT_WIDTH,
        SHIPPED_VOCABULARY,
        SUPERVISOR_VOCABULARY,
        TIE_BREAK,
        DecisionConfig,
        DecisionError,
        decide,
        decision_as_dict,
        error_as_dict,
        replay_tick,
        score_readout,
    )
    from hamming_const import N_NEURONS, REPO_ROOT
    from q88_core import (
        MEM_OUTPUT,
        ParseError,
        Q88RangeError,
        as_finite_float,
        decode_q88,
        encode_q88_hex,
        parse_mem,
        read_utf8_text,
    )

EXPECTED_DECISION = (
    REPO_ROOT / "tools" / "fixtures" / "decision" / "expected.json"
)
SHIPPED_MODEL = REPO_ROOT / "dataset" / "merged_v2" / "snn_model.json"

# Fields pin_failures compares. ``note`` is editorial and exempt.
_PINNED_METADATA = (
    "contract",
    "vocabulary",
    "tie_break",
    "confidence_formula",
    "output_width",
    "supervisor_vocabulary_unbound",
    "n_outputs_json",
    "output_weight_count",
)

_SHIPPED_CFG = {
    "vocabulary": list(SHIPPED_VOCABULARY),
    "confidence_floor": 0.0,
    "abstain_on_tie": False,
}

# IEEE-754 binary64 max; finite input whose derived margin/confidence overflow.
_F64_MAX = struct.unpack(">d", bytes.fromhex("7fefffffffffffff"))[0]


def _f64_bits(value: float) -> str:
    """IEEE-754 binary64 bit pattern as 16 hex digits, MSB first."""
    return struct.pack(">d", value).hex()


def _config_of(spec: dict | None) -> DecisionConfig:
    body = spec if spec is not None else _SHIPPED_CFG
    return DecisionConfig.new(
        body["vocabulary"],
        body["confidence_floor"],
        body["abstain_on_tie"],
    )


def _run_case(row: list[float], config_spec: dict | None) -> dict:
    try:
        config = _config_of(config_spec)
        result = {"ok": True, **decision_as_dict(decide(row, config))}
    except DecisionError as exc:
        result = {"ok": False, **error_as_dict(exc)}
    return result


def _finite_case(
    name: str,
    row: list[float],
    config: dict | None = None,
    extra: dict | None = None,
) -> dict:
    case = {
        "name": name,
        "row_bits": [_f64_bits(value) for value in row],
        "row": row,
        "config": config if config is not None else _SHIPPED_CFG,
        **_run_case(row, config),
    }
    if extra:
        case.update(extra)
    return case


def _non_finite_case(name: str, row: list[float]) -> dict:
    return {
        "name": name,
        "row_bits": [_f64_bits(value) for value in row],
        "row": None,
        "config": _SHIPPED_CFG,
        **_run_case(row, None),
    }


def _shipped_readout() -> list[float]:
    entries = parse_mem(MEM_OUTPUT)
    return [decode_q88(entry.word) for entry in entries]


def _load_json_object(path: Path) -> dict:
    """Decode a JSON object, wrapping decoder failures as ``ParseError``.

    CPython 3.11+ raises a bare ``ValueError`` (not ``JSONDecodeError``) for
    an integer literal longer than ``sys.get_int_max_str_digits()``, and a
    ``RecursionError`` for a document nested past the decoder limit. Both
    must take the documented status-2 path, same as ``q88_core``.
    """
    try:
        payload = json.loads(read_utf8_text(path))
    except json.JSONDecodeError as exc:
        raise ParseError(f"{path.name}: {exc}") from exc
    except RecursionError as exc:
        raise ParseError(
            f"{path.name}: JSON nesting exceeds decoder limit ({exc})"
        ) from exc
    except ValueError as exc:
        raise ParseError(f"{path.name}: unreadable JSON number: {exc}") from exc
    if not isinstance(payload, dict):
        raise ParseError(f"{path.name}: expected an object")
    return payload


def load_shipped_model(path: Path = SHIPPED_MODEL) -> dict:
    """Load ``snn_model.json`` as an object, or raise ``ParseError``.

    Missing, unreadable, invalid-UTF-8, malformed JSON, CPython oversized
    JSON integers, and decoder ``RecursionError`` must exit the documented
    status-2 path. ``decision_parity.main`` only catches ``ParseError``.
    """
    if not path.is_file():
        raise ParseError(f"missing shipped sidecar: {path}")
    return _load_json_object(path)


def assert_output_json_mem_parity(model: dict, mem_entries: list) -> None:
    """Refuse JSON ``output_weights`` that disagree with neuron-major ``.mem``.

    Shape-only checks let value or channel-order drift pass while decisions
    still score the ``.mem`` image. Snap each JSON scalar through Q8.8 and
    compare the hex word, in Distill neuron-major order.
    """
    neurons = model.get("neurons")
    if not isinstance(neurons, list) or len(neurons) != N_NEURONS:
        got = 0 if not isinstance(neurons, list) else len(neurons)
        raise ParseError(f"{SHIPPED_MODEL.name}: expected {N_NEURONS} neurons, got {got}")
    expected = N_NEURONS * OUTPUT_WIDTH
    if len(mem_entries) != expected:
        raise ParseError(
            f"{MEM_OUTPUT.name}: {len(mem_entries)} words, expected {expected}"
        )
    for index, neuron in enumerate(neurons):
        if not isinstance(neuron, dict):
            raise ParseError(
                f"neurons[{index}]: expected a JSON object, got "
                f"{type(neuron).__name__}"
            )
        weights = neuron.get("output_weights")
        if not isinstance(weights, list) or len(weights) != OUTPUT_WIDTH:
            got = (
                type(weights).__name__
                if not isinstance(weights, list)
                else len(weights)
            )
            raise ParseError(
                f"neurons[{index}].output_weights: {got} entries, "
                f"expected {OUTPUT_WIDTH}-wide"
            )
        for channel, weight in enumerate(weights):
            where = f"neurons[{index}].output_weights[{channel}]"
            value = as_finite_float(weight, where)
            try:
                want = encode_q88_hex(value)
            except Q88RangeError as exc:
                raise ParseError(str(exc)) from exc
            mem_index = index * OUTPUT_WIDTH + channel
            got = mem_entries[mem_index].text.upper()
            if want != got:
                raise ParseError(
                    f"{where}: JSON encodes {want}, "
                    f"{MEM_OUTPUT.name}[{mem_index}] is {got}"
                )


def _n_outputs_json() -> int:
    model = load_shipped_model()
    n_outputs = model.get("n_outputs")
    if n_outputs != OUTPUT_WIDTH:
        raise ParseError(
            f"{SHIPPED_MODEL.name}: n_outputs {n_outputs!r}, expected {OUTPUT_WIDTH}"
        )
    assert_output_json_mem_parity(model, parse_mem(MEM_OUTPUT))
    return int(n_outputs)


def _checkpoint_cases() -> list[dict]:
    readout = _shipped_readout()
    if len(readout) != N_NEURONS * OUTPUT_WIDTH:
        raise ParseError(
            f"{MEM_OUTPUT.name}: {len(readout)} words, expected {N_NEURONS * OUTPUT_WIDTH}"
        )
    spikes = [False] * N_NEURONS
    spikes[0] = True
    row = list(score_readout(readout, spikes))
    # Neuron 0's three words, Distill (comfort, temp, power) order.
    expected_row = readout[:OUTPUT_WIDTH]
    if row != expected_row:
        raise ParseError(
            f"neuron-0 spike scored {row}, expected the first three mem words {expected_row}"
        )
    decision = replay_tick(readout, spikes)
    case = _finite_case(
        "checkpoint_neuron0_spike",
        row,
        extra={
            "spikes": [int(flag) for flag in spikes],
            "readout_prefix": expected_row,
            "winning_channel": "comfort",
        },
    )
    replayed = decision_as_dict(decision)
    for key, value in replayed.items():
        if case[key] != value:
            raise ParseError(
                f"checkpoint_neuron0_spike: replay_tick {key}={value!r} "
                f"disagrees with decide {case[key]!r}"
            )
    if case["winning_action"] != "comfort":
        raise ParseError(
            "checkpoint_neuron0_spike: shipped neuron 0 must keep comfort "
            f"as index 0; got {case['winning_action']!r}"
        )
    return [case]


def named_cases() -> list[dict]:
    """Every class the contract names, plus the shipped-ordering pin."""
    cases = [
        _finite_case("normal_comfort", [0.9, 0.2, 0.1]),
        _finite_case("normal_temp", [0.1, 0.8, 0.2]),
        _finite_case("normal_power", [0.1, 0.2, 0.9]),
        _finite_case("tie_first_two", [0.5, 0.5, 0.1]),
        _finite_case("tie_outer", [0.5, 0.1, 0.5]),
        _finite_case("all_equal", [0.4, 0.4, 0.4]),
        _finite_case("all_zero", [0.0, 0.0, 0.0]),
        _finite_case("negative_argmax", [-0.1, -0.5, -0.2]),
        _finite_case(
            "high_confidence_comfort",
            [1.0, 0.0, 0.0],
        ),
        _finite_case(
            "abstain_on_tie",
            [0.5, 0.5, 0.1],
            {
                "vocabulary": list(SHIPPED_VOCABULARY),
                "confidence_floor": 0.0,
                "abstain_on_tie": True,
            },
        ),
        _finite_case(
            "low_confidence_abstain",
            [0.51, 0.49, 0.0],
            {
                "vocabulary": list(SHIPPED_VOCABULARY),
                "confidence_floor": 0.5,
                "abstain_on_tie": False,
            },
        ),
        _finite_case(
            "supervisor_five_wide_pause",
            [0.0, 0.1, 0.2, 0.9, 0.0],
            {
                "vocabulary": list(SUPERVISOR_VOCABULARY),
                "confidence_floor": 0.0,
                "abstain_on_tie": False,
            },
        ),
        _finite_case("empty_row", []),
        _finite_case("width_short", [0.1, 0.2]),
        _finite_case("width_long", [0.1, 0.2, 0.3, 0.4]),
        _finite_case(
            "empty_vocabulary",
            [0.1, 0.2, 0.3],
            {
                "vocabulary": [],
                "confidence_floor": 0.0,
                "abstain_on_tie": False,
            },
        ),
        _non_finite_case("nan_first", [float("nan"), 0.2, 0.1]),
        _non_finite_case("pos_inf", [0.1, float("inf"), 0.1]),
        _non_finite_case("neg_inf", [0.1, 0.2, float("-inf")]),
        _non_finite_case(
            "all_non_finite",
            [float("nan"), float("inf"), float("-inf")],
        ),
        _finite_case(
            "overflow_confidence",
            [_F64_MAX, -_F64_MAX, -_F64_MAX],
        ),
        _finite_case(
            "overflow_denominator",
            [_F64_MAX, _F64_MAX / 2.0, 0.0],
        ),
        _finite_case(
            "overflow_denominator_floor",
            [_F64_MAX, _F64_MAX / 2.0, 0.0],
            {
                "vocabulary": list(SHIPPED_VOCABULARY),
                "confidence_floor": 0.5,
                "abstain_on_tie": False,
            },
        ),
    ]
    cases.extend(_checkpoint_cases())
    return cases


def build_pin() -> dict:
    n_outputs = _n_outputs_json()
    readout = _shipped_readout()
    payload = {
        "contract": CONTRACT_ID,
        "vocabulary": list(SHIPPED_VOCABULARY),
        "tie_break": TIE_BREAK,
        "confidence_formula": CONFIDENCE_FORMULA,
        "output_width": OUTPUT_WIDTH,
        "supervisor_vocabulary_unbound": list(SUPERVISOR_VOCABULARY),
        "n_outputs_json": n_outputs,
        "output_weight_count": len(readout),
        "note": (
            "Cross-language pin for the output-row decision contract "
            "(Linear RM-1328): tools/decision_core.py vs Rust "
            "spikenaut_snn::decision. Shipped vocabulary is Distill "
            "(comfort, temp, power); RM-1150 ALLOW/WARN/THROTTLE/PAUSE/"
            "YIELD_GPU is recorded unbound. Regenerate with "
            "`python3 tools/decision_parity.py --write`."
        ),
        "cases": named_cases(),
    }
    # json.dumps/loads is the on-disk form; compare against that so a
    # float that does not round-trip cannot pass in memory and fail in CI.
    return json.loads(json.dumps(payload))


def write_pin(path: Path = EXPECTED_DECISION) -> dict:
    payload = build_pin()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def load_pin(path: Path = EXPECTED_DECISION) -> dict:
    if not path.is_file():
        raise ParseError(f"missing expected pin: {path}")
    return _load_json_object(path)


def pin_failures(reference: dict, pinned: dict) -> list[str]:
    """Every compared field that disagrees. ``note`` is exempt."""
    failures: list[str] = []
    for key in _PINNED_METADATA:
        if reference.get(key) != pinned.get(key):
            failures.append(
                f"{key}: pinned {pinned.get(key)!r}, reference {reference.get(key)!r}"
            )
    want = reference.get("cases")
    got = pinned.get("cases")
    if not isinstance(want, list) or not isinstance(got, list):
        failures.append("cases: both sides must be arrays")
        return failures
    if len(want) != len(got):
        failures.append(f"cases: pinned {len(got)}, reference {len(want)}")
        return failures
    for index, (expected, actual) in enumerate(zip(want, got, strict=True)):
        if not isinstance(expected, dict) or not isinstance(actual, dict):
            failures.append(f"cases[{index}]: both sides must be objects")
            continue
        name = expected.get("name")
        if actual.get("name") != name:
            failures.append(
                f"cases: pinned name {actual.get('name')!r}, reference {name!r}"
            )
            continue
        if expected != actual:
            failures.append(f"cases[{name}]: pin disagrees with the reference")
    return failures
