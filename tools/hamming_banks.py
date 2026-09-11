"""Load float-JSON and Q8.8 ``.mem`` banks for the Hamming harness."""

from __future__ import annotations

from pathlib import Path

try:
    from .hamming_const import N_INPUTS, N_NEURONS, f32
    from .hamming_lif import LifBank
    from .q88_core import (
        ParseError,
        Q88RangeError,
        as_finite_float,
        decode_q88,
        encode_q88_hex,
        load_model,
        parse_mem,
    )
except ImportError:
    from hamming_const import N_INPUTS, N_NEURONS, f32
    from hamming_lif import LifBank
    from q88_core import (
        ParseError,
        Q88RangeError,
        as_finite_float,
        decode_q88,
        encode_q88_hex,
        load_model,
        parse_mem,
    )


def _require_hidden_shape(neurons, entries, weights_name: str) -> None:
    expected = N_NEURONS * N_INPUTS
    if len(entries) != expected:
        raise ParseError(f"{weights_name}: {len(entries)} words, expected {expected}")
    if not isinstance(neurons, list) or len(neurons) != N_NEURONS:
        got = 0 if not isinstance(neurons, list) else len(neurons)
        raise ParseError(f"snn_model.json: {got} neurons, expected {N_NEURONS}")


def _row_weights(neuron: dict, index: int) -> list:
    weights = neuron["weights"]
    if not isinstance(weights, list) or len(weights) != N_INPUTS:
        got = type(weights).__name__ if not isinstance(weights, list) else len(weights)
        raise ParseError(f"neurons[{index}].weights: {got} entries, expected {N_INPUTS}")
    return weights


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
    _require_hidden_shape(neurons, entries, weights_path.name)
    for i, neuron in enumerate(neurons):
        weights = _row_weights(neuron, i)
        for j, weight in enumerate(weights):
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


def _require_keep_factor(value: float, where: str) -> float:
    if value < 0.0 or value > 1.0:
        raise ParseError(f"{where}: keep-factor {value} is outside [0, 1]")
    return value


def bank_from_json(path: Path, name: str, i_drive: float) -> tuple[LifBank, dict]:
    model = load_model(path)
    neurons = model["neurons"]
    weights = []
    decay = []
    threshold = []
    for i, neuron in enumerate(neurons):
        decay.append(
            _require_keep_factor(
                f32(as_finite_float(neuron["decay_rate"], f"neurons[{i}].decay_rate")),
                f"neurons[{i}].decay_rate",
            )
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


def _require_mem_len(entries, expected: int, name: str) -> None:
    if len(entries) != expected:
        raise ParseError(f"{name}: {len(entries)} words, expected {expected}")


def bank_from_mem(mem_dir: Path, name: str, i_drive: float) -> LifBank:
    """Decode thresholds, keep-factors and hidden weights from ``.mem``."""
    thresh = parse_mem(mem_dir / "parameters.mem")
    decay = parse_mem(mem_dir / "parameters_decay.mem")
    hidden = parse_mem(mem_dir / "parameters_weights.mem")
    _require_mem_len(thresh, N_NEURONS, "parameters.mem")
    _require_mem_len(decay, N_NEURONS, "parameters_decay.mem")
    _require_mem_len(hidden, N_NEURONS * N_INPUTS, "parameters_weights.mem")
    weights = [
        [f32(decode_q88(hidden[n * N_INPUTS + c].word)) for c in range(N_INPUTS)]
        for n in range(N_NEURONS)
    ]
    return LifBank(
        name=name,
        weights=weights,
        decay=[
            _require_keep_factor(
                f32(decode_q88(e.word)), f"parameters_decay.mem[{i}]"
            )
            for i, e in enumerate(decay)
        ],
        threshold=[f32(decode_q88(e.word)) for e in thresh],
        i_drive=f32(i_drive),
    )
