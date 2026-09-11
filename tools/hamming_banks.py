"""Load float-JSON and Q8.8 ``.mem`` banks for the Hamming harness."""

from __future__ import annotations

from pathlib import Path

try:
    from .hamming_const import (
        N_INPUTS,
        N_NEURONS,
        ParseError,
        Q88RangeError,
        as_finite_float,
        decode_q88,
        encode_q88_hex,
        f32,
        load_model,
        parse_mem,
    )
    from .hamming_lif import LifBank
except ImportError:
    from hamming_const import (
        N_INPUTS,
        N_NEURONS,
        ParseError,
        Q88RangeError,
        as_finite_float,
        decode_q88,
        encode_q88_hex,
        f32,
        load_model,
        parse_mem,
    )
    from hamming_lif import LifBank


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
        decay=[f32(decode_q88(e.word)) for e in decay],
        threshold=[f32(decode_q88(e.word)) for e in thresh],
        i_drive=f32(i_drive),
    )
