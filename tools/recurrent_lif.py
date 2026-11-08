# SPDX-License-Identifier: MIT OR Apache-2.0
"""Recurrent keep-LIF variant with explicit topology, delay and state (GH #3).

The shipped exp-025 bank is feed-forward: ``v = keep * v + W @ stim``.
This module defines the one recurrent extension this repository evaluates
and the checkpoint block that declares it, so a replay can never apply
recurrence a checkpoint did not spell out, and a checkpoint that carries
recurrence is never stepped feed-forward by a consumer in this repository:
``replay_core`` applies the block, ``recurrence_eval`` and the Hamming
harness (``hamming_banks.bank_from_json``) refuse it through
``require_feed_forward``, and the Rust loader rejects the unknown key.

Contract (``spikenaut.recurrence.v1``)
--------------------------------------
* **Topology.** A dense ``16 x 16`` matrix ``R[post][pre]`` of signed
  values on the Q8.8 grid, stored post-major like the hidden ``weights``
  (``[neuron][channel]``). Dale's law holds on the *presynaptic* side:
  every entry in a column whose source neuron is ``inhibitory`` is
  ``<= 0``, every entry in an excitatory column is ``>= 0``, and the
  diagonal (self-connection) is exactly ``0``. The ``inhibitory`` flags
  are the checkpoint's own (exp-025: neurons 12-15).
* **Delay.** Exactly one tick. The feedback current at tick ``t`` is
  ``R @ s(t-1)``, where ``s(t-1)`` is the spike vector the bank *emitted*
  at the previous tick -- after K-WTA, i.e. the same vector the readout
  scored. There is no zero-delay (same-tick) recurrence.
* **State.** The membrane ``v`` plus ``s(t-1)``. Both reset at every
  session (``episode_id``) boundary, so the first tick of a session has
  zero feedback current.
* **Order inside a tick.** ``input = W @ stim``, then ``+ I_DRIVE`` on the
  Dale-I rows, then ``+ R @ s(t-1)``; ``v = keep * v + input``; fire at
  ``v >= threshold``; optional K-WTA; reset winners. Every term is
  accumulated in binary32, matching ``hamming_lif.keep_lif_step``.
* **Export.** ``parameters_recurrent_weights.mem``: 256 signed Q8.8 words,
  post-major (word ``post * 16 + pre``). The four feed-forward images keep
  their existing layout, so a consumer without recurrence reads the same
  bytes it always did -- it just must refuse a checkpoint that declares
  this block (``require_feed_forward`` below; ``replay_core`` implements
  the block, the Rust crate and the FPGA path do not, and say so).

With ``R == 0`` the step is bit-identical to the feed-forward reference,
which ``tests/test_recurrence.py`` asserts. Standard library only.
"""

from __future__ import annotations

from dataclasses import dataclass, field

try:  # package import: `python3 -m tools.recurrence_eval`
    from .hamming_const import N_NEURONS, f32
    from .hamming_lif import LifBank, keep_lif_step
    from .q88_core import (
        ParseError,
        Q88RangeError,
        as_finite_float,
        decode_q88,
        encode_q88,
        encode_q88_hex,
    )
except ImportError:  # direct script: `python3 tools/recurrence_eval.py`
    from hamming_const import N_NEURONS, f32
    from hamming_lif import LifBank, keep_lif_step
    from q88_core import (
        ParseError,
        Q88RangeError,
        as_finite_float,
        decode_q88,
        encode_q88,
        encode_q88_hex,
    )

RECURRENCE_SCHEMA = "spikenaut.recurrence.v1"
RECURRENCE_KEY = "recurrence"
RECURRENCE_DELAY_TICKS = 1
RECURRENT_MEM_NAME = "parameters_recurrent_weights.mem"
_BLOCK_KEYS = ("schema", "delay_ticks", "topology", "weights")
STATE_SEMANTICS = (
    "v and s(t-1) reset together at every episode_id boundary; "
    "the first tick of a session has zero feedback"
)
ORDER_SEMANTICS = (
    "input = W@stim (+ I_DRIVE on Dale-I rows) + R@s(t-1); "
    "v = keep*v + input; fire; optional K-WTA; reset winners; "
    "s(t-1) is the emitted (post-K-WTA) spike vector"
)


@dataclass(frozen=True)
class Recurrence:
    """A validated recurrence block. ``weights`` is ``R[post][pre]``."""

    topology: str
    weights: tuple[tuple[float, ...], ...]
    delay_ticks: int = RECURRENCE_DELAY_TICKS
    schema: str = RECURRENCE_SCHEMA

    def nonzero(self) -> int:
        # Truthiness is the exact-zero test: ``0.0`` and ``-0.0`` are the
        # only falsy floats, and every weight here passed ``_on_q88_grid``.
        return sum(1 for row in self.weights for w in row if w)

    def feedback(self, prev_spikes: list[bool]) -> list[float]:
        """``R @ s(t-1)`` accumulated in binary32, pre index ascending.

        Only fired presynaptic neurons contribute; a silent one adds an
        exact ``0.0`` and is skipped, so this is the same sum.
        """
        if len(prev_spikes) != N_NEURONS:
            raise ParseError(
                f"previous spike vector width {len(prev_spikes)}, "
                f"expected {N_NEURONS}"
            )
        out: list[float] = []
        for post in range(N_NEURONS):
            acc = 0.0
            row = self.weights[post]
            for pre, fired in enumerate(prev_spikes):
                if fired:
                    acc = f32(acc + row[pre])
            out.append(acc)
        return out


@dataclass
class RecurrentState:
    """The one extra state variable: last emitted spike vector."""

    prev_spikes: list[bool] = field(
        default_factory=lambda: [False] * N_NEURONS
    )

    def reset(self) -> None:
        self.prev_spikes = [False] * N_NEURONS


def recurrent_lif_step(
    bank: LifBank,
    stim: list[float],
    k: int | None,
    recurrence: Recurrence,
    state: RecurrentState,
) -> list[bool]:
    """One recurrent tick; see the module contract for the exact order."""
    feedback = recurrence.feedback(state.prev_spikes)
    spikes = keep_lif_step(bank, stim, k, feedback=feedback)
    state.prev_spikes = list(spikes)
    return spikes


def _on_q88_grid(value: object, where: str) -> float:
    """``value`` as f32, refusing anything not exactly on the Q8.8 grid.

    Recurrent weights are a declared artifact, not a float to be rounded
    at load time: a value the ``.mem`` image cannot hold bit-for-bit is a
    contract violation, not something to snap quietly.
    """
    number = f32(as_finite_float(value, where))
    try:
        snapped = f32(decode_q88(encode_q88(number)))
    except Q88RangeError as exc:
        raise ParseError(f"{where}: {exc}") from exc
    if snapped != number:
        raise ParseError(
            f"{where}: {value!r} is not on the Q8.8 grid "
            f"(nearest {snapped!r})"
        )
    return number


def dale_flags(model: dict, source: str) -> list[bool]:
    flags: list[bool] = []
    for i, neuron in enumerate(model["neurons"]):
        flag = neuron.get("inhibitory") if isinstance(neuron, dict) else None
        if not isinstance(flag, bool):
            raise ParseError(
                f"{source}: neurons[{i}].inhibitory must be a boolean when "
                f"a {RECURRENCE_KEY!r} block is present, got {flag!r}"
            )
        flags.append(flag)
    return flags


def _require_block_keys(block: object, source: str) -> dict:
    """The block is a JSON object carrying exactly the schema's keys."""
    if not isinstance(block, dict):
        raise ParseError(
            f"{source}: {RECURRENCE_KEY!r} must be a JSON object, got "
            f"{type(block).__name__} (omit the key for a feed-forward "
            "checkpoint)"
        )
    unknown = sorted(set(block) - set(_BLOCK_KEYS))
    missing = [key for key in _BLOCK_KEYS if key not in block]
    if unknown:
        raise ParseError(
            f"{source}: {RECURRENCE_KEY}: unknown key(s) {unknown} "
            f"(schema {RECURRENCE_SCHEMA} knows {list(_BLOCK_KEYS)})"
        )
    if missing:
        raise ParseError(
            f"{source}: {RECURRENCE_KEY}: missing key(s) {missing}"
        )
    return block


def _require_block_shape(block: object, source: str) -> dict:
    """Keys, schema id, delay and topology of a declared recurrence block."""
    block = _require_block_keys(block, source)
    if block["schema"] != RECURRENCE_SCHEMA:
        raise ParseError(
            f"{source}: {RECURRENCE_KEY}.schema must be "
            f"{RECURRENCE_SCHEMA!r}, got {block['schema']!r}"
        )
    delay = block["delay_ticks"]
    if isinstance(delay, bool) or delay != RECURRENCE_DELAY_TICKS:
        raise ParseError(
            f"{source}: {RECURRENCE_KEY}.delay_ticks must be "
            f"{RECURRENCE_DELAY_TICKS} (the only implemented delay), "
            f"got {delay!r}"
        )
    topology = block["topology"]
    if not isinstance(topology, str) or not topology.strip():
        raise ParseError(
            f"{source}: {RECURRENCE_KEY}.topology must be a non-empty string"
        )
    return block


def _validated_rows(
    block: dict, flags: list[bool], source: str
) -> list[list[float]]:
    where = f"{source}: {RECURRENCE_KEY}.weights"
    rows = _require_list_of(block["weights"], where, "rows")
    out: list[list[float]] = []
    for post, row in enumerate(rows):
        row = _require_list_of(row, f"{where}[{post}]", "entries")
        out.append(
            [
                _dale_checked(
                    _on_q88_grid(value, f"{where}[{post}][{pre}]"),
                    post,
                    pre,
                    flags[pre],
                    where,
                )
                for pre, value in enumerate(row)
            ]
        )
    return out


def _require_list_of(value: object, where: str, what: str) -> list:
    """``value`` is a list of exactly ``N_NEURONS`` items."""
    if not isinstance(value, list) or len(value) != N_NEURONS:
        got = len(value) if isinstance(value, list) else type(value).__name__
        raise ParseError(f"{where}: {got} {what}, expected {N_NEURONS}")
    return value


def _dale_checked(
    value: float, post: int, pre: int, pre_inhibitory: bool, where: str
) -> float:
    # ``value`` is a finite Q8.8 grid float (``_on_q88_grid`` ran first), so
    # truthiness is the exact "not 0" test without a float-equality compare.
    if post == pre and value:
        raise ParseError(
            f"{where}[{post}][{pre}]: self-connection must be 0, got {value!r}"
        )
    if pre_inhibitory and value > 0.0:
        raise ParseError(
            f"{where}[{post}][{pre}]: presynaptic neuron {pre} is inhibitory, "
            f"entry must be <= 0, got {value!r}"
        )
    if not pre_inhibitory and value < 0.0:
        raise ParseError(
            f"{where}[{post}][{pre}]: presynaptic neuron {pre} is excitatory, "
            f"entry must be >= 0, got {value!r}"
        )
    return value


def require_feed_forward(model: dict, source: str = "checkpoint") -> dict:
    """Refuse a checkpoint that declares recurrence.

    For consumers that build the feed-forward stepper only. Dropping the
    block silently would publish feed-forward numbers under a recurrent
    model's name, which is exactly what this contract rules out.
    """
    if RECURRENCE_KEY in model:
        raise ParseError(
            f"{source}: checkpoint declares a {RECURRENCE_KEY!r} block "
            f"({RECURRENCE_SCHEMA}); this consumer steps feed-forward only and "
            "would silently ignore it. Use tools/replay_frozen.py, which "
            "applies the block."
        )
    return model


def recurrence_from_model(
    model: dict, source: str = "checkpoint"
) -> Recurrence | None:
    """Parse and validate the checkpoint's recurrence block.

    ``None`` when the key is absent (a feed-forward checkpoint). A present
    key must satisfy the whole contract -- shape, schema, delay, Dale,
    zero diagonal, Q8.8 grid -- or the checkpoint is unconsumable.
    """
    if RECURRENCE_KEY not in model:
        return None
    block = _require_block_shape(model[RECURRENCE_KEY], source)
    flags = dale_flags(model, source)
    rows = _validated_rows(block, flags, source)
    return Recurrence(
        topology=block["topology"],
        weights=tuple(tuple(row) for row in rows),
    )


def lateral_inhibition(
    model: dict, gain: float, source: str = "checkpoint"
) -> Recurrence:
    """The exp-026 topology: I->E lateral inhibition, nothing else.

    ``R[e][i] = -gain`` for every excitatory ``e`` and inhibitory ``i`` per
    the checkpoint's ``inhibitory`` flags; E->E, E->I, I->I and the
    diagonal are 0. ``gain`` must be positive and on the Q8.8 grid.
    """
    value = _on_q88_grid(gain, "recurrence gain")
    if value <= 0.0:
        raise ParseError(f"recurrence gain must be positive, got {gain!r}")
    flags = _require_mixed_population(dale_flags(model, source))
    return Recurrence(
        topology=(
            f"I->E lateral inhibition, gain {value!r}: R[e][i] = {-value!r} "
            "for every excitatory e and inhibitory i (checkpoint "
            "'inhibitory' flags); E->E, E->I, I->I and self connections 0"
        ),
        weights=tuple(tuple(row) for row in _lateral_rows(flags, value)),
    )


def _require_mixed_population(flags: list[bool]) -> list[bool]:
    if not any(flags) or all(flags):
        raise ParseError(
            "lateral inhibition needs at least one excitatory and one "
            f"inhibitory neuron; inhibitory flags are {flags}"
        )
    return flags


def _lateral_rows(flags: list[bool], value: float) -> list[list[float]]:
    """``R[post][pre] = -value`` for inhibitory ``pre`` onto excitatory ``post``."""
    return [
        [
            f32(-value) if (flags[pre] and not flags[post]) else 0.0
            for pre in range(N_NEURONS)
        ]
        for post in range(N_NEURONS)
    ]


def recurrence_block(recurrence: Recurrence) -> dict:
    """JSON-ready ``recurrence`` block for a checkpoint."""
    return {
        "schema": recurrence.schema,
        "delay_ticks": recurrence.delay_ticks,
        "topology": recurrence.topology,
        "weights": [list(row) for row in recurrence.weights],
    }


def recurrent_mem_words(recurrence: Recurrence) -> list[str]:
    """256 signed Q8.8 hex words, post-major (``post * 16 + pre``)."""
    return [encode_q88_hex(w) for row in recurrence.weights for w in row]


def describe_recurrence(recurrence: Recurrence | None) -> dict | None:
    """Manifest-ready description; ``None`` records a feed-forward replay."""
    if recurrence is None:
        return None
    return {
        "schema": recurrence.schema,
        "delay_ticks": recurrence.delay_ticks,
        "topology": recurrence.topology,
        "nonzero_weights": recurrence.nonzero(),
        "state": STATE_SEMANTICS,
        "order": ORDER_SEMANTICS,
        "export": RECURRENT_MEM_NAME + " (256 words, post-major)",
    }
