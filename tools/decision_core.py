"""Deterministic output-row -> decision contract (Linear RM-1328 / GH #6).

Pure: one finite score row in, a typed decision or a fail-closed error out.
No membrane, no weights, no host actuation. The software replay path is
``replay_output_row`` (shipped Distill vocabulary). FPGA action parity
compares against this, not against an ad-hoc argmax.

Shipped head (exp-025 / Distill ``a1fa491``) is 3-wide, neuron-major,
trained as regression onto ``sample_readout_target``:

    index 0  comfort
    index 1  temp
    index 2  power

That order is the checkpoint's intended ordering. RM-1150 Stage-1
``ALLOW/WARN/THROTTLE/PAUSE/YIELD_GPU`` is five-wide and unbound -- mapping
those names onto these three channels would be a guess.

Standard library only. ASCII hyphens only (cp1252-safe).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

OUTPUT_WIDTH = 3
NEURON_COUNT = 16
OUTPUT_WEIGHT_COUNT = NEURON_COUNT * OUTPUT_WIDTH
SHIPPED_VOCABULARY: tuple[str, ...] = ("comfort", "temp", "power")
SUPERVISOR_VOCABULARY: tuple[str, ...] = (
    "ALLOW",
    "WARN",
    "THROTTLE",
    "PAUSE",
    "YIELD_GPU",
)
CONTRACT_ID = "spikenaut-output-row-v1"
TIE_BREAK = "lowest-index"
CONFIDENCE_FORMULA = "margin / (|winner| + |runner_up|)"

KIND_PROPOSE = "propose"
KIND_ABSTAIN = "abstain"
REASON_LOW_CONFIDENCE = "low_confidence"
REASON_TIE = "tie"

ERROR_EMPTY_ROW = "empty_row"
ERROR_WIDTH_MISMATCH = "width_mismatch"
ERROR_NON_FINITE = "non_finite"
ERROR_EMPTY_VOCABULARY = "empty_vocabulary"
ERROR_INVALID_LABEL = "invalid_label"
ERROR_DUPLICATE_LABEL = "duplicate_label"
ERROR_INVALID_CONFIDENCE_FLOOR = "invalid_confidence_floor"
ERROR_DERIVED_NON_FINITE = "derived_non_finite"
ERROR_INVALID_SCORE = "invalid_score"


class DecisionError(ValueError):
    """Fail-closed: the row or config cannot become a Decision."""

    def __init__(self, code: str, message: str, **fields: object) -> None:
        super().__init__(message)
        self.code = code
        self.fields = fields


@dataclass(frozen=True)
class DecisionConfig:
    """Vocabulary and abstention knobs. Construct with ``new`` / ``shipped``."""

    vocabulary: tuple[str, ...]
    confidence_floor: float
    abstain_on_tie: bool

    @classmethod
    def new(
        cls,
        vocabulary: Sequence[str],
        confidence_floor: float = 0.0,
        abstain_on_tie: bool = False,
    ) -> DecisionConfig:
        labels = tuple(vocabulary)
        _validate_config(labels, confidence_floor)
        return cls(labels, float(confidence_floor), abstain_on_tie)

    @classmethod
    def shipped(cls) -> DecisionConfig:
        return cls.new(SHIPPED_VOCABULARY, 0.0, False)

    def width(self) -> int:
        return len(self.vocabulary)


@dataclass(frozen=True)
class Diagnostics:
    winning_index: int
    winning_score: float
    winning_action: str
    runner_up_index: int | None
    runner_up_score: float | None
    margin: float
    confidence: float
    tied: bool
    scores: tuple[float, ...]


@dataclass(frozen=True)
class Decision:
    kind: str
    abstain_reason: str | None
    diagnostics: Diagnostics


def decide(row: Sequence[float], config: DecisionConfig) -> Decision:
    """Generic conversion. Replay of the shipped bank uses ``replay_output_row``."""
    _validate_config(config.vocabulary, config.confidence_floor)
    scores = _validate_row(row, config.width())
    winning_index, runner_up_index, tied = _pick_winner(scores)
    winning_score = scores[winning_index]
    runner_up_score = None if runner_up_index is None else scores[runner_up_index]
    margin, confidence = _margin_and_confidence(winning_score, runner_up_score)
    diagnostics = Diagnostics(
        winning_index=winning_index,
        winning_score=winning_score,
        winning_action=config.vocabulary[winning_index],
        runner_up_index=runner_up_index,
        runner_up_score=runner_up_score,
        margin=margin,
        confidence=confidence,
        tied=tied,
        scores=scores,
    )
    if tied and config.abstain_on_tie:
        return Decision(KIND_ABSTAIN, REASON_TIE, diagnostics)
    if confidence < config.confidence_floor:
        return Decision(KIND_ABSTAIN, REASON_LOW_CONFIDENCE, diagnostics)
    return Decision(KIND_PROPOSE, None, diagnostics)


def replay_output_row(row: Sequence[float]) -> Decision:
    """Software replay path for the shipped Distill head."""
    return decide(row, DecisionConfig.shipped())


def replay_tick(
    neuron_major: Sequence[float],
    spikes: Sequence[bool],
    config: DecisionConfig | None = None,
) -> Decision:
    """Score one spike vector through the readout, then decide.

    This is the software replay step that sits downstream of a keep-LIF
    tick: spikes in, Distill-ordered decision out. It still does not
    actuate anything.
    """
    row = score_readout(neuron_major, spikes)
    return decide(row, DecisionConfig.shipped() if config is None else config)


def score_readout(
    neuron_major: Sequence[float], spikes: Sequence[bool]
) -> tuple[float, ...]:
    """Distill ``pred = readout * spikes`` on the neuron-major 16x3 image."""
    if len(neuron_major) == 0:
        raise DecisionError(ERROR_EMPTY_ROW, "output row is empty")
    if len(neuron_major) != OUTPUT_WEIGHT_COUNT:
        raise DecisionError(
            ERROR_WIDTH_MISMATCH,
            f"output row width {len(neuron_major)}, expected {OUTPUT_WEIGHT_COUNT}",
            got=len(neuron_major),
            expected=OUTPUT_WEIGHT_COUNT,
        )
    if len(spikes) != NEURON_COUNT:
        raise DecisionError(
            ERROR_WIDTH_MISMATCH,
            f"output row width {len(spikes)}, expected {NEURON_COUNT}",
            got=len(spikes),
            expected=NEURON_COUNT,
        )
    weights = _coerce_real_numbers(neuron_major)
    _refuse_non_finite(weights)
    scores = [0.0] * OUTPUT_WIDTH
    for neuron, spiked in enumerate(spikes):
        if not spiked:
            continue
        base = neuron * OUTPUT_WIDTH
        for channel in range(OUTPUT_WIDTH):
            scores[channel] += weights[base + channel]
    return tuple(scores)


def decision_as_dict(decision: Decision) -> dict:
    """JSON-ready form of a successful decision (no NaN)."""
    diag = decision.diagnostics
    return {
        "kind": decision.kind,
        "abstain_reason": decision.abstain_reason,
        "winning_index": diag.winning_index,
        "winning_score": diag.winning_score,
        "winning_action": diag.winning_action,
        "runner_up_index": diag.runner_up_index,
        "runner_up_score": diag.runner_up_score,
        "margin": diag.margin,
        "confidence": diag.confidence,
        "tied": diag.tied,
        "scores": list(diag.scores),
    }


def error_as_dict(exc: DecisionError) -> dict:
    payload = {"error": exc.code, "message": str(exc)}
    payload.update(exc.fields)
    return payload


def _validate_config(vocabulary: Sequence[str], confidence_floor: float) -> None:
    if len(vocabulary) == 0:
        raise DecisionError(ERROR_EMPTY_VOCABULARY, "decision vocabulary is empty")
    seen: list[str] = []
    for index, label in enumerate(vocabulary):
        if not isinstance(label, str):
            raise DecisionError(
                ERROR_INVALID_LABEL,
                f"decision vocabulary label {index} is not a string",
                index=index,
            )
        if label == "":
            raise DecisionError(
                ERROR_INVALID_LABEL,
                f"decision vocabulary label {index} is empty",
                index=index,
            )
        if label in seen:
            raise DecisionError(
                ERROR_DUPLICATE_LABEL,
                f"decision vocabulary repeats {label!r}",
                label=label,
            )
        seen.append(label)
    _require_confidence_floor(confidence_floor)


def _require_confidence_floor(value: object) -> float:
    """Refuse non-reals, booleans, and values outside finite ``[0, 1]``.

    ``bool`` subclasses ``int``, so ``True`` would otherwise become floor
    ``1.0``. A string or ``None`` makes ``math.isfinite`` raise
    ``TypeError``, and ``10**1000`` raises ``OverflowError``. Rust's API
    is ``f64``, so those must be ``invalid_confidence_floor``.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DecisionError(
            ERROR_INVALID_CONFIDENCE_FLOOR,
            f"confidence floor {value!r} is not a finite value in [0, 1]",
            value=value,
        )
    try:
        number = float(value)
    except (OverflowError, ValueError) as exc:
        raise DecisionError(
            ERROR_INVALID_CONFIDENCE_FLOOR,
            "confidence floor is not representable as a finite float",
            value=value,
        ) from exc
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise DecisionError(
            ERROR_INVALID_CONFIDENCE_FLOOR,
            f"confidence floor {number} is not a finite value in [0, 1]",
            value=number,
        )
    return number


def _validate_row(row: Sequence[float], expected: int) -> tuple[float, ...]:
    if len(row) == 0:
        raise DecisionError(ERROR_EMPTY_ROW, "output row is empty")
    if len(row) != expected:
        raise DecisionError(
            ERROR_WIDTH_MISMATCH,
            f"output row width {len(row)}, expected {expected}",
            got=len(row),
            expected=expected,
        )
    scores = _coerce_real_numbers(row)
    _refuse_non_finite(scores)
    return scores


def _coerce_real_numbers(values: Sequence[object]) -> tuple[float, ...]:
    """Refuse booleans, non-reals, and values that cannot become finite f64.

    ``bool`` subclasses ``int``, so ``math.isfinite(True)`` is true and
    ``float(True)`` is ``1.0``. ``replay_output_row([True, False, False])``
    would otherwise propose ``comfort`` with confidence 1. A Python ``int``
    larger than the finite ``float`` range raises ``OverflowError`` from
    ``float()`` / ``math.isfinite``. Rust only accepts ``f64``, so both
    leaks would break the fail-closed cross-language input contract.
    """
    converted: list[float] = []
    for index, value in enumerate(values):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise DecisionError(
                ERROR_INVALID_SCORE,
                f"output score at index {index} is not a real number",
                index=index,
            )
        try:
            number = float(value)
        except (OverflowError, ValueError) as exc:
            raise DecisionError(
                ERROR_INVALID_SCORE,
                f"output score at index {index} is not representable "
                "as a finite float",
                index=index,
            ) from exc
        converted.append(number)
    return tuple(converted)


def _refuse_non_finite(values: Sequence[float]) -> None:
    indices = [index for index, value in enumerate(values) if not math.isfinite(value)]
    if not indices:
        return
    noun = "score" if len(indices) == 1 else "scores"
    idx = "index" if len(indices) == 1 else "indexes"
    listed = ", ".join(str(index) for index in indices)
    raise DecisionError(
        ERROR_NON_FINITE,
        f"non-finite output {noun} at {idx} {listed}",
        indices=indices,
    )


def _pick_winner(scores: Sequence[float]) -> tuple[int, int | None, bool]:
    winning_index = 0
    winning_score = scores[0]
    tied = False
    for index in range(1, len(scores)):
        score = scores[index]
        if score > winning_score:
            winning_score = score
            winning_index = index
            tied = False
        elif score == winning_score:
            tied = True
    return winning_index, _runner_up(scores, winning_index), tied


def _runner_up(scores: Sequence[float], winning_index: int) -> int | None:
    if len(scores) == 1:
        return None
    best: int | None = None
    for index, score in enumerate(scores):
        if index == winning_index:
            continue
        if best is None or score > scores[best]:
            best = index
    return best


def _derived_non_finite_message(margin_bad: bool, confidence_bad: bool) -> str:
    if margin_bad and confidence_bad:
        return "derived margin and confidence are not finite"
    if margin_bad:
        return "derived margin is not finite"
    if confidence_bad:
        return "derived confidence is not finite"
    return "derived diagnostics are not finite"


def _margin_and_confidence(
    winning_score: float, runner_up_score: float | None
) -> tuple[float, float]:
    """Margin and confidence, or fail closed if either overflows.

    Finite scores can still overflow. ``[f64.MAX, -f64.MAX, -f64.MAX]``
    yields Inf margin and NaN confidence. ``[f64.MAX, f64.MAX / 2, 0]``
    keeps a finite margin while ``|winner| + |runner|`` overflows to Inf,
    so the naive ratio becomes 0.0 and a positive floor would abstain.
    ``NaN < floor`` is false, so either leak would propose or abstain with
    the wrong diagnostics. Refuse a non-finite margin, denominator, or
    confidence instead of substituting a number.
    """
    if runner_up_score is None:
        return 0.0, 1.0
    margin = winning_score - runner_up_score
    denom = abs(winning_score) + abs(runner_up_score)
    denom_bad = not math.isfinite(denom)
    # Match Rust `denom > 0.0` (src/decision.rs). After the finite check,
    # exact IEEE zero is the only remaining zero case; a literal `== 0.0`
    # compare is the same test but trips python:S1244.
    if denom_bad:
        confidence = float("nan")
    elif denom > 0.0:
        confidence = margin / denom
    else:
        confidence = 0.0
    margin_bad = not math.isfinite(margin)
    confidence_bad = denom_bad or not math.isfinite(confidence)
    if margin_bad or confidence_bad:
        raise DecisionError(
            ERROR_DERIVED_NON_FINITE,
            _derived_non_finite_message(margin_bad, confidence_bad),
            margin=margin_bad,
            confidence=confidence_bad,
        )
    return margin, confidence
