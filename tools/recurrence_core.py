# SPDX-License-Identifier: MIT OR Apache-2.0
"""exp-026 comparison core (GH #3): step arms, fit readouts, score.

Everything that decides a number in the recurrence comparison lives here;
``recurrence_eval`` is argument handling, orchestration and output. The
protocol it implements:

* **Arms.** ``ff`` is the frozen exp-025 feed-forward bank on the Q8.8
  grid. ``rec-g<gain>`` is the same bank plus ``recurrent_lif``'s I->E
  lateral inhibition at that gain. Every arm steps the same encoded rows
  with the same K-WTA and I-drive, resetting at every ``episode_id``.
* **Target.** Distill's ``sample_readout_target``: ``(comfort, temp,
  power)`` where ``temp`` and ``power`` are the frozen-minmax axons and
  ``comfort = clamp(1 - 0.5*temp - 0.5*power, 0, 1)``, all binary32. The
  target is a function of the current input, so a memoryless linear read
  of the inputs is the ceiling and is reported as a baseline.
* **Readouts.** The shipped 16x3 head is scored as-is (``ff-shipped``).
  Every arm, ``ff`` included, also gets a readout refit on the train
  split: ridge regression on the emitted spike vectors with the exp-025
  outgoing-Dale sign constraint (excitatory columns >= 0, inhibitory
  <= 0), solved exactly by Lawson-Hanson NNLS on the normal equations,
  then snapped to the Q8.8 grid. The ridge strength is chosen on the
  validation split only. Refitting ``ff`` too is what makes the
  recurrence effect separable from the effect of a fresh fit.
* **Metrics.** Per head, mean absolute error per session, averaged with
  equal session weight; the primary score is the mean of the three
  session-weighted MAEs. Pooled MAE/RMSE are kept alongside. Activity
  follows Distill ``health_eval``: spikes per tick, all-16 fraction,
  mean pairwise cofire, distinct patterns, inhibitory spikes, silent
  neurons.
* **Verdict.** Predeclared before the first real run: the validation-
  selected recurrent arm is *promising* only if its test primary is at
  least 5% below the refit feed-forward arm's and no head is more than
  5% worse. Anything else is reported as a negative result.

Standard library only; every float path is the same binary32 stepper the
frozen replay uses.
"""

from __future__ import annotations

import math
import time
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass

try:  # package import: `python3 -m tools.recurrence_eval`
    from .decision_core import SHIPPED_VOCABULARY, replay_output_row, score_readout
    from .hamming_const import LIVE_COLUMNS, N_EXC, N_NEURONS, f32
    from .hamming_encode import Sample
    from .hamming_lif import LifBank, keep_lif_step
    from .q88_core import ParseError, Q88RangeError, decode_q88, encode_q88
    from .recurrent_lif import Recurrence, RecurrentState, recurrent_lif_step
except ImportError:  # direct script: `python3 tools/recurrence_eval.py`
    from decision_core import SHIPPED_VOCABULARY, replay_output_row, score_readout
    from hamming_const import LIVE_COLUMNS, N_EXC, N_NEURONS, f32
    from hamming_encode import Sample
    from hamming_lif import LifBank, keep_lif_step
    from q88_core import ParseError, Q88RangeError, decode_q88, encode_q88
    from recurrent_lif import Recurrence, RecurrentState, recurrent_lif_step

HEADS: tuple[str, ...] = SHIPPED_VOCABULARY
N_HEADS = len(HEADS)
TEMP_AXON = LIVE_COLUMNS.index("gpu_temp_c")
POWER_AXON = LIVE_COLUMNS.index("power_w")
N_LIVE = len(LIVE_COLUMNS)
SPLITS: tuple[str, ...] = ("train", "val", "test")

# Gains on the Q8.8 grid. 1/2 and above silenced every excitatory neuron
# in the design prototype; the grid brackets that collapse deliberately.
GAINS_DEFAULT: tuple[float, ...] = (0.0625, 0.125, 0.25, 0.375, 0.5)
# Ridge strength relative to the number of train ticks (lambda = rel * n).
LAMBDA_GRID: tuple[float, ...] = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)
PROMISING_MIN_IMPROVEMENT = 0.05
PROMISING_MAX_DEGRADATION = 0.05
READOUT_FIT_METHOD = (
    "ridge on emitted spike vectors, no bias, outgoing-Dale sign box "
    "(E >= 0, I <= 0) solved by Lawson-Hanson NNLS on the normal "
    "equations, weights snapped to the Q8.8 grid; lambda = lambda_rel * "
    "n_train_ticks chosen on the validation split"
)
PRIMARY_DEFINITION = (
    "mean over the three heads of the session-weighted MAE (equal weight "
    "per session) in target units [0, 1]; lower is better"
)

Target = tuple[float, float, float]


def readout_target(stim: Sequence[float]) -> Target:
    """Distill ``sample_readout_target`` on one encoded stim (binary32)."""
    temp_u = f32(stim[TEMP_AXON])
    power_u = f32(stim[POWER_AXON])
    comfort = f32(f32(1.0 - f32(0.5 * temp_u)) - f32(0.5 * power_u))
    comfort = f32(min(1.0, max(0.0, comfort)))
    return (comfort, temp_u, power_u)


def targets_for(samples: Sequence[Sample]) -> list[Target]:
    return [readout_target(sample.stim) for sample in samples]


@dataclass(frozen=True)
class Span:
    """One session's half-open tick range inside a split."""

    session: str
    start: int
    stop: int


def session_spans(samples: Sequence[Sample]) -> list[Span]:
    spans: list[Span] = []
    start = 0
    for i in range(1, len(samples) + 1):
        if i == len(samples) or samples[i].episode_id != samples[start].episode_id:
            spans.append(Span(samples[start].episode_id, start, i))
            start = i
    return spans


def require_split(samples: Sequence[Sample], split: str) -> None:
    if not samples:
        raise ParseError(
            f"NOTHING WAS MEASURED: 0 ticks in split {split!r}; the "
            "comparison needs train, val and test sessions"
        )


# ---------------------------------------------------------------------------
# Stepping
# ---------------------------------------------------------------------------


@dataclass
class ArmTrace:
    """Emitted spike masks for one arm on one split, plus wall-clock."""

    masks: list[int]
    seconds: float  # timing only; never written into a deterministic artifact


def step_arm(
    bank: LifBank,
    samples: Sequence[Sample],
    k: int | None,
    recurrence: Recurrence | None,
) -> ArmTrace:
    """Step ``samples`` in file order with a per-session reset.

    Same loop as ``replay_core._run_steps``: the feed-forward arm emits
    exactly the ``spikes`` the frozen replay trace records.
    """
    bank = bank.copy_params()
    bank.reset()
    state = RecurrentState() if recurrence is not None else None
    masks: list[int] = []
    prev: str | None = None
    started = time.perf_counter()
    for sample in samples:
        if sample.episode_id != prev:
            bank.reset()
            if state is not None:
                state.reset()
            prev = sample.episode_id
        stim = list(sample.stim)
        if recurrence is None:
            spikes = keep_lif_step(bank, stim, k)
        else:
            spikes = recurrent_lif_step(bank, stim, k, recurrence, state)
        mask = 0
        for i, fired in enumerate(spikes):
            if fired:
                mask |= 1 << i
        masks.append(mask)
    return ArmTrace(masks=masks, seconds=time.perf_counter() - started)


def mask_bits(mask: int) -> list[int]:
    return [i for i in range(N_NEURONS) if (mask >> i) & 1]


def mask_flags(mask: int) -> list[bool]:
    return [bool((mask >> i) & 1) for i in range(N_NEURONS)]


# ---------------------------------------------------------------------------
# Activity (Distill health_eval definitions)
# ---------------------------------------------------------------------------


def _cofire(counts: list[int], both: list[list[int]]) -> float:
    acc = 0.0
    pairs = 0
    for i in range(N_NEURONS - 1):
        if counts[i] == 0:
            continue
        for j in range(i + 1, N_NEURONS):
            if counts[j] == 0:
                continue
            acc += both[i][j] / math.sqrt(float(counts[i]) * float(counts[j]))
            pairs += 1
    return acc / pairs if pairs else 0.0


def _count_pairs(both: list[list[int]], bits: list[int], count: int) -> None:
    for a in range(len(bits)):
        for b in range(a + 1, len(bits)):
            both[bits[a]][bits[b]] += count


def _pattern_totals(patterns: Counter) -> tuple[list[int], list[list[int]], int, int, int]:
    """Per-neuron counts, pairwise co-fire counts, total spikes, all-16
    ticks and inhibitory spikes, accumulated over the distinct patterns."""
    counts = [0] * N_NEURONS
    both = [[0] * N_NEURONS for _ in range(N_NEURONS)]
    total = 0
    all16 = 0
    inhibitory = 0
    for mask, count in sorted(patterns.items()):
        bits = mask_bits(mask)
        total += count * len(bits)
        if len(bits) == N_NEURONS:
            all16 += count
        inhibitory += count * sum(1 for i in bits if i >= N_EXC)
        for i in bits:
            counts[i] += count
        _count_pairs(both, bits, count)
    return counts, both, total, all16, inhibitory


def activity(masks: Sequence[int]) -> dict:
    """Health-style activity summary of one arm on one split."""
    n = len(masks)
    if n == 0:
        raise ParseError("NOTHING WAS MEASURED: activity over 0 ticks")
    patterns = Counter(masks)
    counts, both, total, all16, inhibitory = _pattern_totals(patterns)
    silent = [i for i in range(N_NEURONS) if counts[i] == 0]
    return {
        "n_ticks": n,
        "spikes_per_tick": total / n,
        "all16_fraction": all16 / n,
        "cofire": _cofire(counts, both),
        "patterns": len(patterns),
        "inhibitory_spikes": inhibitory,
        "excitatory_spikes": total - inhibitory,
        "rate_per_neuron": [counts[i] / n for i in range(N_NEURONS)],
        "silent_neurons": silent,
        "active_fraction": (N_NEURONS - len(silent)) / N_NEURONS,
    }


# ---------------------------------------------------------------------------
# Readout fitting: Dale-constrained ridge via NNLS
# ---------------------------------------------------------------------------


def pattern_stats(masks: Sequence[int], targets: Sequence[Target]) -> dict[int, list[float]]:
    """``mask -> [count, sum_head0, sum_head1, sum_head2]`` over the ticks."""
    if len(masks) != len(targets):
        raise ParseError(
            f"{len(masks)} masks but {len(targets)} targets"
        )
    stats: dict[int, list[float]] = {}
    for mask, target in zip(masks, targets, strict=True):
        row = stats.get(mask)
        if row is None:
            row = [0.0] * (1 + N_HEADS)
            stats[mask] = row
        row[0] += 1.0
        for h in range(N_HEADS):
            row[1 + h] += target[h]
    return stats


def normal_equations(stats: dict[int, list[float]]) -> tuple[list[list[float]], list[list[float]]]:
    """Gram ``X^T X`` (16x16) and ``X^T Y`` (16x3) from pattern statistics."""
    gram = [[0.0] * N_NEURONS for _ in range(N_NEURONS)]
    rhs = [[0.0] * N_HEADS for _ in range(N_NEURONS)]
    for mask in sorted(stats):
        count, *sums = stats[mask]
        bits = mask_bits(mask)
        for i in bits:
            for j in bits:
                gram[i][j] += count
            for h in range(N_HEADS):
                rhs[i][h] += sums[h]
    return gram, rhs


def _pivot_row(a: list[list[float]], col: int, n: int) -> int:
    """Row at or below ``col`` with the largest ``|a[r][col]|``.

    Ties go to the lowest index, so the elimination order is deterministic.
    """
    best = col
    for r in range(col + 1, n):
        if abs(a[r][col]) > abs(a[best][col]):
            best = r
    return best


def _solve(matrix: list[list[float]], vector: list[float]) -> list[float]:
    """Gaussian elimination with partial pivoting on a small dense system.

    Zero tests are truthiness, the repo's form of an exact IEEE-zero test
    (``0.0`` and ``-0.0`` are the only falsy floats). A tolerance would be
    the wrong test: the ridge term keeps every real pivot away from zero,
    and only an exactly singular system must be refused.
    """
    n = len(vector)
    a = [row[:] + [vector[i]] for i, row in enumerate(matrix)]
    _eliminate(a, n)
    return _back_substitute(a, n)


def _eliminate(a: list[list[float]], n: int) -> None:
    """Forward elimination in place on the augmented matrix ``a``."""
    for col in range(n):
        pivot = _pivot_row(a, col, n)
        if not a[pivot][col]:
            raise ParseError("readout fit: singular normal equations")
        a[col], a[pivot] = a[pivot], a[col]
        for r in range(col + 1, n):
            factor = a[r][col] / a[col][col]
            if factor:
                for c in range(col, n + 1):
                    a[r][c] -= factor * a[col][c]


def _back_substitute(a: list[list[float]], n: int) -> list[float]:
    # Row ``col`` is never modified after its own elimination step, so each
    # ``a[r][r]`` here is the pivot that passed ``_eliminate``'s zero test.
    x = [0.0] * n
    for r in range(n - 1, -1, -1):
        acc = a[r][n]
        for c in range(r + 1, n):
            acc -= a[r][c] * x[c]
        x[r] = acc / a[r][r]
    return x


def _gradient(
    matrix: list[list[float]], vector: list[float], u: list[float]
) -> list[float]:
    """``c - A u``, the negative gradient of the NNLS objective."""
    n = len(vector)
    return [
        vector[i] - sum(matrix[i][j] * u[j] for j in range(n))
        for i in range(n)
    ]


def _entering_index(grad: list[float], passive: list[bool], tol: float) -> int | None:
    """Largest positive gradient outside the passive set, ties to the
    lowest index; ``None`` when no coordinate can still improve."""
    best: int | None = None
    for j, g in enumerate(grad):
        if not passive[j] and g > tol and (best is None or g > grad[best]):
            best = j
    return best


def _inner_step(
    matrix: list[list[float]],
    vector: list[float],
    passive: list[bool],
    u: list[float],
    tol: float,
) -> tuple[list[float], bool]:
    """One Lawson-Hanson inner iteration on the passive set.

    Solves the passive-set least squares. If that solution is feasible it
    becomes the iterate and the step is done. Otherwise ``u`` moves toward
    it until the first coordinate reaches zero; those coordinates leave the
    passive set and the caller iterates again.
    """
    n = len(u)
    active_set, z = _passive_solution(matrix, vector, passive, n)
    if all(value > tol for value in z):
        out = [0.0] * n
        for i, value in zip(active_set, z, strict=True):
            out[i] = value
        return out, True
    _move_toward(u, active_set, z, passive, tol)
    return u, False


def _passive_solution(
    matrix: list[list[float]], vector: list[float], passive: list[bool], n: int
) -> tuple[list[int], list[float]]:
    """The passive set and the least-squares solution restricted to it."""
    active_set = [i for i in range(n) if passive[i]]
    sub = [[matrix[i][j] for j in active_set] for i in active_set]
    return active_set, _solve(sub, [vector[i] for i in active_set])


def _move_toward(
    u: list[float], active_set: list[int], z: list[float], passive: list[bool], tol: float
) -> None:
    """Move ``u`` toward ``z`` until the first coordinate reaches zero and
    drop every coordinate that did from the passive set (in place)."""
    alpha = min(
        u[i] / (u[i] - value)
        for i, value in zip(active_set, z, strict=True)
        if value <= tol
    )
    for i, value in zip(active_set, z, strict=True):
        u[i] = u[i] + alpha * (value - u[i])
    for i in active_set:
        if u[i] <= tol:
            passive[i] = False
            u[i] = 0.0


def nnls(matrix: list[list[float]], vector: list[float]) -> list[float]:
    """Lawson-Hanson NNLS on the normal equations ``A u = c``, ``u >= 0``.

    Exact active-set solution of ``min 1/2 u^T A u - c^T u`` for SPD ``A``.
    Ties in the entering variable go to the lowest index, so the fit is
    deterministic. Both loops carry an iteration cap that fails closed: the
    inner ``break`` leaves only the inner loop, so the outer cap is live and
    fires if ``cap`` entering steps pass without the gradient test clearing.
    """
    if not vector:
        raise ParseError("readout fit: empty system")
    n = len(vector)
    tol = 1e-12 * max(1.0, max(abs(v) for v in vector))
    passive = [False] * n
    u = [0.0] * n
    cap = 4 * n + 16
    for _ in range(cap):
        entering = _entering_index(_gradient(matrix, vector, u), passive, tol)
        if entering is None:
            return u
        passive[entering] = True
        for _ in range(cap):
            u, feasible = _inner_step(matrix, vector, passive, u, tol)
            if feasible:
                break
        else:
            raise ParseError("readout fit: NNLS inner loop did not converge")
    raise ParseError("readout fit: NNLS outer loop did not converge")


def snap_q88(value: float, where: str) -> float:
    try:
        return f32(decode_q88(encode_q88(value)))
    except Q88RangeError as exc:
        raise ParseError(f"{where}: {exc}") from exc


def fit_dale_readout(
    gram: list[list[float]],
    rhs: list[list[float]],
    lam: float,
    inhibitory: Sequence[bool],
) -> list[list[float]]:
    """Neuron-major 16x3 readout: Dale-constrained ridge, on the Q8.8 grid.

    ``w = D u`` with ``u >= 0`` and ``D = diag(+1 for E, -1 for I)``, so the
    sign box becomes a plain NNLS on ``D (G + lam I) D`` and ``D b``.
    """
    _require_positive_lambda(lam)
    sign = [-1.0 if flag else 1.0 for flag in inhibitory]
    system = _signed_ridge_system(gram, lam, sign)
    weights = [[0.0] * N_HEADS for _ in range(N_NEURONS)]
    for h in range(N_HEADS):
        u = nnls(system, [sign[i] * rhs[i][h] for i in range(N_NEURONS)])
        for i in range(N_NEURONS):
            weights[i][h] = snap_q88(sign[i] * u[i], f"readout[{i}][{h}]")
    return weights


def _require_positive_lambda(lam: float) -> None:
    if lam <= 0.0 or not math.isfinite(lam):
        raise ParseError(f"readout fit: lambda must be positive, got {lam!r}")


def _signed_ridge_system(
    gram: list[list[float]], lam: float, sign: list[float]
) -> list[list[float]]:
    """``D (G + lam I) D`` for the sign vector ``D = diag(sign)``."""
    return [
        [sign[i] * (gram[i][j] + (lam if i == j else 0.0)) * sign[j] for j in range(N_NEURONS)]
        for i in range(N_NEURONS)
    ]


def flatten_readout(weights: list[list[float]]) -> list[float]:
    return [weights[i][h] for i in range(N_NEURONS) for h in range(N_HEADS)]


def sign_census(weights: list[list[float]], inhibitory: Sequence[bool]) -> dict:
    """How the fitted head sits against outgoing Dale; zeros are allowed."""
    e_neg = _count_side(weights, inhibitory, False, lambda w: w < 0.0)
    i_pos = _count_side(weights, inhibitory, True, lambda w: w > 0.0)
    # Truthiness is the exact-zero test (see ``_solve``): a literal float
    # equality is the same test but trips python:S1244.
    zero = sum(1 for row in weights for w in row if not w)
    return {
        "dale_violations": e_neg + i_pos,
        "zero_weights": zero,
        "nonzero_weights": N_NEURONS * N_HEADS - zero,
    }


def _count_side(
    weights: list[list[float]],
    inhibitory: Sequence[bool],
    side: bool,
    violates: Callable[[float], bool],
) -> int:
    """Weights on the ``side`` (inhibitory or not) rows for which ``violates`` holds."""
    return sum(
        1
        for i in range(N_NEURONS)
        if bool(inhibitory[i]) == side
        for w in weights[i]
        if violates(w)
    )


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


class ReadoutScorer:
    """Scores spike masks through one neuron-major 16x3 head, memoized.

    Uses ``decision_core.score_readout`` / ``replay_output_row`` so the
    numbers are the replay's numbers, not a second argmax.
    """

    def __init__(self, neuron_major: Sequence[float]) -> None:
        self.neuron_major = list(neuron_major)
        self._scores: dict[int, tuple[float, ...]] = {}
        self._decisions: dict[int, int] = {}

    def scores(self, mask: int) -> tuple[float, ...]:
        cached = self._scores.get(mask)
        if cached is None:
            cached = score_readout(self.neuron_major, mask_flags(mask))
            self._scores[mask] = cached
        return cached

    def decision(self, mask: int) -> int:
        cached = self._decisions.get(mask)
        if cached is None:
            cached = replay_output_row(self.scores(mask)).diagnostics.winning_index
            self._decisions[mask] = cached
        return cached


def metrics_from_predictions(
    predictions: Sequence[Sequence[float]],
    targets: Sequence[Target],
    spans: Sequence[Span],
) -> dict:
    """Session-weighted MAE per head, the primary score, and pooled errors."""
    if len(predictions) != len(targets):
        raise ParseError(
            f"{len(predictions)} predictions but {len(targets)} targets"
        )
    if not spans:
        raise ParseError("NOTHING WAS MEASURED: no sessions to score")
    per_session: dict[str, dict] = {}
    abs_sum = [0.0] * N_HEADS
    sq_sum = [0.0] * N_HEADS
    for span in spans:
        session_abs = _session_abs_errors(predictions, targets, span, sq_sum)
        n = span.stop - span.start
        per_session[span.session] = {
            "n_ticks": n,
            "mae": [session_abs[h] / n for h in range(N_HEADS)],
        }
        for h in range(N_HEADS):
            abs_sum[h] += session_abs[h]
    return _metrics_summary(per_session, abs_sum, sq_sum, len(targets))


def _session_abs_errors(
    predictions: Sequence[Sequence[float]],
    targets: Sequence[Target],
    span: Span,
    sq_sum: list[float],
) -> list[float]:
    """Sum of |error| per head over one session; squared errors accumulate
    into ``sq_sum`` in the same tick-then-head order."""
    session_abs = [0.0] * N_HEADS
    for t in range(span.start, span.stop):
        for h in range(N_HEADS):
            err = predictions[t][h] - targets[t][h]
            session_abs[h] += abs(err)
            sq_sum[h] += err * err
    return session_abs


def _metrics_summary(
    per_session: dict[str, dict], abs_sum: list[float], sq_sum: list[float], total: int
) -> dict:
    weighted = [
        math.fsum(row["mae"][h] for row in per_session.values()) / len(per_session)
        for h in range(N_HEADS)
    ]
    return {
        "heads": list(HEADS),
        "session_weighted_mae": weighted,
        "primary": math.fsum(weighted) / N_HEADS,
        "pooled_mae": [abs_sum[h] / total for h in range(N_HEADS)],
        "pooled_rmse": [math.sqrt(sq_sum[h] / total) for h in range(N_HEADS)],
        "sessions": per_session,
    }


def readout_predictions(masks: Sequence[int], scorer: ReadoutScorer) -> list[tuple[float, ...]]:
    return [scorer.scores(mask) for mask in masks]


def decision_agreement(
    masks_a: Sequence[int],
    scorer_a: ReadoutScorer,
    masks_b: Sequence[int],
    scorer_b: ReadoutScorer,
) -> float:
    """Fraction of ticks on which two arm/readout pairs pick the same head."""
    if len(masks_a) != len(masks_b):
        raise ParseError("decision agreement needs equal-length traces")
    if not masks_a:
        raise ParseError("NOTHING WAS MEASURED: decision agreement over 0 ticks")
    same = sum(
        1
        for a, b in zip(masks_a, masks_b, strict=True)
        if scorer_a.decision(a) == scorer_b.decision(b)
    )
    return same / len(masks_a)


def target_argmax(target: Sequence[float]) -> int:
    best = 0
    for h in range(1, len(target)):
        if target[h] > target[best]:
            best = h
    return best


def target_argmax_agreement(
    masks: Sequence[int], scorer: ReadoutScorer, targets: Sequence[Target]
) -> float:
    if len(masks) != len(targets):
        raise ParseError(f"{len(masks)} masks but {len(targets)} targets")
    if not masks:
        raise ParseError("NOTHING WAS MEASURED: target agreement over 0 ticks")
    same = sum(
        1
        for mask, target in zip(masks, targets, strict=True)
        if scorer.decision(mask) == target_argmax(target)
    )
    return same / len(masks)


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------


def constant_predictor(train_targets: Sequence[Target]) -> list[float]:
    n = len(train_targets)
    if n == 0:
        raise ParseError("NOTHING WAS MEASURED: constant baseline over 0 train ticks")
    return [math.fsum(t[h] for t in train_targets) / n for h in range(N_HEADS)]


def persistence_predictions(
    targets: Sequence[Target], spans: Sequence[Span], fallback: Sequence[float]
) -> list[Sequence[float]]:
    """Previous tick's target within the session; the first tick falls back."""
    out: list[Sequence[float]] = [tuple(fallback)] * len(targets)
    for span in spans:
        for t in range(span.start + 1, span.stop):
            out[t] = targets[t - 1]
    return out


def input_features(sample: Sample) -> list[float]:
    """The five live axons plus a bias term (float, not quantized)."""
    return [float(v) for v in sample.stim[:N_LIVE]] + [1.0]


def fit_linear_inputs(
    samples: Sequence[Sample], targets: Sequence[Target], lam: float
) -> list[list[float]]:
    """Closed-form ridge on the inputs: ``(X^T X + lam I) W = X^T Y``."""
    width = N_LIVE + 1
    gram = [[0.0] * width for _ in range(width)]
    rhs = [[0.0] * N_HEADS for _ in range(width)]
    for sample, target in zip(samples, targets, strict=True):
        _accumulate_normal(gram, rhs, input_features(sample), target, width)
    for i in range(width - 1):  # the bias term is not penalized
        gram[i][i] += lam
    return [
        _solve(gram, [rhs[i][h] for i in range(width)]) for h in range(N_HEADS)
    ]


def _accumulate_normal(
    gram: list[list[float]],
    rhs: list[list[float]],
    x: list[float],
    target: Target,
    width: int,
) -> None:
    """Add one sample's outer products to ``X^T X`` and ``X^T Y`` in place."""
    for i in range(width):
        for j in range(width):
            gram[i][j] += x[i] * x[j]
        for h in range(N_HEADS):
            rhs[i][h] += x[i] * target[h]


def linear_predictions(
    samples: Sequence[Sample], weights: list[list[float]]
) -> list[tuple[float, ...]]:
    return [
        tuple(
            math.fsum(w * x for w, x in zip(weights[h], input_features(sample), strict=True))
            for h in range(N_HEADS)
        )
        for sample in samples
    ]


# ---------------------------------------------------------------------------
# Selection and verdict
# ---------------------------------------------------------------------------


def select_by_val(candidates: dict, key: Callable[[dict], float]) -> object:
    """Lowest validation primary; ties go to the earliest (smallest) key."""
    if not candidates:
        raise ParseError("nothing to select from")
    return min(sorted(candidates), key=lambda name: key(candidates[name]))


def relative_change(delta: float, reference: float) -> float | None:
    """``delta / reference``, or ``None`` when no finite ratio exists.

    A reference error of exactly zero makes any nonzero delta an unbounded
    change; reporting it as ``0.0`` would let a regression pass the per-head
    limit, so it is ``None`` (JSON ``null``) and never counts as within limit.
    Zero against zero is a genuine ``0.0``.
    """
    if reference > 0.0:
        return delta / reference
    return None if delta else 0.0


def fmt_ratio(value: float | None) -> str:
    """``+9.48%`` style, or a plain word when the ratio is undefined."""
    return "undefined (reference error is 0)" if value is None else f"{value:+.2%}"


def promising(candidate_test: dict, reference_test: dict) -> dict:
    """The predeclared verdict against the refit feed-forward reference."""
    ref = reference_test["primary"]
    cand = candidate_test["primary"]
    improvement = relative_change(ref - cand, ref)
    degradations = [
        relative_change(
            candidate_test["session_weighted_mae"][h] - reference_test["session_weighted_mae"][h],
            reference_test["session_weighted_mae"][h],
        )
        for h in range(N_HEADS)
    ]
    worst = None if any(d is None for d in degradations) else max(degradations)
    verdict = (
        improvement is not None
        and improvement >= PROMISING_MIN_IMPROVEMENT
        and worst is not None
        and worst <= PROMISING_MAX_DEGRADATION
    )
    return {
        "promising": verdict,
        "primary_improvement": improvement,
        "per_head_degradation": degradations,
        "worst_head_degradation": worst,
        "criterion": (
            f">= {PROMISING_MIN_IMPROVEMENT:.0%} lower test primary than the "
            f"refit feed-forward arm and no head > {PROMISING_MAX_DEGRADATION:.0%} "
            "worse; selection on validation only"
        ),
    }
