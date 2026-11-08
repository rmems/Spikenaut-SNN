# SPDX-License-Identifier: MIT OR Apache-2.0
"""Dense linear algebra for the exp-026 readout fit (GH #3).

Two small kernels on plain Python floats, written out so every operation
is visible and deterministic: Gaussian elimination with partial pivoting
(``solve_dense``) and Lawson-Hanson NNLS on the normal equations
(``nnls``). ``recurrence_core`` wraps them into the Dale-constrained ridge
fit. Standard library only; no numpy, so the arithmetic order is fixed
and the committed ``comparison.json`` reproduces bit for bit.
"""

from __future__ import annotations

try:  # package import: `python3 -m tools.recurrence_eval`
    from .q88_core import ParseError
except ImportError:  # direct script: `python3 tools/recurrence_eval.py`
    from q88_core import ParseError


def _pivot_row(a: list[list[float]], col: int, n: int) -> int:
    """Row at or below ``col`` with the largest ``|a[r][col]|``.

    Ties go to the lowest index, so the elimination order is deterministic.
    """
    best = col
    for r in range(col + 1, n):
        if abs(a[r][col]) > abs(a[best][col]):
            best = r
    return best


def solve_dense(matrix: list[list[float]], vector: list[float]) -> list[float]:
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
    return active_set, solve_dense(sub, [vector[i] for i in active_set])


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
