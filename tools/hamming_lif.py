"""keep-LIF stepper and Distill mixed E/I K-WTA.

The discrete step matches SynapticDistill ``scripts/spikenaut_train.jl``
``tick!(learn=false)``. Public entry remains ``hamming_core.keep_lif_step``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

try:
    from .hamming_const import (
        E_WTA_MIN,
        I_WTA_MAX,
        INHIB_ROWS,
        N_EXC,
        N_INPUTS,
        N_NEURONS,
        ParseError,
        f32,
    )
except ImportError:
    from hamming_const import (
        E_WTA_MIN,
        I_WTA_MAX,
        INHIB_ROWS,
        N_EXC,
        N_INPUTS,
        N_NEURONS,
        ParseError,
        f32,
    )


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


def _kwta_take_counts(
    n_e: int,
    n_i: int,
    k: int,
    i_wta_max: int,
    e_wta_min: int,
) -> tuple[int, int]:
    """Distill mixed E/I quota: how many excitatory / inhibitory firers to keep."""
    take_i = min(i_wta_max, n_i, k)
    take_e = min(n_e, max(e_wta_min, k - take_i), k)
    take_e = min(take_e, k - take_i)
    leftover = k - take_e - take_i
    extra_e = min(leftover, max(0, n_e - take_e))
    take_e += extra_e
    leftover -= extra_e
    take_i += min(leftover, max(0, min(i_wta_max, n_i) - take_i))
    return take_e, take_i


def _firers(spikes: list[bool]) -> list[int]:
    return [i for i, fired in enumerate(spikes) if fired]


def _split_ei(cand: list[int], n_exc: int) -> tuple[list[int], list[int]]:
    return [i for i in cand if i < n_exc], [i for i in cand if i >= n_exc]


def _rank_by_voltage(indices: list[int], voltage: list[float]) -> list[int]:
    ranked = list(indices)
    ranked.sort(key=lambda i: voltage[i], reverse=True)
    return ranked


def _mask_kept(
    width: int,
    e_c: list[int],
    i_c: list[int],
    take_e: int,
    take_i: int,
) -> list[bool]:
    kept = [False] * width
    for i in e_c[:take_e]:
        kept[i] = True
    for i in i_c[:take_i]:
        kept[i] = True
    return kept


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
    cand = _firers(spikes)
    if not cand:
        return list(spikes)
    e_c, i_c = _split_ei(cand, n_exc)
    take_e, take_i = _kwta_take_counts(
        len(e_c), len(i_c), k, i_wta_max, e_wta_min
    )
    return _mask_kept(
        len(spikes),
        _rank_by_voltage(e_c, voltage),
        _rank_by_voltage(i_c, voltage),
        take_e,
        take_i,
    )


def _weighted_input(bank: LifBank, stim: list[float]) -> list[float]:
    """Analog current ``W @ stim`` plus Dale I bias on inhibitory rows."""
    inputs = []
    for n in range(N_NEURONS):
        acc = 0.0
        row = bank.weights[n]
        for c, s in enumerate(stim):
            acc = f32(acc + f32(row[c] * s))
        if n in INHIB_ROWS and bank.i_drive != 0.0:
            acc = f32(acc + f32(bank.i_drive))
        inputs.append(acc)
    return inputs


def keep_lif_step(bank: LifBank, stim: list[float], k: int | None) -> list[bool]:
    """Advance one keep-LIF tick. Used only by this harness.

    ``stim`` is the 16-wide analog encoder output. Poisson pre-spikes exist
    in Distill only for STDP; ``learn=false`` (this measurement) injects
    analog current ``W @ stim``.
    """
    if len(stim) != N_INPUTS:
        raise ParseError(f"stimulus width {len(stim)}, expected {N_INPUTS}")
    inputs = _weighted_input(bank, stim)
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
