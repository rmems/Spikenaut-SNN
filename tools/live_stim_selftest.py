#!/usr/bin/env python3
"""Prove the stim parity comparison can actually fail (issue #52).

Same role as ``hamming_selftest`` and ``q88_selftest``: every check the pin
relies on is mutated here and must be reported. A pin that cannot fail is a
pin that proves nothing about either implementation.

Standard library only.
"""

from __future__ import annotations

import json
import struct

try:  # package import: `python3 -m tools.live_stim_parity`
    from .hamming_const import LIVE_COLUMNS, N_INPUTS
    from .live_stim_pin import (
        _PINNED_METADATA,
        _f64_bits,
        _shape_problem,
        build_pin,
        pin_failures,
    )
    from .q88_core import SelfTestFailure
except ImportError:  # direct script: `python3 tools/live_stim_parity.py`
    from hamming_const import LIVE_COLUMNS, N_INPUTS
    from live_stim_pin import (
        _PINNED_METADATA,
        _f64_bits,
        _shape_problem,
        build_pin,
        pin_failures,
    )
    from q88_core import SelfTestFailure

def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SelfTestFailure(message)


def _self_test_metadata(reference: dict) -> None:
    """Every pinned protocol field must be compared, not just the first."""
    mutations = {
        "fixture": "tools/fixtures/hamming_method/holdout.jsonl",
        "live_columns": list(reversed(LIVE_COLUMNS)),
        "unused_axons": "4:15",
        "frozen_lineage": "deadbeef",
        "frozen_minmax": {column: [0.0, 1.0] for column in LIVE_COLUMNS},
        "encoder": "some other encoder",
        "stepper": "Poisson spikes",
    }
    _require(
        set(mutations) == set(_PINNED_METADATA),
        "the self-test must exercise every field pin_failures compares",
    )
    for key, value in mutations.items():
        mutated = json.loads(json.dumps(reference))
        mutated[key] = value
        _require(
            bool(pin_failures(reference, mutated)),
            f"a changed {key!r} must be reported",
        )

    # `note` is the documented exception; prove it really is exempt.
    editorial = json.loads(json.dumps(reference))
    editorial["note"] = "reworded"
    _require(
        not pin_failures(reference, editorial),
        "`note` is editorial and must not be compared",
    )


def _self_test_samples(reference: dict) -> None:
    """A drifted, truncated or dropped sample row must be reported."""
    # A genuine one-ulp step, on a row whose value is not zero. Row 3 (index 2)
    # is the mid-load reading; nudging axon 0 by one binary32 ulp is the
    # smallest real drift there is, and the case a loose tolerance would hide.
    drifted = json.loads(json.dumps(reference))
    value = drifted["samples"][2]["stim"][0]
    _require(value > 0.0, "the ulp case needs a non-zero reference value")
    drifted["samples"][2]["stim"][0] = struct.unpack(
        ">f", struct.pack(">I", struct.unpack(">I", struct.pack(">f", value))[0] + 1)
    )[0]
    _require(
        bool(pin_failures(reference, drifted)),
        "a one-ulp stim drift must be reported",
    )

    # A stim array cut short must not compare equal on its surviving prefix.
    truncated = json.loads(json.dumps(reference))
    truncated["samples"][0]["stim"] = truncated["samples"][0]["stim"][:3]
    _require(
        bool(pin_failures(reference, truncated)),
        "a truncated stim array must be reported, not compared on its prefix",
    )

    short = json.loads(json.dumps(reference))
    short["samples"].pop()
    _require(
        bool(pin_failures(reference, short)), "a dropped sample row must be reported"
    )


def _first_case_index(reference: dict, *, refused: bool) -> int:
    """Index of the first bit-exact case with the given verdict."""
    for index, case in enumerate(reference["bit_exact"]):
        if case["refused"] is refused:
            return index
    raise SelfTestFailure(
        f"the pin must carry at least one {'refused' if refused else 'accepted'} "
        "bit-exact case"
    )


def _self_test_bit_exact(reference: dict) -> None:
    """The refusal contract must be pinned, and must be able to fail."""
    refused_at = _first_case_index(reference, refused=True)
    accepted_at = _first_case_index(reference, refused=False)

    # An encoder that started accepting what the reference refuses -- the
    # failure mode with no JSON-literal coverage at all.
    flipped = json.loads(json.dumps(reference))
    flipped["bit_exact"][refused_at]["refused"] = False
    flipped["bit_exact"][refused_at]["stim_bits"] = ["00000000"] * N_INPUTS
    _require(
        bool(pin_failures(reference, flipped)),
        "a refusal turned into an acceptance must be reported",
    )

    # One bit of one accepted vector.
    nudged = json.loads(json.dumps(reference))
    bits = int(nudged["bit_exact"][accepted_at]["stim_bits"][0], 16) ^ 1
    nudged["bit_exact"][accepted_at]["stim_bits"][0] = f"{bits:08x}"
    _require(
        bool(pin_failures(reference, nudged)),
        "a single flipped bit in an accepted vector must be reported",
    )

    # Editing the reading itself, which would otherwise silently repoint a
    # case at an input the label no longer describes.
    repointed = json.loads(json.dumps(reference))
    repointed["bit_exact"][0]["reading_bits"][0] = _f64_bits(1.0)
    _require(
        bool(pin_failures(reference, repointed)),
        "an edited bit_exact reading must be reported",
    )


def _self_test_leaks(reference: dict) -> None:
    """A non-zero unused axon must be refused as a leak, by its own name."""
    leaked = list(reference["samples"][0]["stim"])
    leaked[5] = 1e-30
    problem = _shape_problem(leaked, "self-test")
    _require(problem is not None, "a non-zero axon 5 must be refused as a leak")
    _require(
        "UNUSED-AXON LEAK" in problem,
        f"wrong refusal for an unused-axon leak: {problem}",
    )


def self_test() -> None:
    """Prove the comparison can fail, without touching the shipped pin."""
    reference = build_pin()
    _require(
        not pin_failures(reference, json.loads(json.dumps(reference))),
        "the reference payload must compare equal to itself",
    )
    _self_test_metadata(reference)
    _self_test_samples(reference)
    _self_test_bit_exact(reference)
    _self_test_leaks(reference)
    print("SELF-TEST PASSED: the stim parity pin can fail.")


