"""Prove the decision-contract pin comparison can actually fail (RM-1328).

Same role as ``live_stim_selftest``: every check the pin relies on is
mutated here and must be reported. A pin that cannot fail is a pin that
proves nothing about either implementation.

Standard library only.
"""

from __future__ import annotations

import json

try:
    from .decision_core import SHIPPED_VOCABULARY
    from .decision_pin import _PINNED_METADATA, build_pin, pin_failures
    from .q88_core import SelfTestFailure
except ImportError:
    from decision_core import SHIPPED_VOCABULARY
    from decision_pin import _PINNED_METADATA, build_pin, pin_failures
    from q88_core import SelfTestFailure


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SelfTestFailure(message)


def _clone(payload: dict) -> dict:
    return json.loads(json.dumps(payload))


def _self_test_metadata(reference: dict) -> None:
    mutations = {
        "contract": "spikenaut-output-row-v0",
        "vocabulary": list(reversed(SHIPPED_VOCABULARY)),
        "tie_break": "highest-index",
        "confidence_formula": "softmax",
        "output_width": 5,
        "supervisor_vocabulary_unbound": ["ALLOW"],
        "n_outputs_json": 5,
        "output_weight_count": 0,
    }
    _require(
        set(mutations) == set(_PINNED_METADATA),
        "the self-test must exercise every field pin_failures compares",
    )
    for key, value in mutations.items():
        mutated = _clone(reference)
        mutated[key] = value
        _require(
            bool(pin_failures(reference, mutated)),
            f"a changed {key!r} must be reported",
        )
    editorial = _clone(reference)
    editorial["note"] = "reworded"
    _require(
        not pin_failures(reference, editorial),
        "`note` is editorial and must not be compared",
    )


def _self_test_cases(reference: dict) -> None:
    drifted = _clone(reference)
    drifted["cases"][0]["winning_index"] = 2
    _require(
        bool(pin_failures(reference, drifted)),
        "a drifted winning_index must be reported",
    )
    truncated = _clone(reference)
    truncated["cases"] = truncated["cases"][:-1]
    _require(
        bool(pin_failures(reference, truncated)),
        "a truncated case list must be reported",
    )
    renamed = _clone(reference)
    renamed["cases"][0]["name"] = "not-the-case"
    _require(
        bool(pin_failures(reference, renamed)),
        "a renamed case must be reported",
    )
    flipped = _clone(reference)
    error_at = next(
        index for index, case in enumerate(reference["cases"]) if not case["ok"]
    )
    flipped["cases"][error_at]["ok"] = True
    flipped["cases"][error_at]["kind"] = "propose"
    _require(
        bool(pin_failures(reference, flipped)),
        "a fail-closed case rewritten as a proposal must be reported",
    )


def run_self_test() -> int:
    reference = build_pin()
    try:
        _self_test_metadata(reference)
        _self_test_cases(reference)
    except SelfTestFailure as exc:
        print(f"FAIL self-test: {exc}")
        return 1
    print(
        f"OK: decision pin self-test passed ({len(reference['cases'])} cases)."
    )
    return 0
