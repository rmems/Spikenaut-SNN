"""Prove the decision-contract pin comparison can actually fail (RM-1328).

Same role as ``live_stim_selftest``: every check the pin relies on is
mutated here and must be reported. A pin that cannot fail is a pin that
proves nothing about either implementation.

Standard library only.
"""

from __future__ import annotations

import json
import tempfile
from collections.abc import Callable
from pathlib import Path

try:
    from .decision_core import (
        ERROR_INVALID_SCORE,
        KIND_PROPOSE,
        NEURON_COUNT,
        OUTPUT_WEIGHT_COUNT,
        SHIPPED_VOCABULARY,
        DecisionError,
        replay_output_row,
        score_readout,
    )
    from .decision_pin import (
        _PINNED_METADATA,
        assert_output_json_mem_parity,
        build_pin,
        load_pin,
        load_shipped_model,
        pin_failures,
    )
    from .q88_core import MEM_OUTPUT, ParseError, SelfTestFailure, parse_mem
except ImportError:
    from decision_core import (
        ERROR_INVALID_SCORE,
        KIND_PROPOSE,
        NEURON_COUNT,
        OUTPUT_WEIGHT_COUNT,
        SHIPPED_VOCABULARY,
        DecisionError,
        replay_output_row,
        score_readout,
    )
    from decision_pin import (
        _PINNED_METADATA,
        assert_output_json_mem_parity,
        build_pin,
        load_pin,
        load_shipped_model,
        pin_failures,
    )
    from q88_core import MEM_OUTPUT, ParseError, SelfTestFailure, parse_mem


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
    not_object = _clone(reference)
    not_object["cases"][0] = "not-an-object"
    _require(
        bool(pin_failures(reference, not_object)),
        "a non-object case must be reported, not AttributeError",
    )


def _self_test_model_parse_error() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        bad = tmp_path / "snn_model.json"
        bad.write_text("{", encoding="utf-8")
        try:
            load_shipped_model(bad)
        except ParseError:
            pass
        except Exception as exc:
            raise SelfTestFailure(
                f"malformed model JSON raised {type(exc).__name__}, "
                "expected ParseError"
            ) from exc
        else:
            raise SelfTestFailure("malformed model JSON was accepted")

        huge = tmp_path / "huge_int.json"
        huge.write_text('{"n_outputs": 1' + "0" * 5000 + "}", encoding="utf-8")
        try:
            load_shipped_model(huge)
        except ParseError:
            pass
        except Exception as exc:
            raise SelfTestFailure(
                f"oversized JSON integer raised {type(exc).__name__}, "
                "expected ParseError"
            ) from exc
        else:
            raise SelfTestFailure("oversized JSON integer was accepted")

        try:
            load_pin(huge)
        except ParseError:
            pass
        except Exception as exc:
            raise SelfTestFailure(
                f"load_pin oversized JSON integer raised {type(exc).__name__}, "
                "expected ParseError"
            ) from exc
        else:
            raise SelfTestFailure("load_pin accepted an oversized JSON integer")

        deep = tmp_path / "deep.json"
        deep.write_text(_json_text_that_raises_recursion_error(), encoding="utf-8")
        try:
            load_shipped_model(deep)
        except ParseError:
            pass
        except Exception as exc:
            raise SelfTestFailure(
                f"deeply nested JSON raised {type(exc).__name__}, "
                "expected ParseError"
            ) from exc
        else:
            raise SelfTestFailure("deeply nested JSON was accepted")
        try:
            load_pin(deep)
        except ParseError:
            return
        except Exception as exc:
            raise SelfTestFailure(
                f"load_pin deeply nested JSON raised {type(exc).__name__}, "
                "expected ParseError"
            ) from exc
        raise SelfTestFailure("load_pin accepted deeply nested JSON")


def _json_text_that_raises_recursion_error() -> str:
    """A JSON document ``json.loads`` refuses with RecursionError.

    CPython's decoder limit is not ``sys.getrecursionlimit()``. On 3.12 the
    C scanner accepts about 10k nested arrays; on 3.11 the documented case
    is about 1k. Search rather than hard-coding a depth that would parse
    (and then fail only as "expected an object").
    """
    depth = 512
    last = 0
    while depth <= 1_000_000:
        text = "[" * depth + "0" + "]" * depth
        try:
            json.loads(text)
        except RecursionError:
            return text
        last = depth
        depth *= 2
    raise SelfTestFailure(
        f"json.loads accepted nested arrays through depth {last}; "
        "could not construct a RecursionError payload"
    )


def _expect_invalid_score(action: Callable[[], object], *, what: str) -> None:
    try:
        action()
    except DecisionError as exc:
        if exc.code != ERROR_INVALID_SCORE:
            raise SelfTestFailure(
                f"{what} raised {exc.code}, expected {ERROR_INVALID_SCORE}"
            ) from exc
        return
    except Exception as exc:
        raise SelfTestFailure(
            f"{what} raised {type(exc).__name__}, expected DecisionError"
        ) from exc
    raise SelfTestFailure(f"{what} were proposed")


def _self_test_boolean_scores() -> None:
    _expect_invalid_score(
        lambda: replay_output_row([True, False, False]),
        what="boolean scores",
    )
    _expect_invalid_score(
        lambda: replay_output_row([0.9, True, 0.1]),
        what="mixed boolean scores",
    )
    weights = [0.0] * OUTPUT_WEIGHT_COUNT
    weights[0] = True
    _expect_invalid_score(
        lambda: score_readout(weights, [False] * NEURON_COUNT),
        what="boolean readout weights",
    )
    decision = replay_output_row([1, 0, 0])
    if (
        decision.kind != KIND_PROPOSE
        or decision.diagnostics.winning_action != "comfort"
    ):
        raise SelfTestFailure(
            "integer scores 1/0/0 must still propose comfort"
        )


def _self_test_json_mem_parity() -> None:
    model = load_shipped_model()
    entries = parse_mem(MEM_OUTPUT)
    assert_output_json_mem_parity(model, entries)
    drifted = _clone(model)
    drifted["neurons"][0]["output_weights"][0] = (
        drifted["neurons"][0]["output_weights"][0] + 1.0
    )
    try:
        assert_output_json_mem_parity(drifted, entries)
    except ParseError:
        pass
    else:
        raise SelfTestFailure("JSON vs .mem value drift was accepted")
    reordered = _clone(model)
    swapped = False
    for neuron in reordered["neurons"]:
        weights = neuron["output_weights"]
        for left in range(len(weights)):
            for right in range(left + 1, len(weights)):
                if weights[left] != weights[right]:
                    weights[left], weights[right] = weights[right], weights[left]
                    swapped = True
                    break
            if swapped:
                break
        if swapped:
            break
    _require(swapped, "shipped readout must have a pair of unequal channels")
    try:
        assert_output_json_mem_parity(reordered, entries)
    except ParseError:
        return
    raise SelfTestFailure("JSON vs .mem order drift was accepted")


def run_self_test() -> int:
    reference = build_pin()
    try:
        _self_test_metadata(reference)
        _self_test_cases(reference)
        _self_test_model_parse_error()
        _self_test_boolean_scores()
        _self_test_json_mem_parity()
    except SelfTestFailure as exc:
        print(f"FAIL self-test: {exc}")
        return 1
    print(
        f"OK: decision pin self-test passed ({len(reference['cases'])} cases)."
    )
    return 0
