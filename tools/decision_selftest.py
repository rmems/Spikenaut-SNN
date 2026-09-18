"""Prove the decision-contract pin comparison can actually fail (RM-1328).

Same role as ``live_stim_selftest``: every check the pin relies on is
mutated here and must be reported. A pin that cannot fail is a pin that
proves nothing about either implementation.

Standard library only.
"""

from __future__ import annotations

import itertools
import json
import tempfile
from collections.abc import Callable
from pathlib import Path

try:
    from .decision_core import (
        ERROR_EMPTY_ROW,
        ERROR_EMPTY_VOCABULARY,
        ERROR_INVALID_ABSTAIN_ON_TIE,
        ERROR_INVALID_CONFIDENCE_FLOOR,
        ERROR_INVALID_LABEL,
        ERROR_INVALID_SCORE,
        ERROR_INVALID_SPIKE,
        ERROR_NON_FINITE,
        KIND_PROPOSE,
        NEURON_COUNT,
        OUTPUT_WEIGHT_COUNT,
        OUTPUT_WIDTH,
        SHIPPED_VOCABULARY,
        DecisionConfig,
        DecisionError,
        decide,
        replay_output_row,
        replay_tick,
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
    from .q88_core import (
        MEM_OUTPUT,
        ParseError,
        SelfTestFailure,
        encode_q88_hex,
        parse_mem,
    )
except ImportError:
    from decision_core import (
        ERROR_EMPTY_ROW,
        ERROR_EMPTY_VOCABULARY,
        ERROR_INVALID_ABSTAIN_ON_TIE,
        ERROR_INVALID_CONFIDENCE_FLOOR,
        ERROR_INVALID_LABEL,
        ERROR_INVALID_SCORE,
        ERROR_INVALID_SPIKE,
        ERROR_NON_FINITE,
        KIND_PROPOSE,
        NEURON_COUNT,
        OUTPUT_WEIGHT_COUNT,
        OUTPUT_WIDTH,
        SHIPPED_VOCABULARY,
        DecisionConfig,
        DecisionError,
        decide,
        replay_output_row,
        replay_tick,
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
    from q88_core import (
        MEM_OUTPUT,
        ParseError,
        SelfTestFailure,
        encode_q88_hex,
        parse_mem,
    )


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


def _require_parse_error(action: Callable[[], object], *, what: str) -> None:
    try:
        action()
    except ParseError:
        return
    except Exception as exc:
        raise SelfTestFailure(
            f"{what} raised {type(exc).__name__}, expected ParseError"
        ) from exc
    raise SelfTestFailure(f"{what} was accepted")


def _self_test_model_parse_error() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        bad = tmp_path / "snn_model.json"
        bad.write_text("{", encoding="utf-8")
        _require_parse_error(
            lambda: load_shipped_model(bad),
            what="malformed model JSON",
        )
        huge = tmp_path / "huge_int.json"
        huge.write_text('{"n_outputs": 1' + "0" * 5000 + "}", encoding="utf-8")
        _require_parse_error(
            lambda: load_shipped_model(huge),
            what="oversized JSON integer",
        )
        _require_parse_error(
            lambda: load_pin(huge),
            what="load_pin oversized JSON integer",
        )
        deep = tmp_path / "deep.json"
        deep.write_text(_json_text_that_raises_recursion_error(), encoding="utf-8")
        _require_parse_error(
            lambda: load_shipped_model(deep),
            what="deeply nested JSON",
        )
        _require_parse_error(
            lambda: load_pin(deep),
            what="load_pin deeply nested JSON",
        )


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


def _self_test_unrepresentable_scores() -> None:
    _expect_invalid_score(
        lambda: replay_output_row([10**1000, 0, 0]),
        what="unrepresentable integer scores",
    )
    huge_weights = [0.0] * OUTPUT_WEIGHT_COUNT
    huge_weights[0] = 10**1000
    _expect_invalid_score(
        lambda: score_readout(huge_weights, [False] * NEURON_COUNT),
        what="unrepresentable readout weights",
    )


def _self_test_vocabulary_types() -> None:
    _expect_code(
        lambda: DecisionConfig.new([None, "temp", "power"]),
        code=ERROR_INVALID_LABEL,
        what="non-string vocabulary label",
    )
    for vocab in (None, 3, "abc"):
        _expect_code(
            lambda vocab=vocab: DecisionConfig.new(vocab),  # type: ignore[arg-type]
            code=ERROR_EMPTY_VOCABULARY,
            what=f"vocabulary {vocab!r}",
        )
        leaked = DecisionConfig(vocab, 0.0, False)  # type: ignore[arg-type]
        _expect_code(
            lambda leaked=leaked: decide([1.0, 0.0, 0.0], leaked),
            code=ERROR_EMPTY_VOCABULARY,
            what=f"dataclass vocabulary {vocab!r}",
        )
    leaked_gen = DecisionConfig(
        iter(SHIPPED_VOCABULARY), 0.0, False
    )  # type: ignore[arg-type]
    try:
        generated = decide([1.0, 0.0, 0.0], leaked_gen)
    except TypeError as exc:
        raise SelfTestFailure(
            "dataclass iterator vocabulary leaked TypeError"
        ) from exc
    if generated.kind != KIND_PROPOSE:
        raise SelfTestFailure("dataclass iterator vocabulary must still propose")
    if generated.diagnostics.winning_action != "comfort":
        raise SelfTestFailure(
            "dataclass iterator vocabulary must keep Distill order"
        )
    leaked_empty = DecisionConfig(iter(()), 0.0, False)  # type: ignore[arg-type]
    _expect_code(
        lambda: decide([1.0, 0.0, 0.0], leaked_empty),
        code=ERROR_EMPTY_VOCABULARY,
        what="dataclass empty iterator vocabulary",
    )


def _self_test_row_containers() -> None:
    for row in (None, 3, "1.0", {0.1, 0.2, 0.9}, iter((1.0, 0.0, 0.0))):
        _expect_code(
            lambda row=row: replay_output_row(row),  # type: ignore[arg-type]
            code=ERROR_EMPTY_ROW,
            what=f"row {row!r}",
        )


def _self_test_confidence_floor_types() -> None:
    for floor in ("0.5", None, True, 10**1000):
        try:
            DecisionConfig.new(SHIPPED_VOCABULARY, floor)
        except DecisionError as exc:
            if exc.code != ERROR_INVALID_CONFIDENCE_FLOOR:
                raise SelfTestFailure(
                    f"confidence floor {floor!r} raised {exc.code}, "
                    f"expected {ERROR_INVALID_CONFIDENCE_FLOOR}"
                ) from exc
            continue
        except Exception as exc:
            raise SelfTestFailure(
                f"confidence floor {floor!r} raised {type(exc).__name__}, "
                "expected DecisionError"
            ) from exc
        raise SelfTestFailure(f"confidence floor {floor!r} was accepted")
    config = DecisionConfig.new(SHIPPED_VOCABULARY, 0)
    # Truthiness rather than ``!= 0.0``: ``0.0`` and ``-0.0`` are the only
    # falsy floats, which is the same IEEE exact-zero contract as the
    # integer ``0`` floor pin, without a python:S1244 literal compare.
    if config.confidence_floor:
        raise SelfTestFailure("integer confidence floor 0 must remain 0.0")


def _expect_code(
    action: Callable[[], object], *, code: str, what: str
) -> None:
    try:
        action()
    except DecisionError as exc:
        if exc.code != code:
            raise SelfTestFailure(
                f"{what} raised {exc.code}, expected {code}"
            ) from exc
        return
    except Exception as exc:
        raise SelfTestFailure(
            f"{what} raised {type(exc).__name__}, expected DecisionError"
        ) from exc
    raise SelfTestFailure(f"{what} was accepted")


def _self_test_abstain_on_tie_types() -> None:
    for flag in ("false", None, 1, 0):
        _expect_code(
            lambda flag=flag: DecisionConfig.new(SHIPPED_VOCABULARY, 0.0, flag),
            code=ERROR_INVALID_ABSTAIN_ON_TIE,
            what=f"abstain_on_tie {flag!r}",
        )
    proposing = DecisionConfig.new(SHIPPED_VOCABULARY, 0.0, False)
    abstaining = DecisionConfig.new(SHIPPED_VOCABULARY, 0.0, True)
    tied = [0.4, 0.4, 0.4]
    proposed = decide(tied, proposing)
    if proposed.kind != KIND_PROPOSE:
        raise SelfTestFailure("abstain_on_tie False must still propose a tie")
    held = decide(tied, abstaining)
    if held.kind != "abstain":
        raise SelfTestFailure("abstain_on_tie True must abstain on a tie")
    leaked = DecisionConfig(SHIPPED_VOCABULARY, 0.0, "false")  # type: ignore[arg-type]
    _expect_code(
        lambda: decide(tied, leaked),
        code=ERROR_INVALID_ABSTAIN_ON_TIE,
        what="dataclass abstain_on_tie 'false'",
    )


def _self_test_spike_types() -> None:
    weights = [0.0] * OUTPUT_WEIGHT_COUNT
    weights[0] = 1.0
    silent = [False] * NEURON_COUNT
    _expect_code(
        lambda: score_readout(weights, ["false"] * NEURON_COUNT),
        code=ERROR_INVALID_SPIKE,
        what="string spike flags",
    )
    _expect_code(
        lambda: score_readout(weights, [1] * NEURON_COUNT),
        code=ERROR_INVALID_SPIKE,
        what="integer spike flags",
    )
    active = [True] + silent[1:]
    row = score_readout(weights, active)
    if row != (1.0, 0.0, 0.0):
        raise SelfTestFailure(f"boolean True spike must score (1,0,0), got {row}")


def _self_test_overflow_readout() -> None:
    weights = [0.0] * OUTPUT_WEIGHT_COUNT
    weights[0] = 1.7976931348623157e308
    weights[OUTPUT_WIDTH] = 1.7976931348623157e308
    spikes = [True, True] + [False] * (NEURON_COUNT - 2)
    try:
        score_readout(weights, spikes)
    except DecisionError as exc:
        if exc.code != ERROR_NON_FINITE:
            raise SelfTestFailure(
                f"overflowing readout raised {exc.code}, expected {ERROR_NON_FINITE}"
            ) from exc
        return
    except Exception as exc:
        raise SelfTestFailure(
            f"overflowing readout raised {type(exc).__name__}, expected DecisionError"
        ) from exc
    raise SelfTestFailure("overflowing readout returned a finite row")


def _self_test_replay_tick_shipped_only() -> None:
    weights = [0.0] * OUTPUT_WEIGHT_COUNT
    spikes = [False] * NEURON_COUNT
    scored = replay_tick(weights, spikes)
    via_row = replay_output_row(score_readout(weights, spikes))
    if scored != via_row:
        raise SelfTestFailure(
            "replay_tick must equal score_readout plus replay_output_row"
        )
    try:
        replay_tick(weights, spikes, DecisionConfig.shipped())  # type: ignore[call-arg]
    except TypeError:
        return
    raise SelfTestFailure(
        "replay_tick must not accept a config (Rust is shipped-only)"
    )


def _swap_unequal_channels(model: dict) -> bool:
    """Swap two channels whose Q8.8 words differ so the pin can observe drift.

    ``assert_output_json_mem_parity`` compares encoded words, not raw floats.
    A pair that is unequal as floats can still snap to the same Q8.8 word;
    swapping that pair would not fail the pin.
    """
    for neuron in model["neurons"]:
        weights = neuron["output_weights"]
        words = [encode_q88_hex(weight) for weight in weights]
        for left, right in itertools.combinations(range(len(weights)), 2):
            if words[left] != words[right]:
                weights[left], weights[right] = weights[right], weights[left]
                return True
    return False


def _self_test_json_mem_parity() -> None:
    model = load_shipped_model()
    entries = parse_mem(MEM_OUTPUT)
    assert_output_json_mem_parity(model, entries)
    drifted = _clone(model)
    drifted["neurons"][0]["output_weights"][0] = (
        drifted["neurons"][0]["output_weights"][0] + 1.0
    )
    _require_parse_error(
        lambda: assert_output_json_mem_parity(drifted, entries),
        what="JSON vs .mem value drift",
    )
    reordered = _clone(model)
    _require(
        _swap_unequal_channels(reordered),
        "shipped readout must have a pair of unequal Q8.8 channels",
    )
    _require_parse_error(
        lambda: assert_output_json_mem_parity(reordered, entries),
        what="JSON vs .mem order drift",
    )


def run_self_test() -> int:
    try:
        reference = build_pin()
        _self_test_metadata(reference)
        _self_test_cases(reference)
        _self_test_model_parse_error()
        _self_test_boolean_scores()
        _self_test_unrepresentable_scores()
        _self_test_vocabulary_types()
        _self_test_row_containers()
        _self_test_confidence_floor_types()
        _self_test_abstain_on_tie_types()
        _self_test_spike_types()
        _self_test_overflow_readout()
        _self_test_replay_tick_shipped_only()
        _self_test_json_mem_parity()
    except SelfTestFailure as exc:
        print(f"FAIL self-test: {exc}")
        return 1
    print(
        f"OK: decision pin self-test passed ({len(reference['cases'])} cases)."
    )
    return 0
