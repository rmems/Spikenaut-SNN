"""Self-test for the Hamming harness: prove the checker can actually fail.

A measurement tool that cannot fail is worthless, and so is one that
publishes 0.0% Hamming having read nothing. Each scenario builds a
specific regression and asserts ``hamming_core`` rejects it, or that the
in-repo method fixture still yields its pinned numbers.

Scratch files go to a temp directory; nothing here writes into
``dataset/`` or overwrites the method fixture.

Run via the CLI::

    python3 tools/measure_hamming.py --self-test
"""

from __future__ import annotations

import io
import json
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path

try:  # package import
    from .hamming_imports import (
        CONDITION_EXP024,
        CONDITION_SHIPPED,
        FIXTURE_DIR,
        KResult,
        LifBank,
        Measurement,
        ParseError,
        Protocol,
        SHIPPED_DIR,
        SelfTestFailure,
        apply_kwta,
        encode_q88_hex,
        encode_record,
        keep_lif_step,
        load_expected,
        load_jsonl,
        measure,
        measure_method_fixture,
        method_fixture_paths,
        pin_matches,
        report,
        select_samples,
    )
except ImportError:
    from hamming_imports import (
        CONDITION_EXP024,
        CONDITION_SHIPPED,
        FIXTURE_DIR,
        KResult,
        LifBank,
        Measurement,
        ParseError,
        Protocol,
        SHIPPED_DIR,
        SelfTestFailure,
        apply_kwta,
        encode_q88_hex,
        encode_record,
        keep_lif_step,
        load_expected,
        load_jsonl,
        measure,
        measure_method_fixture,
        method_fixture_paths,
        pin_matches,
        report,
        select_samples,
    )


def _require(condition: bool, message: str) -> None:
    """Assert a self-test invariant. Survives ``python -O``."""
    _require.calls += 1
    if not condition:
        raise SelfTestFailure(message)


_require.calls = 0

SELF_TEST_SECTIONS = 18
EXPECTED_REQUIRE_CALLS = 39
EXPECTED_GUARD_CHECKS = 13


def _v3_row(episode: str, mem_util: float = 75.0, temp: float = 0.0) -> dict:
    return {
        "episode_id": episode,
        "mem_util_pct": mem_util,
        "power_w": 8.527000427246094,
        "gpu_temp_c": temp,
        "sm_clock_mhz": 180.0,
        "mem_clock_mhz": 405.0,
    }


def method_fixture_matches_pin(stream) -> int:
    """1 -- the in-repo fixture still yields the pinned Hamming."""
    print("1. method fixture matches its pin", file=stream)
    measurement = measure_method_fixture()
    expected = load_expected(method_fixture_paths()["expect"])
    failures = pin_matches(measurement, expected)
    _require(not failures, f"method fixture drifted: {failures}")
    _require(measurement.k_none.n_ticks == expected["n_ticks"], "tick count")
    _require(measurement.protocol.condition == "method-fixture", "condition")
    _require("74acdd0f" in measurement.protocol.encoder, "lineage missing")
    _require(measurement.protocol.n_ticks > 0, "empty fixture")
    return 5


def empty_holdout_is_parse_error(tmp: Path, stream) -> int:
    """2 -- 0 ticks is a hard failure, never 0.0% Hamming."""
    print("2. empty holdout is ParseError (never 0.0%)", file=stream)
    empty = tmp / "empty.jsonl"
    empty.write_text("", encoding="utf-8")
    paths = method_fixture_paths()
    raised = False
    try:
        measure(
            float_json=paths["float_json"],
            mem_dir=paths["mem_dir"],
            jsonl=empty,
            split="all",
            condition="method-fixture",
            seed="n/a",
            i_drive=0.0,
            weights_label="self-test empty",
        )
    except ParseError as exc:
        raised = True
        _require("NOTHING WAS MEASURED" in str(exc), f"wrong error: {exc}")
    else:
        raise SelfTestFailure("empty holdout published a measurement")
    _require(raised, "empty holdout did not raise")
    return 3


def malformed_jsonl_is_parse_error(tmp: Path, stream) -> int:
    """3 -- a broken line is exit-2 material, not a traceback."""
    print("3. malformed JSONL is ParseError", file=stream)
    bad = tmp / "bad.jsonl"
    bad.write_text("{not json\n", encoding="utf-8")
    paths = method_fixture_paths()
    try:
        measure(
            float_json=paths["float_json"],
            mem_dir=paths["mem_dir"],
            jsonl=bad,
            split="all",
            condition="method-fixture",
            seed="n/a",
            i_drive=0.0,
            weights_label="self-test bad jsonl",
        )
    except ParseError as exc:
        _require("not a JSON object" in str(exc), f"wrong error: {exc}")
    else:
        raise SelfTestFailure("malformed JSONL did not raise ParseError")
    # An integer literal past sys.get_int_max_str_digits() is a bare
    # ValueError from json.loads, not JSONDecodeError; it used to escape as
    # a traceback (exit 1) instead of ParseError (exit 2).
    huge = tmp / "huge_int.jsonl"
    huge.write_text(
        '{"episode_id": "gpu-000170", "mem_util_pct": 1' + "0" * 5000 + "}\n",
        encoding="utf-8",
    )
    try:
        select_samples(load_jsonl(huge), "all")
    except ParseError as exc:
        _require("huge_int.jsonl:1" in str(exc), f"wrong error: {exc}")
    except ValueError as exc:
        raise SelfTestFailure("oversized JSONL integer leaked ValueError") from exc
    else:
        raise SelfTestFailure("oversized JSONL integer was accepted")
    return 4


def missing_episode_id_is_parse_error(stream) -> int:
    """4 -- a v3 row without gpu-###### is refused, not dropped."""
    print("4. missing episode_id is ParseError", file=stream)
    row = _v3_row("gpu-000000")
    del row["episode_id"]
    try:
        select_samples([row], "all")
    except ParseError as exc:
        _require("episode_id" in str(exc), f"wrong error: {exc}")
    else:
        raise SelfTestFailure("missing episode_id was silently dropped")
    return 2


def interleaved_episodes_refused(stream) -> int:
    """5 -- interleaved episode_id is refused (temporal reset would lie)."""
    print("5. interleaved episodes are ParseError", file=stream)
    rows = [
        _v3_row("gpu-000000"),
        _v3_row("gpu-000001"),
        _v3_row("gpu-000000"),
    ]
    try:
        select_samples(rows, "all")
    except ParseError as exc:
        _require("not contiguous" in str(exc), f"wrong error: {exc}")
    else:
        raise SelfTestFailure("interleaved episodes were accepted")
    return 2


def wrong_pin_is_detected(stream) -> int:
    """6 -- a wrong method pin fails; this is how the checker proves it can fail."""
    print("6. wrong method pin is detected", file=stream)
    measurement = measure_method_fixture()
    bogus = {
        "k_none_pct": -1.0,
        "k_none_bits": -1.0,
        "k_4_pct": -1.0,
        "k_4_bits": -1.0,
        "n_ticks": measurement.k_none.n_ticks,
        "hidden_json_mem_mismatches": measurement.hidden_json_mem_mismatches,
    }
    failures = pin_matches(measurement, bogus)
    _require(len(failures) >= 4, f"wrong pin produced {failures}")
    return 1


def unlabeled_merged_v2_refused_as_exp024(tmp: Path, stream) -> int:
    """7 -- exp-024 must not silently score the shipped ramp."""
    print("7. exp-024 refuses dataset/merged_v2", file=stream)
    jsonl = tmp / "tiny.jsonl"
    jsonl.write_text(
        json.dumps(_v3_row("gpu-000170")) + "\n",
        encoding="utf-8",
    )
    try:
        measure(
            float_json=SHIPPED_DIR / "snn_model.json",
            mem_dir=SHIPPED_DIR,
            jsonl=jsonl,
            split="test",
            condition=CONDITION_EXP024,
            seed="123",
            i_drive=0.05,
            weights_label="must be refused",
        )
    except ParseError as exc:
        _require("merged_v2" in str(exc), f"wrong error: {exc}")
    else:
        raise SelfTestFailure("exp-024 accepted shipped merged_v2")
    return 2


def unused_axons_stay_zero(stream) -> int:
    """8 -- encoder leaves axons 5-15 at 0."""
    print("8. unused axons 5-15 stay 0", file=stream)
    stim = encode_record(_v3_row("gpu-000000", mem_util=75.0, temp=69.0))
    _require(len(stim) == 16, "stim width")
    _require(stim[0] == 1.0, f"mem_util scaled to {stim[0]}, expected 1.0")
    _require(stim[2] == 1.0, f"temp scaled to {stim[2]}, expected 1.0")
    _require(all(c == 0.0 for c in stim[5:]), f"unused leaked: {stim[5:]}")
    return 4


def empty_report_refuses(stream) -> int:
    """9 -- report() refuses a 0-tick KResult (backstop for measure())."""
    print("9. report() refuses a 0-tick measurement", file=stream)
    proto = Protocol(
        condition="method-fixture",
        weights_label="synthetic empty",
        float_json=FIXTURE_DIR / "snn_model.json",
        mem_dir=FIXTURE_DIR,
        encoder="synthetic",
        split="all",
        episodes="(none)",
        seed="n/a",
        n_ticks=0,
        i_drive=0.0,
        stepper="synthetic",
        compared="synthetic",
    )
    empty = KResult(k=None, n_ticks=0, disagree_ticks=0, mean_bits=0.0, pct=0.0)
    fake = Measurement(
        protocol=proto,
        k_none=empty,
        k_4=KResult(k=4, n_ticks=0, disagree_ticks=0, mean_bits=0.0, pct=0.0),
        hidden_json_mem_mismatches=0,
        hidden_compared=256,
    )
    sink = io.StringIO()
    _require(report(fake, stream=sink) is False, "empty report returned True")
    _require("NOTHING WAS MEASURED" in sink.getvalue(), "empty report silent")
    return 2


def keep_lif_and_kwta_unit(stream) -> int:
    """10 -- keep-LIF fires, resets, and K-WTA keeps only k firers."""
    print("10. keep-LIF step + K-WTA unit vectors", file=stream)
    weights = [[0.0] * 16 for _ in range(16)]
    for i in range(6):
        weights[i][0] = 2.0
    bank = LifBank(
        name="unit",
        weights=weights,
        decay=[0.5] * 16,
        threshold=[1.0] * 16,
        i_drive=0.0,
    )
    stim = [1.0] + [0.0] * 15
    spikes = keep_lif_step(bank, stim, None)
    _require(spikes[:6] == [True] * 6, f"k=none spikes {spikes[:6]}")
    _require(spikes[6:] == [False] * 10, f"k=none tail {spikes[6:]}")
    _require(all(v == 0.0 for v, s in zip(bank.v, spikes) if s), "winners reset")

    raw = [True] * 6 + [False] * 10
    volts = [6.0, 5.0, 4.0, 3.0, 2.0, 1.0] + [0.0] * 10
    kept = apply_kwta(raw, volts, 4)
    _require(kept[:4] == [True] * 4, f"k=4 keep {kept[:4]}")
    _require(kept[4:6] == [False, False], f"k=4 drop {kept[4:6]}")
    _require(encode_q88_hex(1.0) == "0100", "Q8.8 1.0")
    return 6


def _cli_main():
    try:
        from .measure_hamming import main
    except ImportError:
        from measure_hamming import main

    return main


def nonfinite_i_drive_is_rejected(stream) -> int:
    """11 -- nan / inf / -inf --i-drive is a CLI error, never a quiet OK."""
    print("11. non-finite --i-drive is rejected", file=stream)
    main = _cli_main()
    for token in ("nan", "inf", "-inf"):
        try:
            rc = main([f"--i-drive={token}"])
        except SystemExit as exc:
            rc = exc.code
        _require(rc == 2, f"--i-drive={token} exited {rc}, expected 2")
    return 3


def malformed_expect_json_is_parse_error(tmp: Path, stream) -> int:
    """12 -- missing/nonnumeric pin fields are ParseError, not KeyError."""
    print("12. malformed --expect-json is ParseError", file=stream)
    missing = tmp / "missing_fields.json"
    missing.write_text("{}", encoding="utf-8")
    try:
        load_expected(missing)
    except ParseError as exc:
        _require("k_none_pct" in str(exc), f"wrong error: {exc}")
    else:
        raise SelfTestFailure("empty expect-json did not raise ParseError")
    bad = tmp / "nonnumeric.json"
    bad.write_text(
        json.dumps(
            {
                "k_none_pct": "nope",
                "k_none_bits": 2.0,
                "k_4_pct": 100.0,
                "k_4_bits": 3.0,
                "n_ticks": 4,
                "hidden_json_mem_mismatches": 2,
                "condition": "method-fixture",
            }
        ),
        encoding="utf-8",
    )
    try:
        load_expected(bad)
    except ParseError as exc:
        _require("k_none_pct" in str(exc), f"wrong error: {exc}")
    else:
        raise SelfTestFailure("nonnumeric expect-json did not raise ParseError")
    huge = tmp / "huge_int.json"
    huge.write_text('{"k_none_pct": 1' + "0" * 5000 + "}", encoding="utf-8")
    try:
        load_expected(huge)
    except ParseError:
        pass
    except ValueError as exc:
        raise SelfTestFailure("oversized pin integer leaked ValueError") from exc
    else:
        raise SelfTestFailure("oversized pin integer did not raise ParseError")
    return 5


def custom_method_fixture_without_pin_refused(stream) -> int:
    """13 -- method-fixture + custom --mem-dir without a pin is exit 2."""
    print("13. method-fixture + custom --mem-dir without pin is refused", file=stream)
    rc = _cli_main()(
        ["--condition", "method-fixture", "--mem-dir", str(SHIPPED_DIR)]
    )
    _require(rc == 2, f"custom method-fixture exited {rc}, expected 2")
    return 1


def overflowing_live_number_is_parse_error(stream) -> int:
    """14 -- a huge live-column int is ParseError, not OverflowError."""
    print("14. overflowing live column is ParseError (not OverflowError)", file=stream)
    row = _v3_row("gpu-000000")
    row["mem_util_pct"] = 10**1000
    try:
        select_samples([row], "all")
    except OverflowError as exc:
        raise SelfTestFailure("overflow leaked OverflowError") from exc
    except ParseError as exc:
        _require("mem_util_pct" in str(exc), f"wrong error: {exc}")
    else:
        raise SelfTestFailure("overflowing live column was accepted")
    return 3


def mixed_live_forbidden_refused(stream) -> int:
    """15 -- a live row plus a forbidden derived sensor is refused."""
    print("15. live + forbidden derived row is ParseError", file=stream)
    row = _v3_row("gpu-000000")
    row["tick_rate"] = 1.0
    try:
        select_samples([row], "all")
    except ParseError as exc:
        _require("tick_rate" in str(exc) or "derived" in str(exc), f"wrong error: {exc}")
    else:
        raise SelfTestFailure("mixed live+forbidden row was accepted")
    return 2


def exp024_delta_suppressed_for_wrong_split(stream) -> int:
    """16 -- exp-024 deltas are N/A unless the recorded holdout matches."""
    print("16. exp-024 deltas are N/A on a different holdout", file=stream)
    try:
        from .hamming_report import render
    except ImportError:
        from hamming_report import render

    proto = Protocol(
        condition="exp-024",
        weights_label="synthetic train split",
        float_json=FIXTURE_DIR / "snn_model.json",
        mem_dir=FIXTURE_DIR,
        encoder="synthetic",
        split="train",
        episodes="gpu-000000",
        seed="123",
        n_ticks=4,
        i_drive=0.05,
        stepper="synthetic",
        compared="synthetic",
    )
    scored = KResult(k=None, n_ticks=4, disagree_ticks=1, mean_bits=1.0, pct=10.0)
    fake = Measurement(
        protocol=proto,
        k_none=scored,
        k_4=KResult(k=4, n_ticks=4, disagree_ticks=1, mean_bits=1.0, pct=20.0),
        hidden_json_mem_mismatches=0,
        hidden_compared=256,
    )
    text = "\n".join(render(fake))
    _require("N/A" in text or "not comparable" in text, f"deltas not marked N/A: {text}")
    _require("delta_pct=" not in text, "subtracted exp-024 deltas on train split")
    return 2


def exp024_delta_suppressed_for_partial_holdout(stream) -> int:
    """17 -- sharing the holdout's endpoints is not being the holdout."""
    print("17. exp-024 deltas are N/A when a middle episode is missing", file=stream)
    try:
        from .hamming_core import (
            EXP024_TEST_EPISODE_IDS,
            EXP024_TEST_N_TICKS,
            episode_span_of_ids,
        )
        from .hamming_report import _exp024_comparable, render
    except ImportError:
        from hamming_core import (
            EXP024_TEST_EPISODE_IDS,
            EXP024_TEST_N_TICKS,
            episode_span_of_ids,
        )
        from hamming_report import _exp024_comparable, render

    def holdout(episodes: str) -> Measurement:
        n = EXP024_TEST_N_TICKS
        proto = Protocol(
            condition="exp-024",
            weights_label="synthetic",
            float_json=FIXTURE_DIR / "snn_model.json",
            mem_dir=FIXTURE_DIR,
            encoder="synthetic",
            split="test",
            episodes=episodes,
            seed="123",
            n_ticks=n,
            i_drive=0.05,
            stepper="synthetic",
            compared="synthetic",
        )
        return Measurement(
            protocol=proto,
            k_none=KResult(k=None, n_ticks=n, disagree_ticks=1, mean_bits=1.0, pct=10.0),
            k_4=KResult(k=4, n_ticks=n, disagree_ticks=1, mean_bits=1.0, pct=20.0),
            hidden_json_mem_mismatches=0,
            hidden_compared=256,
        )

    # Positive control: the exact recorded span is comparable.
    _require(
        _exp024_comparable(holdout(episode_span_of_ids(EXP024_TEST_EPISODE_IDS))),
        "the exact recorded holdout span was not treated as comparable",
    )
    # Same endpoints and tick count, one middle episode dropped: not the holdout.
    partial = [e for e in EXP024_TEST_EPISODE_IDS if e != "gpu-000185"]
    text = "\n".join(render(holdout(episode_span_of_ids(partial))))
    _require("N/A" in text or "not comparable" in text, f"deltas not marked N/A: {text}")
    _require("delta_pct=" not in text, "subtracted exp-024 deltas on a partial holdout")
    return 3


def recurrent_checkpoint_refused_feed_forward(stream) -> int:
    """18 -- a checkpoint that declares recurrence must never be scored
    feed-forward with the block silently dropped (GH #3 contract)."""
    print("18. a recurrent checkpoint is refused by the feed-forward harness", file=stream)
    variant = SHIPPED_DIR.parent / "merged_v2_rec_v1"
    try:
        measure(
            float_json=variant / "snn_model.json",
            mem_dir=variant,
            jsonl=FIXTURE_DIR / "holdout.jsonl",
            split="all",
            condition=CONDITION_SHIPPED,
            seed="n/a",
            i_drive=0.05,
            weights_label="must be refused",
        )
    except ParseError as exc:
        _require("recurrence" in str(exc), f"wrong error: {exc}")
    else:
        raise SelfTestFailure("the Hamming harness scored a recurrent checkpoint feed-forward")
    return 2


def _scenarios() -> tuple[tuple[Callable[..., int], ...], tuple[Callable[..., int], ...]]:
    standalone = (
        method_fixture_matches_pin,
        missing_episode_id_is_parse_error,
        interleaved_episodes_refused,
        wrong_pin_is_detected,
        unused_axons_stay_zero,
        empty_report_refuses,
        keep_lif_and_kwta_unit,
        nonfinite_i_drive_is_rejected,
        custom_method_fixture_without_pin_refused,
        overflowing_live_number_is_parse_error,
        mixed_live_forbidden_refused,
        exp024_delta_suppressed_for_wrong_split,
        exp024_delta_suppressed_for_partial_holdout,
        recurrent_checkpoint_refused_feed_forward,
    )
    tempdir = (
        empty_holdout_is_parse_error,
        malformed_jsonl_is_parse_error,
        unlabeled_merged_v2_refused_as_exp024,
        malformed_expect_json_is_parse_error,
    )
    return standalone, tempdir


def _suite_is_fully_accounted_for(sections_run: int, scenario_checks: int) -> int:
    scenario_requires = _require.calls
    _require(
        sections_run == SELF_TEST_SECTIONS,
        f"self-test ran {sections_run} numbered sections, expected "
        f"{SELF_TEST_SECTIONS} -- a scenario was dropped",
    )
    _require(
        scenario_requires == EXPECTED_REQUIRE_CALLS,
        f"scenarios made {scenario_requires} _require calls, expected "
        f"{EXPECTED_REQUIRE_CALLS} -- an assertion was added or removed "
        "without updating its scenario's count",
    )
    _require(
        scenario_checks - scenario_requires == EXPECTED_GUARD_CHECKS,
        f"scenarios reported {scenario_checks} assertions against "
        f"{scenario_requires} _require calls, leaving "
        f"{scenario_checks - scenario_requires} try/except guards, expected "
        f"{EXPECTED_GUARD_CHECKS}",
    )
    return 3


def self_test(stream=sys.stdout) -> bool:
    """Prove this harness actually rejects real regressions."""
    print("Hamming holdout harness self-test", file=stream)
    print("", file=stream)
    _require.calls = 0
    checks = 0
    sections_run = 0

    standalone, tempdir_scenarios = _scenarios()
    for scenario in standalone:
        checks += scenario(stream)
        sections_run += 1

    with tempfile.TemporaryDirectory(prefix="hamming-selftest-") as tmpdir:
        tmp = Path(tmpdir)
        for scenario in tempdir_scenarios:
            checks += scenario(tmp, stream)
            sections_run += 1

    checks += _suite_is_fully_accounted_for(sections_run, checks)
    print(
        f"SELF-TEST PASSED: {checks} assertions across {sections_run} sections.\n"
        "The harness rejects an empty holdout, malformed JSONL, a missing "
        "episode_id,\ninterleaved episodes, a wrong method pin, and unlabeled "
        "merged_v2 as exp-024;\nunused axons stay 0; report() refuses a 0-tick "
        "result; keep-LIF + K-WTA unit\nvectors hold; non-finite --i-drive, "
        "malformed expect-json, overflowing live\ncolumns, mixed forbidden "
        "sensors, and unpinned custom method-fixture paths\nare refused; "
        "exp-024 deltas stay N/A on a different holdout; a checkpoint that\n"
        "declares recurrence is refused; and the in-repo method fixture still\n"
        "matches its pin.",
        file=stream,
    )
    return True
