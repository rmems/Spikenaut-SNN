"""Scenarios 6, 8 and 9: the result-set and ``ParseError`` contracts.

What the verifier promises when it *cannot* check something. An empty or
short result set must FAIL rather than report a clean verification of
nothing, and every malformed model -- a list or null top level, a null
neuron, a nonnumeric or non-finite scalar, an oversized integer literal --
must exit through ``ParseError`` (exit code 2) rather than a traceback.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

try:  # package import: `python3 -m tools.verify_q88`
    from .q88_selftest import _require
except ImportError:  # direct script: `python3 tools/verify_q88.py`
    from q88_selftest import _require

try:  # package import: `python3 -m tools.verify_q88`
    from .q88_core import (
        EXPECTED_ARTIFACTS,
        EXPECTED_SECTIONS,
        MEM_OUTPUT,
        MODEL_JSON,
        N_INPUTS,
        N_OUTPUT_WEIGHTS,
        OUTPUT_WEIGHTS_CANON_SHA256,
        OUTPUT_WEIGHTS_HEX,
        ParseError,
        Q88RangeError,
        SelfTestFailure,
        ShippedPin,
        as_finite_float,
        check_signed_section,
        encode_q88,
        load_model,
        parse_mem,
        verify_shipped,
    )
    from .q88_report import report, worst_residual_lsb
except ImportError:  # direct script: `python3 tools/verify_q88.py`
    from q88_core import (
        EXPECTED_ARTIFACTS,
        EXPECTED_SECTIONS,
        MEM_OUTPUT,
        MODEL_JSON,
        N_INPUTS,
        N_OUTPUT_WEIGHTS,
        OUTPUT_WEIGHTS_CANON_SHA256,
        OUTPUT_WEIGHTS_HEX,
        ParseError,
        Q88RangeError,
        SelfTestFailure,
        ShippedPin,
        as_finite_float,
        check_signed_section,
        encode_q88,
        load_model,
        parse_mem,
        verify_shipped,
    )
    from q88_report import report, worst_residual_lsb


def _signed_only_run_passes() -> int:
    """A signed-only run passes and claims only the pin, with a 0.0 residual."""
    checks = 0
    # The signed-only path (max_residual_lsb is None) must still pass and
    # must still report 0.0 rather than raising.
    signed_only = [
        check_signed_section(
            "output weights (no residual)",
            "pin + signed invariants",
            parse_mem(MEM_OUTPUT),
            N_OUTPUT_WEIGHTS,
            pin=ShippedPin(
                hex_words=OUTPUT_WEIGHTS_HEX,
                canon_sha256=OUTPUT_WEIGHTS_CANON_SHA256,
            ),
        )
    ]
    _require(
        all(r.evidence.max_residual_lsb is None for r in signed_only),
        "signed-only section must carry no residual data (this is the crash path)",
    )
    _require(
        worst_residual_lsb(signed_only) == 0.0,
        "signed-only sections must report 0.0 residual, not raise",
    )
    silent = io.StringIO()
    signed_ok = report(signed_only, stream=silent)
    _require(signed_ok, "signed-only report() should pass and not raise")
    _require(
        "cross-validated" not in silent.getvalue(),
        f"signed-only report() claimed a float cross-validation that never "
        f"happened: {silent.getvalue()!r}",
    )
    checks += 4
    return checks


def _short_run_fails(shipped) -> int:
    """A run missing a section fails even though every section it has is clean."""
    checks = 0
    # A short run -- a section dropped upstream -- must fail too, even
    # though every section it does contain verifies cleanly.
    _require(
        shipped[0].ok,
        "sanity: the first shipped section must itself verify, or the "
        "short-run assertion below would prove nothing",
    )
    silent = io.StringIO()
    short_run_ok = report(
        shipped[:1],
        stream=silent,
        expected_sections=EXPECTED_SECTIONS,
        expected_artifacts=EXPECTED_ARTIFACTS,
    )
    _require(
        not short_run_ok,
        f"report() accepted 1 of {EXPECTED_SECTIONS} sections as a clean run",
    )
    _require(
        "INCOMPLETE RUN" in silent.getvalue(),
        f"short run was rejected, but not as an incomplete run: "
        f"{silent.getvalue()!r}",
    )
    checks += 3
    return checks


def empty_result_set_fails(tmp: Path, stream) -> int:
    """An empty or short result set FAILS, and worst_residual_lsb stays crash-safe."""
    del tmp  # uniform scenario signature; this one writes no scratch files
    shipped = verify_shipped()
    checks = 0
    #
    # Two distinct concerns that the previous version conflated, and got
    # backwards in the second case:
    #
    #   * Amazon Q: max() over an empty generator of max_residual_lsb
    #     raises ValueError. worst_residual_lsb() must return 0.0 for an
    #     empty list and for signed-only sections. Still true, still tested.
    #   * Codex: report([]) returned SUCCESS -- all([]) is True -- printing
    #     "OK: 0 Q8.8 values verified" plus the claim that 48 output
    #     weights were pinned, having read nothing. The old section 6
    #     asserted that pass and so enshrined the bug. An empty (or short)
    #     result set is now a hard failure.
    print(
        "6. empty/short results FAIL; empty residual list stays crash-safe",
        file=stream,
    )
    _require(
        worst_residual_lsb([]) == 0.0,
        "empty results must report 0.0 residual, not raise "
        "(the max()-on-empty crash must stay fixed)",
    )
    silent = io.StringIO()
    empty_ok = report([], stream=silent)
    _require(
        not empty_ok,
        "report([]) returned SUCCESS -- a verifier that reports a clean "
        "pass having checked nothing is a false-confidence failure",
    )
    empty_text = silent.getvalue()
    _require(
        "OK:" not in empty_text,
        f"report([]) failed but still printed an OK line: {empty_text!r}",
    )
    _require(
        "pinned" not in empty_text,
        "report([]) claimed output weights were pinned, having read nothing",
    )
    checks += 4

    checks += _short_run_fails(shipped)

    checks += _signed_only_run_passes()
    print(
        "   ok: report([]) and a 1-of-4-section run both FAIL; "
        "worst_residual_lsb([]) returns 0.0;\n"
        "       signed-only run passes and claims only the pin",
        file=stream,
    )
    print("", file=stream)
    return checks


def malformed_model_json_parse_error(tmp: Path, stream) -> int:
    """Malformed model JSON exits through ParseError, never a traceback."""
    checks = 0
    #
    # CodeRabbit: a list/null top level made model.get("neurons") raise
    # AttributeError, and a null neuron made `key not in neuron` raise
    # TypeError. Both escaped the documented ParseError path, so main()
    # would traceback instead of returning exit code 2. Shape must be
    # validated before any field is read.
    print(
        "8. non-object model records are REJECTED as ParseError",
        file=stream,
    )
    good_model = load_model(MODEL_JSON)

    null_neuron_model = json.loads(json.dumps(good_model))
    null_neuron_model["neurons"][3] = None

    scalar_weights_model = json.loads(json.dumps(good_model))
    scalar_weights_model["neurons"][0]["weights"] = N_INPUTS

    malformed_models = [
        ("list top level", []),
        ("null top level", None),
        ("null neuron entry", null_neuron_model),
        ("non-list neuron weights", scalar_weights_model),
    ]
    for label, payload in malformed_models:
        bad_path = tmp / f"model_{label.replace(' ', '_')}.json"
        bad_path.write_text(json.dumps(payload), encoding="utf-8")
        try:
            load_model(bad_path)
        except ParseError:
            pass
        except Exception as exc:  # AttributeError / TypeError = the bug
            raise SelfTestFailure(
                f"{label}: load_model raised "
                f"{type(exc).__name__}({exc}), expected ParseError"
            ) from exc
        else:
            raise SelfTestFailure(
                f"{label}: load_model ACCEPTED a malformed model"
            )
        checks += 1
    print(
        f"   ok: {len(malformed_models)} malformed models "
        f"({', '.join(label for label, _ in malformed_models)}) "
        f"all raise ParseError",
        file=stream,
    )
    print("", file=stream)
    return checks


def _reject_oversized_json_int(tmp: Path) -> int:
    """An integer literal past CPython's digit limit must be a ParseError.

    json.loads raises a bare ValueError -- not JSONDecodeError -- for a literal
    longer than sys.get_int_max_str_digits() (4300 by default on 3.11+), so it
    would escape main()'s handlers as a traceback and exit 1 instead of the
    documented parse-error exit 2.
    """
    path = tmp / "model_huge_int.json"
    path.write_text('{"neurons": [1' + "0" * 5000 + "]}", encoding="utf-8")
    try:
        load_model(path)
    except ParseError:
        return 1
    except ValueError as exc:
        raise SelfTestFailure(
            f"load_model leaked {type(exc).__name__} on an oversized JSON integer"
        ) from exc
    raise SelfTestFailure("load_model ACCEPTED an oversized JSON integer")


def _finite_float_positive_controls() -> int:
    """The scalar guard must not reject valid numbers, or it is just breaking the tool."""
    checks = 0
    # Positive controls first: the guard must not reject valid numbers,
    # including a JSON integer, or it would just be breaking the tool.
    _require(
        as_finite_float(1, "int") == 1.0,
        "as_finite_float rejected a valid JSON integer",
    )
    _require(
        as_finite_float(-0.02734375, "float") == -0.02734375,
        "as_finite_float rejected a valid JSON float",
    )
    try:
        as_finite_float(1e308, "neurons[0].threshold")
    except ParseError as exc:
        _require(
            "neurons[0].threshold" in str(exc) and "Q8.8" in str(exc),
            f"huge-finite ParseError must name the field and Q8.8; got {exc}",
        )
    except OverflowError as exc:
        raise SelfTestFailure(
            "as_finite_float leaked OverflowError on 1e308"
        ) from exc
    else:
        raise SelfTestFailure("as_finite_float ACCEPTED 1e308")
    try:
        encode_q88(1e308)
    except Q88RangeError:
        pass
    except OverflowError as exc:
        raise SelfTestFailure("encode_q88 leaked OverflowError on 1e308") from exc
    else:
        raise SelfTestFailure("encode_q88 ACCEPTED 1e308")
    checks += 5
    return checks


BAD_SCALARS: list[tuple[str, str, object]] = [
    ("null threshold", "threshold", None),
    ("string threshold", "threshold", "0.5"),
    ("object threshold", "threshold", {"value": 0.5}),
    ("bool threshold", "threshold", True),
    ("nan threshold", "threshold", float("nan")),
    ("posinf threshold", "threshold", float("inf")),
    ("null decay", "decay_rate", None),
    ("list decay", "decay_rate", [0.9]),
    ("bool decay", "decay_rate", False),
    ("neginf decay", "decay_rate", float("-inf")),
    ("huge finite threshold", "threshold", 1e308),
]


BAD_WEIGHTS: list[tuple[str, object]] = [
    ("null weight", None),
    ("object weight", {"w": 0.1}),
    ("string weight", "0.1"),
    ("bool weight", True),
    ("nan weight", float("nan")),
]


def _reject_each_bad_scalar(bad_scalars, bad_weights, _reject, good_model, tmp) -> int:
    """Every bad scalar and bad weight must raise ParseError."""
    checks = 0
    for label, field, payload in bad_scalars:
        broken = json.loads(json.dumps(good_model))
        broken["neurons"][2][field] = payload
        _reject(label, broken, tmp / f"model_{label.replace(' ', '_')}.json")
        checks += 1
    for label, payload in bad_weights:
        broken = json.loads(json.dumps(good_model))
        broken["neurons"][5]["weights"][7] = payload
        _reject(label, broken, tmp / f"model_{label.replace(' ', '_')}.json")
        checks += 1
    return checks


def nonnumeric_fields_parse_error(tmp: Path, stream) -> int:
    """Nonnumeric and overflowing scalar fields exit through ParseError."""
    good_model = load_model(MODEL_JSON)
    checks = 0
    #
    # Codex: load_model() validated container shape but not scalar type,
    # so a model with the right shape and {"threshold": null} -- or an
    # object-valued weight -- passed every check and then raised an
    # uncaught TypeError at float(), tracebacking instead of honoring the
    # documented exit-code-2 parse-error contract.
    #
    # bool is the subtle one: it subclasses int, so isinstance(True, int)
    # is True and a naive numeric guard would silently encode True as 1.0.
    print(
        "9. nonnumeric threshold/decay/weight values are REJECTED as ParseError",
        file=stream,
    )
    checks += _reject_oversized_json_int(tmp)

    checks += _finite_float_positive_controls()

    def _reject(label: str, payload: object, bad_path: Path) -> None:
        """Assert load_model(bad_path) raises ParseError, nothing else."""
        bad_path.write_text(json.dumps(payload), encoding="utf-8")
        try:
            load_model(bad_path)
        except ParseError:
            return
        except Exception as exc:  # TypeError from float() = the bug
            raise SelfTestFailure(
                f"{label}: load_model raised {type(exc).__name__}({exc}), "
                f"expected ParseError"
            ) from exc
        raise SelfTestFailure(
            f"{label}: load_model ACCEPTED a nonnumeric field"
        )

    checks += _reject_each_bad_scalar(BAD_SCALARS, BAD_WEIGHTS, _reject, good_model, tmp)

    print(
        f"   ok: {len(BAD_SCALARS)} bad thresholds/decays and "
        f"{len(BAD_WEIGHTS)} bad weights all raise ParseError\n"
        f"       (null, string, object/list, bool, NaN, +-inf, 1e308); "
        f"valid ints and floats still accepted",
        file=stream,
    )
    print("", file=stream)
    return checks
