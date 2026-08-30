"""Self-test for the Q8.8 verifier: prove the checker can actually fail.

A checker that cannot fail is worthless, and so is one that passes without
reading anything. Each numbered scenario below builds a specific regression
and asserts ``q88_core`` rejects it: clamp-to-zero output weights (issue #15),
1-LSB hidden-weight drift, truncated files, well-formed output-weight
corruption against the shipped-file pin, malformed model JSON (a list/null top
level, a null neuron, a non-numeric threshold/decay/weight, or a huge-but-finite
``1e308``) and invalid-UTF-8 ``.mem`` files surfacing as ``ParseError`` rather
than a traceback / ``OverflowError`` / ``UnicodeDecodeError``, an empty or short
result set (which must FAIL while ``worst_residual_lsb`` stays crash-safe), and
a CRLF checkout of the output-weight image (which must still verify).

Scratch files are written to a temp directory; nothing here ever writes into
``dataset/``.

Run via the CLI::

    python3 tools/verify_q88.py --self-test
"""

from __future__ import annotations

import hashlib
import io
import json
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path

from q88_core import (
    EXPECTED_ARTIFACTS,
    EXPECTED_SECTIONS,
    MEM_HIDDEN,
    MEM_OUTPUT,
    MODEL_JSON,
    N_INPUTS,
    N_OUTPUT_WEIGHTS,
    OUTPUT_WEIGHTS_CANON_SHA256,
    OUTPUT_WEIGHTS_HEX,
    ParseError,
    Q88RangeError,
    SectionResult,
    SelfTestFailure,
    ShippedPin,
    as_finite_float,
    canonical_mem_digest,
    check_against_floats,
    check_signed_section,
    decode_q88,
    encode_q88,
    encode_q88_clamp_to_zero,
    encode_q88_hex,
    load_model,
    parse_mem,
    read_utf8_text,
    verify_shipped,
)
from q88_report import report, worst_residual_lsb

def _require(condition: bool, message: str) -> None:
    """Assert a self-test invariant.

    Raises:
        SelfTestFailure: ``condition`` is false. Used instead of bare
            ``assert`` so the checks survive ``python -O``.

    """
    if not condition:
        raise SelfTestFailure(message)


def _write_mem(path: Path, words: list[int]) -> None:
    """Write raw 16-bit words as a .mem image, one 4-digit hex line each.

    Self-test scratch files only -- callers pass a temp path. Nothing here
    ever writes into ``dataset/``.
    """
    path.write_text("".join(f"{w:04X}\n" for w in words), encoding="utf-8")


# Numbered scenarios self_test() must run. Counted and asserted at the end so
# a section deleted or accidentally skipped fails loudly instead of silently
# shrinking the suite.
SELF_TEST_SECTIONS = 11


def _codec_unit_vectors(stream) -> int:
    """Decode and round-trip the Q8.8 vectors documented in README.md."""
    checks = 0
    print("1. Q8.8 codec unit vectors", file=stream)
    vectors = [
        ("0100", 1.0),
        ("00DA", 218 / 256),
        ("00CC", 204 / 256),
        ("0000", 0.0),
        ("FFF9", -7 / 256),
        ("FFFF", -1 / 256),
        ("8000", -128.0),
        ("7FFF", 32767 / 256),
    ]
    for hex_text, expected in vectors:
        got = decode_q88(int(hex_text, 16))
        _require(
            got == expected,
            f"decode_q88({hex_text}) = {got!r}, expected {expected!r}",
        )
        back = encode_q88_hex(got)
        _require(
            back == hex_text,
            f"round-trip {hex_text} -> {got!r} -> {back}, expected {hex_text}",
        )
        checks += 2
    print(f"   ok: {len(vectors)} vectors decode and round-trip", file=stream)
    print("", file=stream)
    return checks


def _shipped_artifacts_verify(stream) -> int:
    """The four shipped artifacts must verify clean, with every section run."""
    checks = 0
    print("2. shipped artifacts verify", file=stream)
    shipped = verify_shipped()
    _require(
        len(shipped) == EXPECTED_SECTIONS,
        f"verify_shipped() returned {len(shipped)} sections, expected "
        f"{EXPECTED_SECTIONS} -- a section was dropped",
    )
    _require(
        all(r.ok for r in shipped),
        "the shipped .mem files do not verify -- see the default (non-self-test) run",
    )
    _require(
        report(
            shipped,
            stream=io.StringIO(),
            expected_sections=EXPECTED_SECTIONS,
            expected_artifacts=EXPECTED_ARTIFACTS,
        ),
        "the shipped artifacts did not survive the full report() verdict",
    )
    print(
        f"   ok: {len(shipped)} sections, {sum(r.count for r in shipped)} values, "
        f"0 mismatches",
        file=stream,
    )
    checks += 3
    print("", file=stream)
    return checks


def _clamp_to_zero_rejected(tmp: Path, stream) -> int:
    """Reject output weights re-encoded by the issue #15 clamp-to-zero encoder.

    The one that matters. The clamped file is still 48 valid hex lines, so
    only a sign-aware check catches it -- and it must be caught both against
    the true floats and by the standalone signed-section invariants.
    """
    checks = 0
    print(
        "3. clamp-to-zero output weights are REJECTED (issue #15 hazard)",
        file=stream,
    )
    real = parse_mem(MEM_OUTPUT)
    real_floats = [e.value for e in real]
    n_negative = sum(1 for v in real_floats if v < 0)
    _require(
        n_negative > 0,
        "shipped output weights have no negatives; self-test cannot "
        "demonstrate the clamp regression",
    )

    clamped_path = tmp / "parameters_output_weights_clamped.mem"
    _write_mem(clamped_path, [encode_q88_clamp_to_zero(v) for v in real_floats])
    clamped = parse_mem(clamped_path)

    # 3a. against the true float vector, every negative must be flagged.
    against_truth = check_against_floats(
        "output weights (clamp-to-zero candidate)",
        "true float vector vs clamp-to-zero encoding",
        real_floats,
        [f"output weight {i}" for i in range(len(real_floats))],
        clamped,
    )
    _require(
        not against_truth.ok,
        "verifier ACCEPTED a clamp-to-zero output-weight file compared "
        "against the true floats -- it cannot fail, so it is worthless",
    )
    _require(
        len(against_truth.mismatches) == n_negative,
        f"expected {n_negative} flagged values (one per inhibitory weight), "
        f"got {len(against_truth.mismatches)}",
    )
    checks += 2
    print(
        f"   ok: rejected, {len(against_truth.mismatches)} of "
        f"{len(real_floats)} values flagged "
        f"(exactly the {n_negative} inhibitory weights)",
        file=stream,
    )
    for mismatch in against_truth.mismatches[:3]:
        print(mismatch.render(), file=stream)
    if len(against_truth.mismatches) > 3:
        print(
            f"       ... and {len(against_truth.mismatches) - 3} more",
            file=stream,
        )

    # 3b. and the standalone signed-section check must reject it too,
    #     without needing any float reference at all.
    invariants = check_signed_section(
        "output weights (clamp-to-zero candidate)",
        "signed two's-complement invariants",
        clamped,
        N_OUTPUT_WEIGHTS,
    )
    _require(
        not invariants.ok,
        "sign-integrity check ACCEPTED a clamp-to-zero file -- the shipped "
        "output weights would not be protected, since snn_model.json has "
        "no output layer to compare against",
    )
    _require(
        any("SIGN INTEGRITY FAILURE" in n for n in invariants.notes),
        "clamp-to-zero file was rejected, but not for losing its negative "
        f"weights; notes were {invariants.notes}",
    )
    checks += 2
    print(
        "   ok: sign-integrity check also rejects it with no float "
        "reference needed",
        file=stream,
    )
    print("", file=stream)
    return checks


def _corrupted_hidden_weight_rejected(tmp: Path, stream) -> int:
    """Reject a single 1-LSB drift in the JSON-backed hidden-weight section."""
    checks = 0
    print("4. a single corrupted hidden weight is REJECTED", file=stream)
    hidden_entries = parse_mem(MEM_HIDDEN)
    model = load_model(MODEL_JSON)
    hidden_floats = [
        float(w) for neuron in model["neurons"] for w in neuron["weights"]
    ]
    corrupt_words = [e.word for e in hidden_entries]
    corrupt_words[137] ^= 0x0001  # one LSB, the smallest possible drift
    corrupt_path = tmp / "parameters_weights_corrupt.mem"
    _write_mem(corrupt_path, corrupt_words)
    corrupted = check_against_floats(
        "hidden weights (corrupt candidate)",
        "snn_model.json vs 1-LSB-corrupted mem",
        hidden_floats,
        [f"hidden {i}" for i in range(len(hidden_floats))],
        parse_mem(corrupt_path),
    )
    _require(
        not corrupted.ok and len(corrupted.mismatches) == 1,
        f"expected exactly 1 mismatch from a 1-LSB corruption, got "
        f"{len(corrupted.mismatches)}",
    )
    _require(
        corrupted.mismatches[0].index == 137,
        f"flagged index {corrupted.mismatches[0].index}, expected 137",
    )
    checks += 2
    print("   ok: rejected, 1 mismatch at the corrupted index", file=stream)
    print(corrupted.mismatches[0].render(), file=stream)
    print("", file=stream)
    return checks


def _truncated_file_rejected(tmp: Path, stream) -> int:
    """Reject a .mem image that stops short of its expected value count."""
    real = parse_mem(MEM_OUTPUT)
    checks = 0
    print("5. a truncated .mem is REJECTED", file=stream)
    short_path = tmp / "parameters_output_weights_short.mem"
    _write_mem(short_path, [e.word for e in real][:-1])
    short = check_signed_section(
        "output weights (truncated candidate)",
        "signed two's-complement invariants",
        parse_mem(short_path),
        N_OUTPUT_WEIGHTS,
    )
    _require(not short.ok, "verifier ACCEPTED a 47-value output-weight file")
    checks += 1
    print("   ok: rejected on value count", file=stream)
    print("", file=stream)
    return checks


def _empty_result_set_fails(tmp: Path, stream) -> int:
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
    print(
        "   ok: report([]) and a 1-of-4-section run both FAIL; "
        "worst_residual_lsb([]) returns 0.0;\n"
        "       signed-only run passes and claims only the pin",
        file=stream,
    )
    print("", file=stream)
    return checks


def _wellformed_corruption_rejected(tmp: Path, stream) -> int:
    """Reject well-formed output-weight corruption that still round-trips.

    An FFF9->FFF8 swap decodes and re-encodes cleanly, so only the
    shipped-file pin catches it.
    """
    real = parse_mem(MEM_OUTPUT)
    checks = 0
    #
    # Sign-integrity + decode->re-encode accept any well-formed 48-word
    # file with at least one high-bit word. The gold pin is what makes
    # FFF9->FFF8 and "47 zeros + FFFF" fail.
    print(
        "7. well-formed output-weight corruption is REJECTED (shipped-file pin)",
        file=stream,
    )
    gold = tuple(e.text.upper() for e in real)
    _require(
        gold == OUTPUT_WEIGHTS_HEX,
        "gold tuple derived from the shipped file does not match the pin",
    )
    _require(
        canonical_mem_digest(real) == OUTPUT_WEIGHTS_CANON_SHA256,
        "shipped output-weight canonical-token sha256 does not match the pin",
    )
    _require(
        len(gold) == N_OUTPUT_WEIGHTS,
        f"shipped file has {len(gold)} words, expected {N_OUTPUT_WEIGHTS}",
    )
    _require(
        gold[0] == "FFF9",
        f"self-test expects shipped word 0 to be FFF9, got {gold[0]}",
    )

    mutated_words = [e.word for e in real]
    mutated_words[0] = 0xFFF8
    mut_path = tmp / "parameters_output_weights_fff8.mem"
    _write_mem(mut_path, mutated_words)
    unpinned_fff8 = check_signed_section(
        "output weights (FFF9->FFF8, no pin)",
        "sign-integrity + round-trip only",
        parse_mem(mut_path),
        N_OUTPUT_WEIGHTS,
    )
    _require(
        unpinned_fff8.ok,
        "sanity: FFF9->FFF8 still passes sign+round-trip; the pin is load-bearing",
    )
    pinned_fff8 = check_signed_section(
        "output weights (FFF9->FFF8)",
        "gold hex pin",
        parse_mem(mut_path),
        N_OUTPUT_WEIGHTS,
        pin=ShippedPin(hex_words=gold),
    )
    _require(
        not pinned_fff8.ok,
        "verifier ACCEPTED FFF9->FFF8 corruption against the gold pin",
    )
    _require(
        len(pinned_fff8.mismatches) == 1 and pinned_fff8.mismatches[0].index == 0,
        f"expected 1 mismatch at index 0, got {pinned_fff8.mismatches}",
    )
    checks += 6
    print("   ok: FFF9->FFF8 rejected by gold pin, accepted without it", file=stream)

    junk_path = tmp / "parameters_output_weights_junk.mem"
    _write_mem(junk_path, [0] * 47 + [0xFFFF])
    unpinned_junk = check_signed_section(
        "output weights (47 zeros + FFFF, no pin)",
        "sign-integrity + round-trip only",
        parse_mem(junk_path),
        N_OUTPUT_WEIGHTS,
    )
    _require(
        unpinned_junk.ok,
        "sanity: 47 zeros + FFFF still passes sign+round-trip; the pin is load-bearing",
    )
    pinned_junk = check_signed_section(
        "output weights (47 zeros + FFFF)",
        "gold hex pin",
        parse_mem(junk_path),
        N_OUTPUT_WEIGHTS,
        pin=ShippedPin(hex_words=gold),
    )
    _require(
        not pinned_junk.ok,
        "verifier ACCEPTED 47 zeros + one FFFF against the gold pin",
    )
    checks += 2
    print("   ok: 47 zeros + FFFF rejected by gold pin, accepted without it", file=stream)
    print("", file=stream)
    return checks


def _malformed_model_json_parse_error(tmp: Path, stream) -> int:
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


def _nonnumeric_fields_parse_error(tmp: Path, stream) -> int:
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
    checks += 4

    bad_scalars = [
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
    bad_weights = [
        ("null weight", None),
        ("object weight", {"w": 0.1}),
        ("string weight", "0.1"),
        ("bool weight", True),
        ("nan weight", float("nan")),
    ]

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

    print(
        f"   ok: {len(bad_scalars)} bad thresholds/decays and "
        f"{len(bad_weights)} bad weights all raise ParseError\n"
        f"       (null, string, object/list, bool, NaN, +-inf, 1e308); "
        f"valid ints and floats still accepted",
        file=stream,
    )
    print("", file=stream)
    return checks


def _crlf_checkout_still_verifies(tmp: Path, stream) -> int:
    """A CRLF checkout of the output-weight image still verifies.

    The pin hashes canonical tokens, not raw bytes, precisely so this holds.
    """
    real = parse_mem(MEM_OUTPUT)
    checks = 0
    #
    # Codex: the pin used to hash the file's raw bytes. On a Windows
    # checkout with Git's core.autocrlf, the same 48 words materialize
    # with CRLF endings, so the hash failed on a semantically unchanged
    # artifact. The pin now hashes the canonical token sequence instead
    # (and .gitattributes pins *.mem to LF as defense in depth).
    print(
        "10. a CRLF-line-ending copy of the shipped file still verifies",
        file=stream,
    )
    # Both fixtures are *constructed*, never assumed from the checkout.
    # Reading the file and calling it the LF form is wrong on a CRLF
    # checkout -- there the two fixtures come out identical and the
    # negative control below fails, reporting a bug in the file that is
    # really a bug in this test. Normalize to LF first, then build CRLF
    # from it, so both forms exist whatever git handed us.
    shipped_bytes = MEM_OUTPUT.read_bytes()
    lf_bytes = shipped_bytes.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    crlf_bytes = lf_bytes.replace(b"\n", b"\r\n")
    _require(
        b"\r\n" in crlf_bytes and b"\r" not in lf_bytes,
        "CRLF fixture is not actually CRLF -- this section would prove nothing",
    )
    _require(
        crlf_bytes != lf_bytes,
        "the two line-ending fixtures are identical -- the file has no newline",
    )
    _require(
        hashlib.sha256(crlf_bytes).hexdigest()
        != hashlib.sha256(lf_bytes).hexdigest(),
        "negative control: a raw-byte hash must differ across line endings, "
        "otherwise this section is not testing the reported bug",
    )
    crlf_path = tmp / "parameters_output_weights_crlf.mem"
    crlf_path.write_bytes(crlf_bytes)
    crlf_entries = parse_mem(crlf_path)
    _require(
        len(crlf_entries) == N_OUTPUT_WEIGHTS,
        f"CRLF copy parsed to {len(crlf_entries)} words, expected "
        f"{N_OUTPUT_WEIGHTS}",
    )
    _require(
        canonical_mem_digest(crlf_entries) == OUTPUT_WEIGHTS_CANON_SHA256,
        "the canonical-token pin is still line-ending dependent",
    )
    crlf_result = check_signed_section(
        "output weights (CRLF checkout)",
        "shipped-file canonical sha256 + gold hex pin",
        crlf_entries,
        N_OUTPUT_WEIGHTS,
        pin=ShippedPin(
            hex_words=OUTPUT_WEIGHTS_HEX,
            canon_sha256=OUTPUT_WEIGHTS_CANON_SHA256,
        ),
    )
    _require(
        crlf_result.ok,
        f"a CRLF copy of the shipped output weights failed verification: "
        f"{crlf_result.notes}",
    )
    checks += 5

    # Same idea one step further: lower-case words with padding around
    # them are the same artifact, and canonicalization must say so.
    noisy_path = tmp / "parameters_output_weights_noisy.mem"
    noisy_path.write_text(
        "".join(f"  {e.text.lower()}  \r\n" for e in real), encoding="utf-8"
    )
    _require(
        canonical_mem_digest(parse_mem(noisy_path)) == OUTPUT_WEIGHTS_CANON_SHA256,
        "canonicalization must normalize hex case and surrounding whitespace",
    )
    checks += 1
    print(
        "   ok: CRLF copy verifies against both pins "
        "(raw bytes differ, canonical tokens do not);\n"
        "       lower-case + padded words canonicalize identically",
        file=stream,
    )
    print("", file=stream)
    return checks


def _reject_substituted_sections(shipped: list[SectionResult]) -> int:
    """A duplicated section must not pass as a complete run.

    Four individually clean results can still be the wrong four: this
    checks parameters_weights.mem twice and never opens the output
    weights, keeping the arity guard satisfied the whole time.
    """
    checks = 0
    hidden = next(r for r in shipped if "parameters_weights.mem" in r.name)
    substituted = [hidden, hidden] + [
        r
        for r in shipped
        if "parameters.mem" in r.name or "parameters_decay.mem" in r.name
    ]
    _require(
        len(substituted) == EXPECTED_SECTIONS,
        "the substitution fixture must keep the section count, or it would "
        "be caught by the arity guard and prove nothing",
    )
    _require(
        all(r.ok for r in substituted),
        "every substituted section must be individually clean, or the "
        "verdict would fail for the wrong reason",
    )
    subbed_out = io.StringIO()
    _require(
        not report(
            substituted,
            stream=subbed_out,
            expected_sections=EXPECTED_SECTIONS,
            expected_artifacts=EXPECTED_ARTIFACTS,
        ),
        "report() ACCEPTED a run that checked parameters_weights.mem twice "
        "and parameters_output_weights.mem never",
    )
    _require(
        "parameters_output_weights.mem was checked 0 time(s)"
        in subbed_out.getvalue(),
        "the failure must name the artifact that went unchecked",
    )
    # Negative control: without the identity guard the arity guard alone
    # lets it through, so the new check is what is doing the work.
    _require(
        report(
            substituted,
            stream=io.StringIO(),
            expected_sections=EXPECTED_SECTIONS,
        ),
        "the arity guard alone should still accept the substitution -- if "
        "it does not, this section is not testing the identity guard",
    )
    checks += 5
    return checks


def _reject_unreadable_artifacts(tmp: Path) -> int:
    """An unreadable-but-present artifact takes the same exit-2 path.

    A read-protected file is the realistic case, but it cannot be
    exercised as root, so a directory and a vanished path stand in --
    both raise OSError from read_text after is_file() has passed or
    been bypassed, which is the failure being guarded.
    """
    checks = 0
    for label, target in (
        ("a directory", tmp),
        ("a vanished file", tmp / "does_not_exist.mem"),
    ):
        try:
            read_utf8_text(target)
        except ParseError:
            pass
        except OSError as exc:
            raise SelfTestFailure(
                f"read_utf8_text leaked {type(exc).__name__} on {label}"
            ) from exc
        else:
            raise SelfTestFailure(f"read_utf8_text ACCEPTED {label}")
        checks += 1
    return checks


def _invalid_utf8_parse_error(tmp: Path, stream) -> int:
    """Invalid-UTF-8 and unreadable .mem files exit through ParseError."""
    shipped = verify_shipped()
    checks = 0
    #
    # Codex: parse_mem() opened the file as UTF-8 and iterated lines;
    # a corrupted image with invalid UTF-8 raised UnicodeDecodeError,
    # which main() does not catch. Wrap the decode as ParseError
    # (same pattern as the JSON read path). Temp file only -- never
    # mutate dataset/merged_v2.
    print(
        "11. a .mem with invalid UTF-8 is REJECTED as ParseError",
        file=stream,
    )
    bad_utf8_path = tmp / "parameters_output_weights_bad_utf8.mem"
    bad_utf8_path.write_bytes(b"FFF9\n\xff\xfe\n001E\n")
    try:
        parse_mem(bad_utf8_path)
    except ParseError:
        pass
    except UnicodeDecodeError as exc:
        raise SelfTestFailure(
            "parse_mem leaked UnicodeDecodeError on invalid UTF-8"
        ) from exc
    except Exception as exc:
        raise SelfTestFailure(
            f"parse_mem raised {type(exc).__name__}({exc}), expected ParseError"
        ) from exc
    else:
        raise SelfTestFailure("parse_mem ACCEPTED a .mem with invalid UTF-8")
    try:
        read_utf8_text(bad_utf8_path)
    except ParseError:
        pass
    except UnicodeDecodeError as exc:
        raise SelfTestFailure(
            "read_utf8_text leaked UnicodeDecodeError"
        ) from exc
    else:
        raise SelfTestFailure("read_utf8_text ACCEPTED invalid UTF-8")
    checks += 2

    checks += _reject_substituted_sections(shipped)
    print(
        "   ok: a duplicated section FAILS on artifact identity "
        "(arity alone accepts it)",
        file=stream,
    )

    # Emitted text must survive a non-UTF-8 console. A Windows process
    # under cp1252 raised UnicodeEncodeError on the arrows in the summary,
    # so a *successful* verification exited 1 with a traceback.
    source = Path(__file__).read_text(encoding="utf-8")
    non_ascii = sorted({c for c in source if ord(c) > 127})
    _require(
        not non_ascii,
        f"non-ASCII in this file may reach stdout: {non_ascii}",
    )
    cp1252_out = io.TextIOWrapper(
        io.BytesIO(), encoding="cp1252", errors="strict", newline=""
    )
    try:
        report(
            shipped,
            stream=cp1252_out,
            expected_sections=EXPECTED_SECTIONS,
            expected_artifacts=EXPECTED_ARTIFACTS,
        )
        cp1252_out.flush()
    except UnicodeEncodeError as exc:
        raise SelfTestFailure(
            f"report() output is not encodable as cp1252: {exc}"
        ) from exc
    checks += 2
    print(
        "   ok: output is ASCII and encodes cleanly to a cp1252 console",
        file=stream,
    )

    checks += _reject_unreadable_artifacts(tmp)
    print("   ok: invalid UTF-8 .mem raises ParseError, not UnicodeDecodeError", file=stream)
    print("   ok: an unreadable artifact raises ParseError, not OSError", file=stream)
    print("", file=stream)
    return checks


# The numbered scenarios, in the order self_test() runs and prints them. These
# tuples ARE the suite, so the count is asserted against SELF_TEST_SECTIONS
# rather than derived from len(): deriving it would make a dropped scenario
# self-consistent and therefore invisible.
_STANDALONE_SCENARIOS: tuple[Callable[[object], int], ...] = (
    _codec_unit_vectors,
    _shipped_artifacts_verify,
)
_TEMPDIR_SCENARIOS: tuple[Callable[[Path, object], int], ...] = (
    _clamp_to_zero_rejected,
    _corrupted_hidden_weight_rejected,
    _truncated_file_rejected,
    _empty_result_set_fails,
    _wellformed_corruption_rejected,
    _malformed_model_json_parse_error,
    _nonnumeric_fields_parse_error,
    _crlf_checkout_still_verifies,
    _invalid_utf8_parse_error,
)


def self_test(stream=sys.stdout) -> bool:
    """Prove this checker actually rejects real regressions."""
    print("Q8.8 verifier self-test", file=stream)
    print("", file=stream)
    checks = 0
    sections_run = 0

    for scenario in _STANDALONE_SCENARIOS:
        checks += scenario(stream)
        sections_run += 1

    with tempfile.TemporaryDirectory(prefix="q88-selftest-") as tmpdir:
        tmp = Path(tmpdir)
        for scenario in _TEMPDIR_SCENARIOS:
            checks += scenario(tmp, stream)
            sections_run += 1

    _require(
        sections_run == SELF_TEST_SECTIONS,
        f"self-test ran {sections_run} numbered sections, expected "
        f"{SELF_TEST_SECTIONS} -- a scenario was dropped",
    )
    checks += 1

    print(
        f"SELF-TEST PASSED: {checks} assertions across {sections_run} sections.\n"
        f"The verifier rejects clamp-to-zero signed encoding, 1-LSB weight "
        f"drift, truncated\n"
        f"memory images, and well-formed output-weight corruption against the "
        f"shipped-file\n"
        f"pin; malformed model JSON, nonnumeric fields, Q8.8-scale overflow "
        f"(1e308), and\n"
        f"invalid-UTF-8 .mem files exit through ParseError rather than a "
        f"traceback; an empty\n"
        f"or short result set FAILS while worst_residual_lsb() stays "
        f"crash-safe; a CRLF\n"
        f"checkout of the output-weight image still verifies; and the "
        f"shipped artifacts are\n"
        f"accepted.",
        file=stream,
    )
    return True
