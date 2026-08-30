"""Scenarios 1-5 and 7: the encoded values, and the pins that anchor them.

Each builds a specific corruption of the shipped artifacts and asserts
``q88_core`` rejects it -- clamp-to-zero signed encoding (the issue #15
hazard), 1-LSB hidden-weight drift, a truncated image, and a well-formed
output-weight change that only the shipped-file pin can catch.

Scenario 6 (an empty or short result set) and 8-9 (the ``ParseError``
contract) live in ``q88_contract_scenarios``; 10-11 (line endings and
encodings) live in ``q88_encoding_scenarios``.
"""

from __future__ import annotations

import io
from pathlib import Path

try:  # package import: `python3 -m tools.verify_q88`
    from .q88_selftest import _require, _write_mem
except ImportError:  # direct script: `python3 tools/verify_q88.py`
    from q88_selftest import _require, _write_mem

try:  # package import: `python3 -m tools.verify_q88`
    from .q88_core import (
        EXPECTED_ARTIFACTS,
        EXPECTED_SECTIONS,
        MEM_HIDDEN,
        MEM_OUTPUT,
        MODEL_JSON,
        N_OUTPUT_WEIGHTS,
        OUTPUT_WEIGHTS_CANON_SHA256,
        OUTPUT_WEIGHTS_HEX,
        ShippedPin,
        canonical_mem_digest,
        check_against_floats,
        check_signed_section,
        decode_q88,
        encode_q88_clamp_to_zero,
        encode_q88_hex,
        load_model,
        parse_mem,
        verify_shipped,
    )
    from .q88_report import report
except ImportError:  # direct script: `python3 tools/verify_q88.py`
    from q88_core import (
        EXPECTED_ARTIFACTS,
        EXPECTED_SECTIONS,
        MEM_HIDDEN,
        MEM_OUTPUT,
        MODEL_JSON,
        N_OUTPUT_WEIGHTS,
        OUTPUT_WEIGHTS_CANON_SHA256,
        OUTPUT_WEIGHTS_HEX,
        ShippedPin,
        canonical_mem_digest,
        check_against_floats,
        check_signed_section,
        decode_q88,
        encode_q88_clamp_to_zero,
        encode_q88_hex,
        load_model,
        parse_mem,
        verify_shipped,
    )
    from q88_report import report


def codec_unit_vectors(stream) -> int:
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


def shipped_artifacts_verify(stream) -> int:
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


def _clamp_rejected_by_sign_invariants(clamped, stream) -> int:
    """The signed-section check rejects the clamped file with no float reference."""
    checks = 0
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
    return checks


def _clamp_flagged_against_floats(real_floats, clamped, n_negative, stream) -> int:
    """Against the true float vector, every inhibitory weight must be flagged."""
    checks = 0
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
    return checks


def clamp_to_zero_rejected(tmp: Path, stream) -> int:
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
    checks += 1

    clamped_path = tmp / "parameters_output_weights_clamped.mem"
    _write_mem(clamped_path, [encode_q88_clamp_to_zero(v) for v in real_floats])
    clamped = parse_mem(clamped_path)

    checks += _clamp_flagged_against_floats(real_floats, clamped, n_negative, stream)

    checks += _clamp_rejected_by_sign_invariants(clamped, stream)
    print("", file=stream)
    return checks


def corrupted_hidden_weight_rejected(tmp: Path, stream) -> int:
    """Reject a single 1-LSB drift in the JSON-backed hidden-weight section."""
    checks = 0
    print("4. a single corrupted hidden weight is REJECTED", file=stream)
    hidden_entries = parse_mem(MEM_HIDDEN)
    model = load_model(MODEL_JSON)
    hidden_floats = [
        float(w) for neuron in model["neurons"] for w in neuron["weights"]
    ]
    corrupt_words = [e.word for e in hidden_entries]
    # Index 137 is safe unguarded because scenario 2 runs first and asserts the
    # shipped artifacts verify, which includes their arity -- a short
    # parameters_weights.mem fails there, before this line is reached. Keep
    # shipped_artifacts_verify ahead of this scenario in _scenarios().
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


def truncated_file_rejected(tmp: Path, stream) -> int:
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


def _gold_pin_rejects_zeroed_image(tmp, stream, gold) -> int:
    """47 zeros plus FFFF is well-formed and signed; only the gold pin catches it."""
    checks = 0

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
    return checks


def _shipped_file_matches_its_pins(gold, real) -> int:
    """The shipped file must match both pins before it can serve as the gold reference."""
    checks = 0
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
    checks += 4
    return checks


def _gold_pin_rejects_swapped_word(tmp, stream, real, gold) -> int:
    """An FFF9->FFF8 swap round-trips cleanly; only the gold pin catches it."""
    checks = 0
    print(
        "7. well-formed output-weight corruption is REJECTED (shipped-file pin)",
        file=stream,
    )
    checks += _shipped_file_matches_its_pins(gold, real)

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
    checks += 3
    return checks


def wellformed_corruption_rejected(tmp: Path, stream) -> int:
    """Reject well-formed output-weight corruption that still round-trips.

    An FFF9->FFF8 swap decodes and re-encodes cleanly, so only the
    shipped-file pin catches it.
    """
    real = parse_mem(MEM_OUTPUT)
    gold = tuple(e.text.upper() for e in real)
    checks = 0
    #
    # Sign-integrity + decode->re-encode accept any well-formed 48-word
    # file with at least one high-bit word. The gold pin is what makes
    # FFF9->FFF8 and "47 zeros + FFFF" fail.
    checks += _gold_pin_rejects_swapped_word(tmp, stream, real, gold)
    print("   ok: FFF9->FFF8 rejected by gold pin, accepted without it", file=stream)
    checks += _gold_pin_rejects_zeroed_image(tmp, stream, gold)
    print("", file=stream)
    return checks
