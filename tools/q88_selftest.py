"""Self-test for the Q8.8 verifier: prove the checker can actually fail.

A checker that cannot fail is worthless, and so is one that passes without
reading anything. Each numbered scenario builds a specific regression and
asserts ``q88_core`` rejects it: clamp-to-zero output weights (issue #15),
1-LSB hidden-weight drift, truncated files, well-formed output-weight
corruption against the shipped-file pin, malformed model JSON (a list/null top
level, a null neuron, a non-numeric threshold/decay/weight, or a huge-but-finite
``1e308``) and invalid-UTF-8 ``.mem`` files surfacing as ``ParseError`` rather
than a traceback / ``OverflowError`` / ``UnicodeDecodeError``, an empty or short
result set (which must FAIL while ``worst_residual_lsb`` stays crash-safe), and
a CRLF checkout of the output-weight image (which must still verify).

This module is the driver and the shared assertion helpers. The scenarios
themselves are grouped by what they establish:

``q88_value_scenarios``
    1-5 and 7 -- the encoded values, and the pins that anchor them.
``q88_contract_scenarios``
    6, 8 and 9 -- what the verifier promises when it cannot check something.
``q88_encoding_scenarios``
    10 and 11 -- bytes, line endings and console encodings.

Scratch files are written to a temp directory; nothing here ever writes into
``dataset/``.

Run via the CLI::

    python3 tools/verify_q88.py --self-test
"""

from __future__ import annotations

import sys
import tempfile
from collections.abc import Callable
from pathlib import Path

try:  # package import: `python3 -m tools.verify_q88`
    from .q88_core import SelfTestFailure
except ImportError:  # direct script: `python3 tools/verify_q88.py`
    from q88_core import SelfTestFailure


def _require(condition: bool, message: str) -> None:
    """Assert a self-test invariant, and count the assertion.

    ``_require.calls`` counts every call since self_test() last reset it.
    Each scenario reports its own assertion count as a hand-maintained
    ``checks += N`` literal, and those have drifted; the invariant at the end
    of self_test() uses this counter to prove the reported total is the number
    of assertions actually run.

    Raises:
        SelfTestFailure: ``condition`` is false. Used instead of bare
            ``assert`` so the checks survive ``python -O``.

    """
    _require.calls += 1
    if not condition:
        raise SelfTestFailure(message)


_require.calls = 0


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

# The two halves of the assertion total the scenarios report between them:
# _require calls, and assertions made by a try/except/else proving a call
# raised. Asserted by _suite_is_fully_accounted_for().
EXPECTED_REQUIRE_CALLS = 67
EXPECTED_GUARD_CHECKS = 31


def _scenarios() -> tuple[tuple[Callable[..., int], ...], tuple[Callable[..., int], ...]]:
    """The numbered scenarios, in the order self_test() runs and prints them.

    These tuples ARE the suite, so the count is asserted against
    SELF_TEST_SECTIONS rather than derived from len(): deriving it would make a
    dropped scenario self-consistent and therefore invisible.

    Imported here rather than at module scope because every scenario module
    imports ``_require`` and ``_write_mem`` from this one. Whole modules rather
    than names, so this stays one shim instead of eighteen repeated imports.
    """
    try:  # package import: `python3 -m tools.verify_q88`
        from . import q88_contract_scenarios as contract
        from . import q88_encoding_scenarios as encoding
        from . import q88_value_scenarios as value
    except ImportError:  # direct script: `python3 tools/verify_q88.py`
        import q88_contract_scenarios as contract
        import q88_encoding_scenarios as encoding
        import q88_value_scenarios as value

    standalone = (
        value.codec_unit_vectors,
        value.shipped_artifacts_verify,
    )
    tempdir = (
        value.clamp_to_zero_rejected,
        value.corrupted_hidden_weight_rejected,
        value.truncated_file_rejected,
        contract.empty_result_set_fails,
        value.wellformed_corruption_rejected,
        contract.malformed_model_json_parse_error,
        contract.nonnumeric_fields_parse_error,
        encoding.crlf_checkout_still_verifies,
        encoding.invalid_utf8_parse_error,
    )
    return standalone, tempdir


def _suite_is_fully_accounted_for(sections_run, scenario_checks) -> int:
    """The suite ran every section, and reported every assertion it ran.

    Two independent books have to agree. Sections are counted against
    SELF_TEST_SECTIONS so a dropped scenario cannot shrink the suite quietly.
    Assertions are counted twice over: _require keeps its own tally, which no
    edit to a ``checks += N`` literal can fake, and the remainder must be the
    hand-counted try/except guards. Both halves are pinned because both have
    been wrong -- _shipped_file_matches_its_pins() ran four assertions and
    returned zero, and three scenarios were each short by one.
    """
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
        f"without updating its scenario's count",
    )
    _require(
        scenario_checks - scenario_requires == EXPECTED_GUARD_CHECKS,
        f"scenarios reported {scenario_checks} assertions against "
        f"{scenario_requires} _require calls, leaving "
        f"{scenario_checks - scenario_requires} try/except guards, expected "
        f"{EXPECTED_GUARD_CHECKS} -- a reported count does not match the "
        f"assertions actually run",
    )
    return 3


def self_test(stream=sys.stdout) -> bool:
    """Prove this checker actually rejects real regressions."""
    print("Q8.8 verifier self-test", file=stream)
    print("", file=stream)
    _require.calls = 0
    checks = 0
    sections_run = 0

    standalone, tempdir_scenarios = _scenarios()
    for scenario in standalone:
        checks += scenario(stream)
        sections_run += 1

    with tempfile.TemporaryDirectory(prefix="q88-selftest-") as tmpdir:
        tmp = Path(tmpdir)
        for scenario in tempdir_scenarios:
            checks += scenario(tmp, stream)
            sections_run += 1

    checks += _suite_is_fully_accounted_for(sections_run, checks)

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
