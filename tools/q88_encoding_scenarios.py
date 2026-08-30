"""Scenarios 10 and 11: bytes, line endings and console encodings.

The artifacts are text files read on machines that disagree about what text
is. A CRLF checkout must still verify against the pins; a ``.mem`` using
non-ASCII separators or padding must be REJECTED, because ``$readmemh`` does
not read them as whitespace; invalid UTF-8 and an unreadable artifact must
surface as ``ParseError``; and the verdict must encode cleanly to a cp1252
console.
"""

from __future__ import annotations

import hashlib
import io
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
        N_OUTPUT_WEIGHTS,
        OUTPUT_WEIGHTS_CANON_SHA256,
        OUTPUT_WEIGHTS_HEX,
        ParseError,
        SectionResult,
        SelfTestFailure,
        ShippedPin,
        canonical_mem_digest,
        check_signed_section,
        parse_mem,
        read_utf8_text,
        verify_shipped,
    )
    from .q88_report import report
except ImportError:  # direct script: `python3 tools/verify_q88.py`
    from q88_core import (
        EXPECTED_ARTIFACTS,
        EXPECTED_SECTIONS,
        MEM_OUTPUT,
        N_OUTPUT_WEIGHTS,
        OUTPUT_WEIGHTS_CANON_SHA256,
        OUTPUT_WEIGHTS_HEX,
        ParseError,
        SectionResult,
        SelfTestFailure,
        ShippedPin,
        canonical_mem_digest,
        check_signed_section,
        parse_mem,
        read_utf8_text,
        verify_shipped,
    )
    from q88_report import report


# Modules the ASCII scan must find in this package: the four originals plus the
# three scenario modules and __init__. A floor, not an equality, so adding a
# module does not fail the suite -- but a glob that silently finds nothing does.
EXPECTED_PACKAGE_MODULES = 8


def _canonicalisation_ignores_case_and_padding(tmp, real) -> int:
    """Lower-case words with padding are the same artifact; canonicalization must say so."""
    checks = 0
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
    return checks


def _crlf_and_lf_fixtures_agree(tmp) -> int:
    """LF and CRLF forms of the same image must both verify against the pin."""
    checks = 0
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
    checks += 6
    return checks


def crlf_checkout_still_verifies(tmp: Path, stream) -> int:
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
    checks += _crlf_and_lf_fixtures_agree(tmp)

    checks += _canonicalisation_ignores_case_and_padding(tmp, real)
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


def _output_survives_cp1252(shipped) -> int:
    """Emitted text must survive a non-UTF-8 console."""
    checks = 0
    # Emitted text must survive a non-UTF-8 console. A Windows process
    # under cp1252 raised UnicodeEncodeError on the arrows in the summary,
    # so a *successful* verification exited 1 with a traceback.
    # Every module in the package, discovered rather than listed: a hand-kept
    # list already missed one module once, and splitting the suite into more
    # of them would only make that easier.
    modules = sorted(Path(__file__).parent.glob("*.py"))
    _require(
        len(modules) >= EXPECTED_PACKAGE_MODULES,
        f"scanned {len(modules)} modules, expected at least "
        f"{EXPECTED_PACKAGE_MODULES} -- the glob is not finding the package",
    )
    offenders = {}
    for module in modules:
        found = sorted({c for c in module.read_text(encoding="utf-8") if ord(c) > 127})
        if found:
            offenders[module.name] = found
    _require(
        not offenders,
        f"non-ASCII in {sorted(offenders)} may reach stdout: {offenders}",
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
    checks += 3
    return checks


def _invalid_utf8_raises_parse_error(tmp) -> int:
    """Both readers must surface invalid UTF-8 as ParseError, never UnicodeDecodeError."""
    checks = 0
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
    return checks


def invalid_utf8_parse_error(tmp: Path, stream) -> int:
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
    checks += _invalid_utf8_raises_parse_error(tmp)

    checks += _reject_substituted_sections(shipped)
    print(
        "   ok: a duplicated section FAILS on artifact identity "
        "(arity alone accepts it)",
        file=stream,
    )

    checks += _output_survives_cp1252(shipped)
    print(
        "   ok: output is ASCII and encodes cleanly to a cp1252 console",
        file=stream,
    )

    checks += _reject_unreadable_artifacts(tmp)
    checks += _reject_non_ascii_mem_syntax(tmp)
    print("   ok: invalid UTF-8 .mem raises ParseError, not UnicodeDecodeError", file=stream)
    print("   ok: non-ASCII separators/padding rejected; LF, CRLF, space/tab accepted", file=stream)
    print("   ok: an unreadable artifact raises ParseError, not OSError", file=stream)
    print("", file=stream)
    return checks


def _reject_non_ascii_mem_syntax(tmp: Path) -> int:
    """A .mem using non-ASCII separators or padding must be REJECTED.

    str.splitlines() breaks on U+2028/U+2029/U+0085 and str.strip() removes
    NBSP, so an image built with those parses into the right words and matches
    the pin -- while $readmemh does not read them as whitespace. Accepting one
    would bless a malformed FPGA artifact.
    """
    checks = 0
    words = [e.text for e in parse_mem(MEM_OUTPUT)]

    for label, text in (
        ("U+2028 line separator", "\u2028".join(words) + "\n"),
        ("U+0085 next line", "\u0085".join(words) + "\n"),
        ("NBSP padding", "\n".join("\u00a0" + w + "\u00a0" for w in words) + "\n"),
    ):
        path = tmp / f"mem_{label.split()[0].strip('+U')}.mem"
        path.write_text(text, encoding="utf-8")
        try:
            parse_mem(path)
        except ParseError:
            checks += 1
        else:
            raise SelfTestFailure(
                f"parse_mem ACCEPTED a .mem using {label}; $readmemh would not"
            )

    # Positive controls: the ASCII syntax the hardware reader does accept must
    # still parse, or the guard is just breaking the tool.
    for label, text in (
        ("LF", "\n".join(words) + "\n"),
        ("CRLF", "\r\n".join(words) + "\r\n"),
        ("space and tab padding", "\n".join(" \t" + w + "\t " for w in words) + "\n"),
    ):
        path = tmp / f"mem_ok_{label.split()[0]}.mem"
        path.write_text(text, encoding="utf-8")
        _require(
            len(parse_mem(path)) == N_OUTPUT_WEIGHTS,
            f"parse_mem REJECTED a valid {label} image",
        )
        checks += 1
    return checks
