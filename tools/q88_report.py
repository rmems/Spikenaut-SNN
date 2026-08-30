"""Rendering the Q8.8 verification verdict: per-section blocks and the summary.

Kept apart from ``q88_core`` because deciding whether the artifacts verify and
describing that decision to a reader are different jobs. Nothing here changes a
verdict; ``report`` only refuses to *print* a clean one for a run that checked
nothing.
"""

from __future__ import annotations

import sys

try:  # package import: `python3 -m tools.verify_q88`
    from .q88_core import DATA_DIR, REPO_ROOT, SectionResult
except ImportError:  # direct script: `python3 tools/verify_q88.py`
    from q88_core import DATA_DIR, REPO_ROOT, SectionResult


def worst_residual_lsb(results: list[SectionResult]) -> float:
    """Max residual across sections that have one; 0.0 if none do.

    An empty residual list must not raise -- ``report()`` is called with
    signed-only results (``max_residual_lsb`` is None) and in tests with
    no sections at all. ``max()`` on an empty generator is a ValueError.
    """
    residuals = [
        r.evidence.max_residual_lsb
        for r in results
        if r.evidence.max_residual_lsb is not None
    ]
    return max(residuals) if residuals else 0.0


def _stream_safe(text: str, stream) -> str:
    """Render `text` so writing it to `stream` cannot raise.

    Everything this tool emits is ASCII by construction except the checkout
    path, which the user chooses. On a non-UTF-8 console (a Windows cp1252
    process, say) a path holding an unencodable character made a *successful*
    verification die with UnicodeEncodeError and exit 1. The path is
    diagnostic, so escaping the offending characters beats crashing on them.
    """
    encoding = getattr(stream, "encoding", None) or "utf-8"
    try:
        text.encode(encoding)
    except (UnicodeEncodeError, LookupError):
        return text.encode(encoding, "backslashreplace").decode(encoding)
    return text


def _artifact_coverage_failures(
    results: list[SectionResult], expected_artifacts: tuple[str, ...] | None
) -> list[str]:
    """Each expected artifact must be covered by exactly one section.

    Counting sections is not enough: duplicating one entry while dropping
    another keeps the arity right and still leaves an artifact unread.
    """
    if expected_artifacts is None:
        return []
    # Which artifact each section actually read, taken from its label.
    seen = [
        artifact
        for artifact in expected_artifacts
        for r in results
        if artifact in r.name
    ]
    return [
        f"INCOMPLETE RUN: {artifact} was checked {seen.count(artifact)} "
        f"time(s), expected exactly 1. A duplicated or missing "
        f"section keeps the count right while leaving an "
        f"artifact unverified."
        for artifact in expected_artifacts
        if seen.count(artifact) != 1
    ]


def _structural_failures(
    results: list[SectionResult],
    expected_sections: int | None,
    expected_artifacts: tuple[str, ...] | None,
) -> list[str]:
    """Reasons the run itself did not happen, regardless of per-section verdicts.

    Separate from "did the values match": four individually clean sections can
    still be the wrong four, and zero sections is not a pass.
    """
    if not results:
        return [
            "NOTHING WAS CHECKED: 0 sections. all([]) is True, so an unguarded "
            "verdict would report a clean pass having read no artifact."
        ]

    structural: list[str] = []
    if expected_sections is not None and len(results) != expected_sections:
        structural.append(
            f"INCOMPLETE RUN: {len(results)} section(s), expected "
            f"{expected_sections}. A section was dropped, so this verdict "
            f"would cover less than the artifact set it claims."
        )
    structural.extend(_artifact_coverage_failures(results, expected_artifacts))
    if sum(r.count for r in results) == 0:
        structural.append(
            "NOTHING WAS CHECKED: sections were present but held 0 values."
        )
    return structural


def _render_pass(results: list[SectionResult]) -> list[str]:
    """The clean verdict, counting only checks that actually ran."""
    cross_checked = sum(
        r.count for r in results if r.evidence.max_residual_lsb is not None
    )
    pinned = sum(r.evidence.pinned_count for r in results)
    lines = [
        f"OK: {sum(r.count for r in results)} Q8.8 values verified, 0 mismatches."
    ]
    if cross_checked:
        lines.append(
            f"    {cross_checked} of them cross-validated value-by-value "
            f"against snn_model.json floats"
        )
        lines.append(
            f"    (worst |json - mem| residual {worst_residual_lsb(results):.6f} LSB)."
        )
    if pinned:
        lines.append(
            f"    {pinned} signed output weight(s) pinned to the "
            f"shipped-file canonical-token sha256"
        )
        lines.append(
            "    / gold hex words, plus sign-integrity and canonical "
            "round-trip (defense in depth)."
        )
    lines.append(
        "    This is the float->Q8.8 encoding half of issue #4; it is not "
        "a close of #4."
    )
    lines.append(
        "    Hamming-on-holdout is the close bar, not weight MSE / "
        "bit-identical export."
    )
    return lines


def _render_failure(
    results: list[SectionResult], structural: list[str]
) -> list[str]:
    """The failure verdict: structural reasons first, then the bad sections."""
    lines = ["FAILED:"]
    lines.extend(f"        {reason}" for reason in structural)
    failed = [r.name.split("(")[0].strip() for r in results if not r.ok]
    if failed:
        total_mismatches = sum(len(r.mismatches) for r in results)
        detail = (
            f"{total_mismatches} value mismatch(es)"
            if total_mismatches
            else "no value mismatches, but a section-level invariant broke "
            "(see notes above)"
        )
        lines.append(
            f"        {len(failed)} of {len(results)} section(s) did not "
            f"verify -- {', '.join(failed)}."
        )
        lines.append(f"        {detail}.")
    return lines


def report(
    results: list[SectionResult],
    stream=sys.stdout,
    expected_sections: int | None = None,
    expected_artifacts: tuple[str, ...] | None = None,
) -> bool:
    """Print the full per-section report and the verdict; return True if clean.

    An empty ``results`` list is a HARD FAILURE, never a pass. ``all([])`` is
    True, so an unguarded verdict printed "OK: 0 Q8.8 values verified" -- plus
    the claim that 48 output weights were pinned -- having read no artifact at
    all. A verifier that reports a clean pass having checked nothing is worse
    than one that crashes, because it is believed. Pass ``expected_sections``
    to reject a short run too, where a section was dropped upstream.

    The summary reports only what was actually verified: the cross-validated
    count, the worst residual and the pinned-word count are all counted out of
    ``results``, so no line can assert a check that did not run.

    ``worst_residual_lsb`` stays empty-safe (a section may legitimately carry
    no residual); the empty-set verdict is decided here, not there.
    """
    print("Q8.8 export verification", file=stream)
    print(f"repo root: {_stream_safe(str(REPO_ROOT), stream)}", file=stream)
    print(
        f"artifacts: {_stream_safe(str(DATA_DIR.relative_to(REPO_ROOT)), stream)}/",
        file=stream,
    )
    print("", file=stream)
    for result in results:
        print(result.render(), file=stream)
    print("", file=stream)

    structural = _structural_failures(results, expected_sections, expected_artifacts)
    ok = not structural and all(r.ok for r in results)
    lines = _render_pass(results) if ok else _render_failure(results, structural)
    print("\n".join(lines), file=stream)
    return ok
