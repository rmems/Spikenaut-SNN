#!/usr/bin/env python3
"""Hold every published copy of the model card to the corrections already landed.

This model is published twice: ``README.md`` in the GitHub repository, and the
Hugging Face model card at ``rmems/Spikenaut-SNN``. The card is the copy most
readers meet first, and the two have drifted -- when this tool was written the
card was missing every correction from #32 and #33, so it still stated a
1.6 KB footprint against a 672-byte artifact, still linked a repository that
had moved, and still named no measured input channels.

That is worse than either copy being wrong on its own. A correction that lands
in one place and not the other leaves the project asserting both the claim and
its retraction, and the reader has no way to know which they are holding.

What this checks
----------------
A claim is not prose to be diffed. Two copies can word the same fact
differently and both be right, and a line-by-line diff of these files is mostly
noise -- the card carries front-matter tags, an architecture image and a
loading section the repository copy has no reason to. So this does not diff.

It asserts a *manifest*: for each correction that has landed, the wording that
must be present and the stale wording that must be gone. A copy satisfying the
manifest may say anything else it likes.

Usage
-----
``check_model_card.py`` checks the repository's own ``README.md``.

``check_model_card.py --card path/to/card.md`` checks another copy. To check
what is actually published::

    git fetch hf && git show hf/main:README.md > /tmp/card.md
    python3 tools/check_model_card.py --card /tmp/card.md

Exit codes match the rest of ``tools/``: ``0`` verified, ``1`` a claim failed,
``2`` the file could not be read. A manifest whose length is not exactly
``EXPECTED_CLAIMS`` is a hard failure, never a pass -- reporting a clean card
having checked the wrong set is worse than crashing, because it gets believed.

``--self-test`` proves each claim can actually fail.
"""

from __future__ import annotations

import argparse
import io
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
README = REPO_ROOT / "README.md"

# Guards against a manifest silently emptied or padded by a bad edit. Bump
# deliberately when a claim is added or retired.
EXPECTED_CLAIMS = 7

# Historical 1.6 KB is allowed only on a line that also carries this
# annotation. The spec-table row is forbidden even when annotated.
FOOTPRINT_PROVENANCE_NOTE = "disagrees with the shipped artifact"
STALE_FOOTPRINT_SPEC = "| Memory footprint | 1.6 KB"
STALE_FOOTPRINT_FIGURE = "1.6 KB"
STALE_INPUT_CHANNELS_ROW = "| Input channels | 16 |"


@dataclass(frozen=True)
class Claim:
    """One landed correction, and how to tell whether a copy carries it."""

    name: str
    why: str
    required: tuple[str, ...] = ()
    forbidden: tuple[str, ...] = ()
    # (stale, annotation): stale may appear only on a line that also has annotation.
    forbidden_unless_same_line: tuple[tuple[str, str], ...] = ()


CLAIMS: tuple[Claim, ...] = (
    Claim(
        name="footprint",
        why=(
            "The four .mem artifacts hold 336 Q8.8 codes, which is 672 bytes. "
            "1.6 KB (1638) was never the size of anything shipped here. The "
            "historical Memory-usage row may keep 1.6 KB only when it says "
            "that figure disagrees with the shipped artifact."
        ),
        required=("672 bytes",),
        forbidden=(STALE_FOOTPRINT_SPEC,),
        forbidden_unless_same_line=((STALE_FOOTPRINT_FIGURE, FOOTPRINT_PROVENANCE_NOTE),),
    ),
    Claim(
        name="measured-input-channels",
        why=(
            "Every figure quoted across #2, #3, #4 and #13 was measured on five "
            "live GPU sensors. A cofire or Hamming number is unreadable without "
            "knowing it was five channels and not sixteen. The unqualified "
            "16-channel specification row is the claim the correction removed."
        ),
        required=(
            "mem_util_pct",
            "power_w",
            "gpu_temp_c",
            "sm_clock_mhz",
            "mem_clock_mhz",
        ),
        forbidden=(STALE_INPUT_CHANNELS_ROW,),
    ),
    Claim(
        name="neuromod-is-a-declared-dependency",
        why=(
            "Cargo.toml declares neuromod from crates.io. The old "
            "'Published, but not a dependency' wording, or a plain "
            "'crates.io dependency' without the Declared marker, would "
            "contradict the manifest."
        ),
        forbidden=(
            "Published, but **not** a dependency",
            "| LIF engine, learning rules, neuromodulators | crates.io dependency |",
        ),
    ),
    Claim(
        name="moved-repositories",
        why=(
            "silicon-bridge and silicon-hdl moved to the rmems owner; the "
            "Limen-Neural paths only 301-redirect."
        ),
        forbidden=(
            "Limen-Neural/silicon-bridge",
            "Limen-Neural/silicon-hdl",
        ),
    ),
    Claim(
        name="hardware-figures-unreproducible",
        why=(
            "No RTL, constraints, Vivado project or report is checked in, so a "
            "reader cannot regenerate the power and utilisation table from this "
            "artifact. Saying they are tool estimates, or mentioning 'no RTL' "
            "in the architecture introduction, is not the landed disclosure "
            "that every reported figure is unreproducible."
        ),
        required=(
            "unreproducible",
            "cannot regenerate",
        ),
    ),
    Claim(
        name="no-uncited-external-power-figure",
        why=(
            "A single-LIF-neuron milliwatt range was once attributed to "
            "'published Artix-7 work' with no citation -- an uncheckable claim "
            "added by the edit that removed uncheckable claims. The range "
            "itself is the unsupported figure, not only its attribution."
        ),
        forbidden=(
            "published Artix-7 work",
            "85-95 mW",
        ),
    ),
    Claim(
        name="eprop-ottt-identity-unassessed",
        why=(
            "The card once treated identical E-prop/OTTT gradients as an "
            "established cause of the weight ramp. No learning rule is "
            "implemented here, so that identity is unassessed -- a copy that "
            "omits the qualification still asserts a conclusion this artifact "
            "cannot support."
        ),
        required=(
            "unassessed",
            "neither confirm as a check that",
        ),
    ),
)


@dataclass(frozen=True)
class Failure:
    """One way a copy fell short of the manifest."""

    claim: str
    detail: str


def read_text(path: Path) -> str:
    """Read a card, or exit 2 naming what could not be read."""
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        print(f"could not read {path}: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


def _fold_dashes(text: str) -> str:
    """Treat en/em dashes as ASCII hyphens so a card cannot hide 85-95 mW."""
    return text.replace("\u2013", "-").replace("\u2014", "-")


def _required_failures(text: str, claim: Claim) -> Iterator[Failure]:
    for needle in claim.required:
        if needle not in text:
            yield Failure(claim.name, f"missing {needle!r}")


def _forbidden_failures(text: str, claim: Claim) -> Iterator[Failure]:
    for needle in claim.forbidden:
        if needle in text:
            yield Failure(claim.name, f"still carries {needle!r}")


def _unannotated_failures(text: str, claim: Claim) -> Iterator[Failure]:
    for stale, annotation in claim.forbidden_unless_same_line:
        for line in text.splitlines():
            if stale in line and annotation not in line:
                yield Failure(claim.name, f"still carries unannotated {stale!r}")
                break


def check(text: str, claims: tuple[Claim, ...] = CLAIMS) -> list[Failure]:
    """Every way `text` falls short of `claims`."""
    text = _fold_dashes(text)
    failures: list[Failure] = []
    for claim in claims:
        failures.extend(_required_failures(text, claim))
        failures.extend(_forbidden_failures(text, claim))
        failures.extend(_unannotated_failures(text, claim))
    return failures


def _manifest_count_matches(checked: int) -> bool:
    return checked == EXPECTED_CLAIMS


def _print_by_claim(failures: list[Failure]) -> None:
    by_claim: dict[str, list[str]] = {}
    for failure in failures:
        by_claim.setdefault(failure.claim, []).append(failure.detail)
    for claim in CLAIMS:
        details = by_claim.get(claim.name)
        if not details:
            continue
        print(f"\n  [{claim.name}]")
        for detail in details:
            print(f"    - {detail}")
        print(f"    why: {claim.why}")


def report(label: str, failures: list[Failure], checked: int) -> bool:
    """Print the verdict. Returns True when the copy is clean."""
    # A drifted manifest would report the wrong set of corrections, so refuse
    # that outcome rather than announce a verification that checked the
    # wrong thing.
    if not _manifest_count_matches(checked):
        print(
            f"FAIL {label}: manifest has {checked} claims, expected exactly "
            f"{EXPECTED_CLAIMS} -- refusing to report a clean card",
            file=sys.stderr,
        )
        return False

    if not failures:
        print(f"OK: {label} carries all {checked} landed corrections.")
        return True

    print(f"FAIL {label}: {len(failures)} problem(s) across {len({f.claim for f in failures})} claim(s).")
    _print_by_claim(failures)
    return False


def _strip_required(text: str, claim: Claim) -> str:
    broken = text
    for needle in claim.required:
        broken = broken.replace(needle, "")
    return broken


def _inject_forbidden(text: str, claim: Claim) -> str:
    extra = list(claim.forbidden)
    already = "\n".join(extra)
    for stale, _annotation in claim.forbidden_unless_same_line:
        if stale not in already:
            extra.append(stale)
    if not extra:
        return text
    return text + "".join(f"\n{needle}\n" for needle in extra)


def _isolation_variants(good: str, claim: Claim) -> list[str]:
    variants: list[str] = []
    if claim.required:
        variants.append(_strip_required(good, claim))
    if claim.forbidden or claim.forbidden_unless_same_line:
        variants.append(_inject_forbidden(good, claim))
    return variants


def _exercise_claim(good: str, claim: Claim) -> bool:
    """True when each isolation variant reports exactly this claim."""
    variants = _isolation_variants(good, claim)
    if not variants:
        print(f"self-test: {claim.name} has nothing to break", file=sys.stderr)
        return False
    for broken in variants:
        names = {failure.claim for failure in check(broken)}
        if names != {claim.name}:
            print(
                f"self-test: breaking {claim.name} reported {sorted(names)}",
                file=sys.stderr,
            )
            return False
    return True


def _with_stdio_silenced(fn):
    """Run `fn` with stdout/stderr discarded. Used by the manifest-count guard."""
    sink = io.StringIO()
    stderr, sys.stderr = sys.stderr, sink
    stdout, sys.stdout = sys.stdout, sink
    try:
        return fn()
    finally:
        sys.stderr, sys.stdout = stderr, stdout


def _guard_refuses(checked: int) -> bool:
    """True when report() refuses this count on an otherwise clean card."""
    return not _with_stdio_silenced(lambda: report("synthetic", [], checked=checked))


def self_test() -> bool:
    """Prove each claim can fail, and that the exact-count guard bites."""
    good = read_text(README)
    if check(good):
        print("self-test: the repository README does not satisfy its own manifest", file=sys.stderr)
        return False

    checked = 0
    for claim in CLAIMS:
        if not _exercise_claim(good, claim):
            return False
        checked += 1

    if checked != len(CLAIMS):
        print("self-test: did not exercise every claim", file=sys.stderr)
        return False

    # Wrong counts must refuse rather than pass. Their complaint is the
    # expected result here, so keep it off the console -- a self-test that
    # prints FAIL while succeeding trains readers to ignore the word.
    for wrong in (0, EXPECTED_CLAIMS - 1, EXPECTED_CLAIMS + 1):
        if not _guard_refuses(wrong):
            print(
                f"self-test: manifest count {wrong} reported clean "
                f"(expected exactly {EXPECTED_CLAIMS})",
                file=sys.stderr,
            )
            return False

    print(
        f"self-test: {checked} claims each fail when broken; "
        "exact manifest count enforced."
    )
    return True


def _card_label(path: Path) -> str:
    label = str(path)
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return label


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--card",
        type=Path,
        default=README,
        help="the copy to check (default: this repository's README.md)",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="prove each claim can fail, then exit",
    )
    args = parser.parse_args(argv)

    if args.self_test:
        return 0 if self_test() else 1

    text = read_text(args.card)
    return 0 if report(_card_label(args.card), check(text), len(CLAIMS)) else 1


if __name__ == "__main__":
    sys.exit(main())
