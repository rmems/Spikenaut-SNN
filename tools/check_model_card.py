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
``2`` the file could not be read. An empty or short manifest is a hard failure,
never a pass -- reporting a clean card having checked nothing is worse than
crashing, because it gets believed.

``--self-test`` proves each claim can actually fail.
"""

from __future__ import annotations

import argparse
import io
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
README = REPO_ROOT / "README.md"

# Guards against a manifest silently emptied by a bad edit. Bump deliberately
# when a claim is added or retired.
EXPECTED_CLAIMS = 6


@dataclass(frozen=True)
class Claim:
    """One landed correction, and how to tell whether a copy carries it."""

    name: str
    why: str
    required: tuple[str, ...] = ()
    forbidden: tuple[str, ...] = ()


CLAIMS: tuple[Claim, ...] = (
    Claim(
        name="footprint",
        why=(
            "The four .mem artifacts hold 336 Q8.8 codes, which is 672 bytes. "
            "1.6 KB (1638) was never the size of anything shipped here."
        ),
        required=("672 bytes",),
    ),
    Claim(
        name="measured-input-channels",
        why=(
            "Every figure quoted across #2, #3, #4 and #13 was measured on five "
            "live GPU sensors. A cofire or Hamming number is unreadable without "
            "knowing it was five channels and not sixteen."
        ),
        required=(
            "mem_util_pct",
            "power_w",
            "gpu_temp_c",
            "sm_clock_mhz",
            "mem_clock_mhz",
        ),
    ),
    Claim(
        name="neuromod-is-not-a-dependency",
        why=(
            "Cargo.toml excludes neuromod on purpose. Listing it as a plain "
            "'crates.io dependency' contradicts the table's own Declared legend."
        ),
        forbidden=("| LIF engine, learning rules, neuromodulators | crates.io dependency |",),
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
            "artifact. Saying they are tool estimates is not the same as saying "
            "they cannot be checked."
        ),
        required=("no RTL",),
    ),
    Claim(
        name="no-uncited-external-power-figure",
        why=(
            "A single-LIF-neuron milliwatt range was once attributed to "
            "'published Artix-7 work' with no citation -- an uncheckable claim "
            "added by the edit that removed uncheckable claims."
        ),
        forbidden=("published Artix-7 work",),
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


def check(text: str, claims: tuple[Claim, ...] = CLAIMS) -> list[Failure]:
    """Every way `text` falls short of `claims`."""
    failures: list[Failure] = []
    for claim in claims:
        for needle in claim.required:
            if needle not in text:
                failures.append(Failure(claim.name, f"missing {needle!r}"))
        for needle in claim.forbidden:
            if needle in text:
                failures.append(Failure(claim.name, f"still carries {needle!r}"))
    return failures


def report(label: str, failures: list[Failure], checked: int) -> bool:
    """Print the verdict. Returns True when the copy is clean."""
    # An empty manifest would report every card clean, so refuse that outcome
    # rather than announce a verification that checked nothing.
    if checked < EXPECTED_CLAIMS:
        print(
            f"FAIL {label}: manifest has {checked} claims, expected at least "
            f"{EXPECTED_CLAIMS} -- refusing to report a clean card",
            file=sys.stderr,
        )
        return False

    if not failures:
        print(f"OK: {label} carries all {checked} landed corrections.")
        return True

    by_claim: dict[str, list[str]] = {}
    for failure in failures:
        by_claim.setdefault(failure.claim, []).append(failure.detail)

    print(f"FAIL {label}: {len(failures)} problem(s) across {len(by_claim)} claim(s).")
    for claim in CLAIMS:
        details = by_claim.get(claim.name)
        if not details:
            continue
        print(f"\n  [{claim.name}]")
        for detail in details:
            print(f"    - {detail}")
        print(f"    why: {claim.why}")
    return False


def self_test() -> bool:
    """Prove each claim can fail, and that the manifest guard bites."""
    good = read_text(README)
    if check(good):
        print("self-test: the repository README does not satisfy its own manifest", file=sys.stderr)
        return False

    checked = 0
    for claim in CLAIMS:
        # Break exactly this claim and require exactly it to be reported.
        broken = good
        for needle in claim.required:
            broken = broken.replace(needle, "")
        for needle in claim.forbidden:
            broken += f"\n{needle}\n"

        names = {failure.claim for failure in check(broken)}
        if names != {claim.name}:
            print(
                f"self-test: breaking {claim.name} reported {sorted(names)}",
                file=sys.stderr,
            )
            return False
        checked += 1

    if checked != len(CLAIMS):
        print("self-test: did not exercise every claim", file=sys.stderr)
        return False

    # And the short-manifest guard must refuse rather than pass. Its complaint
    # is the expected result here, so keep it off the console -- a self-test
    # that prints FAIL while succeeding trains readers to ignore the word.
    stderr, sys.stderr = sys.stderr, io.StringIO()
    stdout, sys.stdout = sys.stdout, io.StringIO()
    try:
        guard_passed = report("synthetic", [], checked=0)
    finally:
        sys.stderr, sys.stdout = stderr, stdout
    if guard_passed:
        print("self-test: an empty manifest reported clean", file=sys.stderr)
        return False

    print(f"self-test: {checked} claims each fail when broken; short manifest refused.")
    return True


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
    label = str(args.card)
    try:
        label = str(args.card.resolve().relative_to(REPO_ROOT))
    except ValueError:
        pass
    return 0 if report(label, check(text), len(CLAIMS)) else 1


if __name__ == "__main__":
    sys.exit(main())
