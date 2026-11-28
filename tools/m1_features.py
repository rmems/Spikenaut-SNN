"""Emit paired M1 feature rows in sensor units: python -m tools.m1_features."""

import argparse
from collections import deque
from dataclasses import dataclass
import json
import math
import sys

if __package__:
    from .hamming_const import LIVE_COLUMNS
    from .hamming_encode import Sample
    from .m1_arm_plan import SMA_SOURCES, SMA_WINDOW, row_eligibility
    from .m1_cohort import Cohort, add_cohort_args, cohort_from_args
    from .q88_core import ParseError
else:
    from hamming_const import LIVE_COLUMNS
    from hamming_encode import Sample
    from m1_arm_plan import SMA_SOURCES, SMA_WINDOW, row_eligibility
    from m1_cohort import Cohort, add_cohort_args, cohort_from_args
    from q88_core import ParseError

SMA_INDICES = tuple(LIVE_COLUMNS.index(name) for name in SMA_SOURCES)


@dataclass(frozen=True, slots=True)
class PairedRow:
    split: str
    sample: Sample
    raw: tuple[float, ...]
    kinetic: tuple[float, ...]

    def arms(self):
        return {"RAW": self.raw, "KINETIC": self.kinetic, "HYBRID": self.raw + self.kinetic}


def update_windows(windows, values, missing):
    for name, index, window in zip(SMA_SOURCES, SMA_INDICES, windows, strict=True):
        if name in missing:
            window.clear()
        else:
            window.append(values[index])


def paired_rows(cohort: Cohort):
    """Yield exactly the common rows defined by the arm plan, in source order."""
    last_episode = None
    for sample, values, eligibility in zip(
        cohort.samples, cohort.measurements, row_eligibility(cohort), strict=True
    ):
        split, episode, _, raw_ready, kinetic_ready = eligibility
        if episode != last_episode:
            windows = [deque(maxlen=SMA_WINDOW) for _ in SMA_SOURCES]
            last_episode = episode
        update_windows(windows, values, sample.missing)
        if raw_ready and kinetic_ready:
            yield PairedRow(split, sample, values, tuple(math.fsum(w) / SMA_WINDOW for w in windows))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_cohort_args(parser)
    args = parser.parse_args(argv)
    try:
        cohort = cohort_from_args(args)
        for row in paired_rows(cohort):
            print(json.dumps({
                "split": row.split, "episode_id": row.sample.episode_id,
                "source_line": row.sample.source_line, **row.arms(),
            }, allow_nan=False))
    except (ParseError, OSError, OverflowError) as exc:
        print(f"M1 feature extraction refused: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
