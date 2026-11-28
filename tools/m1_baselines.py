"""Score conventional models on the M1 paired rows in whole held-out sessions."""

import argparse
import json
import math
import sys

from .m1_arm_plan import ARMS, paired_coverage
from .m1_cohort import add_cohort_args, cohort_from_args
from .m1_features import paired_rows
from .q88_core import ParseError
from .recurrence_core import (
    HEADS, _accumulate_normal, constant_predictor, metrics_from_predictions,
    readout_target, session_spans,
)
from .recurrence_linalg import solve_dense

# Fixed before scoring either holdout. The intercept is unpenalized.
RIDGE = 0.001


def features(row, arm):
    if arm == "RAW":
        return row.raw
    if arm == "KINETIC":
        return row.kinetic
    return row.raw + row.kinetic


def design(values, bounds):
    return [((v - lo) / (hi - lo) if hi > lo else 0.0)
            for v, (lo, hi) in zip(values, bounds, strict=True)] + [1.0]


def training_bounds(rows, arm):
    bounds = []
    for index in range(len(features(rows[0], arm))):
        lo = min(features(r, arm)[index] for r in rows)
        hi = max(features(r, arm)[index] for r in rows)
        bounds.append((lo, hi))
    return bounds


def fit(rows, targets, arm):
    bounds = training_bounds(rows, arm)
    width = len(bounds) + 1
    gram = [[0.0] * width for _ in range(width)]
    rhs = [[0.0] * len(HEADS) for _ in range(width)]
    for row, target in zip(rows, targets, strict=True):
        x = design(features(row, arm), bounds)
        _accumulate_normal(gram, rhs, x, target, width)
    for i in range(width - 1):
        gram[i][i] += RIDGE
    return {"train_minmax": bounds, "weights": [solve_dense(gram, b) for b in zip(*rhs)]}


def predict(model, row, arm):
    x = design(features(row, arm), model["train_minmax"])
    return tuple(math.fsum(a * b for a, b in zip(w, x, strict=True)) for w in model["weights"])


def preceding_targets(cohort):
    # Persistence uses the preceding source observation, even when that tick
    # was ineligible for an unrelated missing sensor. Never bridge episodes.
    previous_targets = {}
    previous = None
    for sample in cohort.samples:
        if (cohort.split_by_episode[sample.episode_id] != "train" and previous is not None
                and previous.episode_id == sample.episode_id
                and not {"gpu_temp_c", "power_w"}.intersection(previous.missing)):
            previous_targets[(sample.episode_id, sample.source_line)] = readout_target(previous.stim)
        previous = sample
    return previous_targets


def score_holdout(rows, previous_targets, train_mean, models):
    samples = [r.sample for r in rows]
    spans = session_spans(samples)
    targets = [readout_target(s.stim) for s in samples]
    def score(predictions):
        return metrics_from_predictions(predictions, targets, spans)
    return {
        "train_mean": score([train_mean] * len(rows)),
        "persistence": score([previous_targets.get((s.episode_id, s.source_line), train_mean)
                              for s in samples]),
        **{arm: score([predict(models[arm], row, arm) for row in rows]) for arm in ARMS},
    }


def grouped_rows(cohort, coverage):
    splits = {s: [] for s in coverage}
    for row in paired_rows(cohort):
        splits[row.split].append(row)
    empty = [s for s, rows in splits.items() if not rows]
    if empty:
        raise ParseError(f"no paired rows in {empty}")
    return splits


def evaluate(cohort):
    coverage = paired_coverage(cohort)
    splits = grouped_rows(cohort, coverage)
    targets = [readout_target(r.sample.stim) for r in splits["train"]]
    train_mean = constant_predictor(targets)
    models = {arm: fit(splits["train"], targets, arm) for arm in ARMS}
    previous_targets = preceding_targets(cohort)
    metrics = {s: score_holdout(splits[s], previous_targets, train_mean, models)
               for s in ("val", "test")}
    return {
        "schema_version": "spikenaut.m1-conventional-baselines.v1",
        "target": list(HEADS), "target_horizon_ticks": 0, "train_mean": train_mean,
        "ridge_lambda": RIDGE, "models": models, "metrics": metrics,
        "cohort": cohort.report, "paired_coverage": coverage,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_cohort_args(parser)
    args = parser.parse_args(argv)
    try:
        result = evaluate(cohort_from_args(args))
        print(json.dumps(result, sort_keys=True, allow_nan=False))
    except (ParseError, OSError, OverflowError) as exc:
        print(f"M1 baseline evaluation refused: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
