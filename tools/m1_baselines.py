"""Score conventional models on the M1 paired rows in whole held-out sessions."""

import argparse
import json
import math
import sys

from .m1_arm_plan import ARMS, build_plan
from .m1_cohort import add_cohort_args, cohort_from_args
from .m1_features import paired_rows
from .model_bank import BankError
from .q88_core import ParseError
from .recurrence_core import (
    HEADS, constant_predictor, metrics_from_predictions, readout_target, session_spans,
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


def fit(rows, targets, arm):
    bounds = []
    for index in range(len(features(rows[0], arm))):
        lo = min(features(r, arm)[index] for r in rows)
        hi = max(features(r, arm)[index] for r in rows)
        bounds.append((lo, hi))
    width = len(bounds) + 1
    gram = [[0.0] * width for _ in range(width)]
    rhs = [[0.0] * width for _ in HEADS]
    for row, target in zip(rows, targets, strict=True):
        x = design(features(row, arm), bounds)
        for i, xi in enumerate(x):
            for j in range(i + 1):
                gram[i][j] += xi * x[j]
            for head, value in enumerate(target):
                rhs[head][i] += xi * value
    for i in range(width):
        for j in range(i):
            gram[j][i] = gram[i][j]
        if i < width - 1:
            gram[i][i] += RIDGE
    return {"train_minmax": bounds, "weights": [solve_dense(gram, b) for b in rhs]}


def predict(model, row, arm):
    x = design(features(row, arm), model["train_minmax"])
    return tuple(math.fsum(a * b for a, b in zip(w, x, strict=True)) for w in model["weights"])


def evaluate(cohort):
    plan = build_plan(cohort)
    if plan["splits_without_common_rows"]:
        raise ParseError(f"no paired rows in {plan['splits_without_common_rows']}")
    splits = {s: [] for s in plan["paired_coverage"]}
    for row in paired_rows(cohort):
        splits[row.split].append(row)
    targets = {s: [readout_target(r.sample.stim) for r in rows] for s, rows in splits.items()}
    train_mean = constant_predictor(targets["train"])
    models = {arm: fit(splits["train"], targets["train"], arm) for arm in ARMS}

    # Persistence uses the preceding source observation, even when that tick
    # was ineligible for an unrelated missing sensor. Never bridge episodes.
    previous_targets = {}
    previous = None
    for sample in cohort.samples:
        if (cohort.split_by_episode[sample.episode_id] != "train" and previous is not None
                and previous.episode_id == sample.episode_id
                and not {"gpu_temp_c", "power_w"}.intersection(previous.missing)):
            previous_targets[sample.source_line] = readout_target(previous.stim)
        previous = sample
    metrics = {}
    for split in ("val", "test"):
        rows = splits[split]
        spans = session_spans([r.sample for r in rows])
        def score(predictions):
            return metrics_from_predictions(predictions, targets[split], spans)
        metrics[split] = {
            "train_mean": score([train_mean] * len(rows)),
            "persistence": score([previous_targets.get(r.sample.source_line, train_mean) for r in rows]),
            **{arm: score([predict(models[arm], row, arm) for row in rows]) for arm in ARMS},
        }
    return {
        "schema_version": "spikenaut.m1-conventional-baselines.v1",
        "target": list(HEADS), "target_horizon_ticks": 0, "train_mean": train_mean,
        "ridge_lambda": RIDGE, "models": models, "metrics": metrics,
        "cohort": cohort.report, "paired_coverage": plan["paired_coverage"],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_cohort_args(parser)
    args = parser.parse_args(argv)
    try:
        result = evaluate(cohort_from_args(args))
        print(json.dumps(result, sort_keys=True, allow_nan=False))
    except (ParseError, BankError, OSError, OverflowError) as exc:
        print(f"M1 baseline evaluation refused: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
