"""Session-weighted metrics and validation-selected forecasting baselines."""

from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from .campaign import write_json, sha256

TARGETS = [
    "temperature_change_1s_c",
    "power_change_1s_w",
    "temperature_change_5s_c",
    "power_change_5s_w",
]


def metrics(actual, predicted, session_ids, target_std):
    actual, predicted, std = map(np.asarray, (actual, predicted, target_std))
    ids = np.asarray(session_ids)
    if (
        actual.ndim != 2
        or actual.shape[1:] != (4,)
        or actual.shape != predicted.shape
        or len(actual) == 0
    ):
        raise ValueError("metrics require nonempty paired four-target arrays")
    if len(ids) != len(actual) or std.shape != (4,) or not np.all(std > 0):
        raise ValueError("invalid session IDs or target scales")
    if not all(np.all(np.isfinite(v)) for v in (actual, predicted, std)):
        raise ValueError("nonfinite predictions or targets")
    by_session = {}
    for sid in sorted(set(ids)):
        errors = predicted[ids == sid] - actual[ids == sid]
        mae = np.mean(np.abs(errors), axis=0)
        by_session[str(sid)] = {
            "n": len(errors),
            "mae": mae.tolist(),
            "rmse": np.sqrt(np.mean(errors**2, axis=0)).tolist(),
            "primary": float(np.mean(mae[2:] / std[2:])),
        }
    return {
        "primary": float(np.mean([s["primary"] for s in by_session.values()])),
        "mae": np.mean([s["mae"] for s in by_session.values()], axis=0).tolist(),
        "rmse": np.mean([s["rmse"] for s in by_session.values()], axis=0).tolist(),
        "per_session": by_session,
    }


def select_baseline(results):
    return min(results, key=lambda name: results[name]["validation"]["primary"])


def examples(data, split, history=False):
    x, y, ids, keys = [], [], [], []
    for session in data["sessions"]:
        if session["split"] != split:
            continue
        frames = session["frames"]
        for e in session["examples"]:
            indices = [e["frame_index"]] + (e["history_indices"] if history else [])
            x.append([value for i in indices for value in frames[i]["x"]])
            y.append(e["y"])
            ids.append(session["session_id"])
            keys.append((session["session_id"], e["frame_index"]))
    return np.asarray(x, dtype=float), np.asarray(y, dtype=float), ids, keys


def ridge_fit(x, y, alpha):
    design = np.column_stack((np.ones(len(x)), x))
    penalty = np.eye(design.shape[1]) * alpha
    penalty[0, 0] = 0
    return np.linalg.solve(design.T @ design + penalty, design.T @ y)


def ridge_predict(x, weights):
    return np.column_stack((np.ones(len(x)), x)) @ weights


def baselines(data, output):
    output = Path(output)
    norm = data["normalization"]
    mean, std = np.array(norm["x_mean"]), np.array(norm["x_std"])
    ymean, ystd = np.array(norm["y_mean"]), np.array(norm["y_std"])
    results = {}
    arrays = {split: examples(data, split) for split in ("train", "validation", "test")}
    results["persistence"] = {
        split: metrics(v[1], np.zeros_like(v[1]), v[2], ystd)
        for split, v in arrays.items()
        if split != "train"
    }
    all_predictions = {}
    for name, history in [("current_ridge", False), ("history_ridge", True)]:
        rows = {split: examples(data, split, history) for split in arrays}
        xm, xs = np.tile(mean, 5 if history else 1), np.tile(std, 5 if history else 1)
        normalized = {split: (v[0] - xm) / xs for split, v in rows.items()}
        candidates = []
        for alpha in (0.001, 0.01, 0.1, 1, 10):
            weights = ridge_fit(
                normalized["train"], (rows["train"][1] - ymean) / ystd, alpha
            )
            prediction = ridge_predict(normalized["validation"], weights) * ystd + ymean
            candidates.append(
                (
                    metrics(
                        rows["validation"][1], prediction, rows["validation"][2], ystd
                    )["primary"],
                    alpha,
                    weights,
                )
            )
        _, alpha, weights = min(candidates, key=lambda item: item[0])
        results[name] = {
            "selected_alpha": alpha,
            "validation_candidates": [
                {"alpha": a, "primary": p} for p, a, _ in candidates
            ],
        }
        all_predictions[name] = []
        for split in ("validation", "test"):
            prediction = ridge_predict(normalized[split], weights) * ystd + ymean
            results[name][split] = metrics(
                rows[split][1], prediction, rows[split][2], ystd
            )
            all_predictions[name].extend(
                {
                    "session_id": sid,
                    "frame_index": i,
                    "split": split,
                    "prediction": p.tolist(),
                }
                for (sid, i), p in zip(rows[split][3], prediction)
            )
        write_json(
            output / (name + "-checkpoint.json"),
            {
                "schema_version": "ridge-forecast-v1",
                "feature_map_id": data["feature_map_id"],
                "feature_map": data["feature_map"],
                "target_names": data.get("target_names", TARGETS),
                "normalization": norm,
                "history": history,
                "alpha": alpha,
                "weights": weights.tolist(),
                "provenance": data.get("provenance"),
            },
        )
    all_predictions["persistence"] = [
        {"session_id": sid, "frame_index": i, "split": split, "prediction": [0.0] * 4}
        for split in ("validation", "test")
        for sid, i in arrays[split][3]
    ]
    write_json(output / "baseline-predictions.json", all_predictions)
    write_json(output / "baselines.json", results)
    return results


def checked_predictions(artifact, run):
    if (artifact.get("arm"), artifact.get("seed")) != (run["arm"], run["seed"]):
        raise ValueError("prediction artifact identity does not match arm/seed")
    rows = artifact["predictions"]
    keys = [(p["session_id"], p["frame_index"]) for p in rows]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate prediction keys")
    if any(p["split"] not in ("validation", "test") for p in rows):
        raise ValueError("unexpected prediction split")
    return rows


def comparison(prepared_path, output, baseline_results=None):
    output = Path(output)
    data = json.loads(Path(prepared_path).read_text())
    base = baseline_results if baseline_results is not None else baselines(data, output)
    strongest = select_baseline(base)
    std = np.asarray(data["normalization"]["y_std"])
    report = {
        "schema_version": "anticipation-comparison-v1",
        "prepared_sha256": sha256(prepared_path),
        "target_names": data.get("target_names", TARGETS),
        "baseline_selected_on_validation": strongest,
        "baselines": base,
        "runs": [],
        "quality": data.get("quality"),
        "limits": "Three test sessions support a pilot conclusion only; exp-025 is lineage with a different task/input contract.",
    }
    summary_path = output / "snn" / "summary.json"
    summary = (
        json.loads(summary_path.read_text()) if summary_path.exists() else {"runs": []}
    )
    if summary_path.exists() and summary.get("prepared_sha256") != sha256(
        prepared_path
    ):
        raise ValueError("SNN summary does not match the prepared data hash")
    expected = {(arm, seed) for arm in ("uniform", "mixed") for seed in (123, 456, 789)}
    observed = [(r["arm"], r["seed"]) for r in summary["runs"]]
    if len(set(observed)) != len(observed) or not set(observed) <= expected:
        raise ValueError("unexpected or duplicate SNN run")
    summary["runs"].extend(
        {"arm": arm, "seed": seed, "status": "unfinished"}
        for arm, seed in sorted(expected - set(observed))
    )
    for run in summary["runs"]:
        item = dict(run)
        predpath = run.get("predictions")
        if predpath:
            pred = json.loads((output / "snn" / predpath).read_text())
            prediction_rows = checked_predictions(pred, run)
            for split in ("validation", "test"):
                _, actual, ids, keys = examples(data, split)
                lookup = {
                    (p["session_id"], p["frame_index"]): p["prediction"]
                    for p in prediction_rows
                    if p["split"] == split
                }
                if len(lookup) != len(keys) or set(lookup) != set(keys):
                    raise ValueError(
                        "SNN predictions do not use the common eligible examples"
                    )
                item[split] = metrics(
                    actual, np.array([lookup[k] for k in keys]), ids, std
                )
            ref = base[strongest]["test"]
            score = item["test"]
            item["primary_relative_improvement"] = (
                (ref["primary"] - score["primary"]) / ref["primary"]
                if ref["primary"]
                else None
            )
            item["promising"] = bool(
                ref["primary"] > 0
                and score["primary"] <= 0.95 * ref["primary"]
                and all(score["mae"][i] <= 1.05 * ref["mae"][i] for i in (2, 3))
            )
            item["diagnostics"] = pred.get("diagnostics")
        report["runs"].append(item)
    report["complete"] = len(report["runs"]) == 6 and all(
        r.get("status") == "complete" and "test" in r for r in report["runs"]
    )
    report["seed_variation"] = {
        arm: {
            "mean_primary": float(np.mean(scores)),
            "std_primary": float(np.std(scores)),
            "min_primary": float(min(scores)),
            "max_primary": float(max(scores)),
        }
        for arm in ("uniform", "mixed")
        if (
            scores := [
                r["test"]["primary"]
                for r in report["runs"]
                if r["arm"] == arm and "test" in r
            ]
        )
    }
    lines = [
        "# Anticipation pilot results",
        "",
        f"Strongest baseline selected on validation: **{strongest}**.",
        "",
        "| Model | Seed | Validation primary | Test primary | Test 5s temperature MAE (C) | Test 5s power MAE (W) | Promising |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for name, b in base.items():
        lines.append(
            f"| {name} | — | {b['validation']['primary']:.5f} | {b['test']['primary']:.5f} | {b['test']['mae'][2]:.4f} | {b['test']['mae'][3]:.4f} | baseline |"
        )
    for r in report["runs"]:
        if "test" in r:
            lines.append(
                f"| {r['arm']} LIF | {r['seed']} | {r['validation']['primary']:.5f} | {r['test']['primary']:.5f} | {r['test']['mae'][2]:.4f} | {r['test']['mae'][3]:.4f} | {r['promising']} |"
            )
        else:
            lines.append(
                f"| {r['arm']} LIF | {r['seed']} | unfinished | — | — | — | {r['status']} |"
            )
    passing = [r for r in report["runs"] if r.get("promising")]
    lines += [
        "",
        f"{len(passing)} of six scheduled SNN runs meet both predeclared improvement criteria.",
        "All comparisons completed."
        if report["complete"]
        else "**Campaign comparison incomplete.**",
        "",
        report["limits"],
        "",
        "Primary = average of temperature/power five-second MAE divided by training target standard deviations, with sessions weighted equally.",
        "Physical MAE and RMSE at both horizons, per-session errors, normalization, rejection reasons, spikes and silent neurons are retained in the machine-readable artifacts.",
    ]
    scored = [(name, value) for name, value in base.items()]
    scored += [(f"{r['arm']} / {r['seed']}", r) for r in report["runs"] if "test" in r]
    lines += [
        "",
        "## Physical-unit test errors",
        "",
        "Values below average the per-session metrics with equal session weights.",
        "",
        "| Model | Target | MAE | RMSE |",
        "|---|---|---:|---:|",
    ]
    labels = [
        "Temperature +1s (C)",
        "Power +1s (W)",
        "Temperature +5s (C)",
        "Power +5s (W)",
    ]
    for name, value in scored:
        for i, label in enumerate(labels):
            lines.append(
                f"| {name} | {label} | {value['test']['mae'][i]:.5f} | {value['test']['rmse'][i]:.5f} |"
            )
    session_ids = sorted(
        {sid for _, value in scored for sid in value["test"]["per_session"]}
    )
    lines += [
        "",
        "## Every test session",
        "",
        "| Model | " + " | ".join(session_ids) + " |",
        "|---|" + "---:|" * len(session_ids),
    ]
    for name, value in scored:
        cells = [
            f"{value['test']['per_session'][sid]['primary']:.5f}" for sid in session_ids
        ]
        lines.append("| " + name + " | " + " | ".join(cells) + " |")
    lines += [
        "",
        "## Variation across seeds",
        "",
        "| Arm | Mean primary | Standard deviation | Minimum | Maximum |",
        "|---|---:|---:|---:|---:|",
    ]
    for arm, values in report["seed_variation"].items():
        lines.append(
            f"| {arm} | {values['mean_primary']:.5f} | {values['std_primary']:.5f} | {values['min_primary']:.5f} | {values['max_primary']:.5f} |"
        )
    lines += [
        "",
        "## Spike activity",
        "",
        "Diagnostics combine validation and test ticks, including warm-up.",
        "",
        "| Arm | Seed | Silent neurons | Mean rate per neuron (Hz) |",
        "|---|---:|---:|---:|",
    ]
    for run in report["runs"]:
        diagnostic = run.get("diagnostics")
        if diagnostic:
            lines.append(
                f"| {run['arm']} | {run['seed']} | {len(diagnostic['silent_neurons'])} | {np.mean(diagnostic['spike_rate_hz']):.5f} |"
            )
    write_json(output / "comparison.json", report)
    (output / "report.md").write_text("\n".join(lines) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prepared", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    comparison(args.prepared, args.output)
