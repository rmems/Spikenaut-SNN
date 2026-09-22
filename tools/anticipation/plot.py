"""Export comparison and learning-curve figures from persisted pilot results."""

import argparse
import json
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_results(output):
    output = Path(output)
    report = json.loads((output / "comparison.json").read_text())
    entries = [(name, "baseline", value) for name, value in report["baselines"].items()]
    entries += [
        (f"{r['arm']} / {r['seed']}", r["arm"], r)
        for r in report["runs"]
        if "test" in r
    ]
    colors = {"baseline": "#687587", "uniform": "#7063bd", "mixed": "#158c8c"}
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(
        [e[0] for e in entries],
        [e[2]["test"]["primary"] for e in entries],
        color=[colors[e[1]] for e in entries],
    )
    ax.invert_yaxis()
    ax.set_xlabel(
        "Five-second standardized MAE · equal weight per test session · lower is better"
    )
    ax.set_title("Machine-state anticipation pilot")
    ref = report["baselines"][report["baseline_selected_on_validation"]]["test"][
        "primary"
    ]
    ax.axvline(
        ref * 0.95, color="#b95335", linestyle="--", label="5% improvement threshold"
    )
    ax.legend(loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(output / "comparison.png", dpi=180)
    fig.savefig(output / "comparison.svg")
    plt.close(fig)
    _plot_learning_curves(output, report)


def _plot_learning_curves(output, report):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    seedcolors = {123: "#4378a6", 456: "#bf6c38", 789: "#4b9970"}
    for run in report["runs"]:
        _plot_run_curve(output, run, axes, seedcolors)
    axes[0].set_ylabel("Training standardized squared error")
    axes[1].set_ylabel("Validation five-second standardized MAE")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "learning-curves.png", dpi=180)
    fig.savefig(output / "learning-curves.svg")
    plt.close(fig)


def _plot_run_curve(output, run, axes, seedcolors):
    curve = output / "snn" / f"{run['arm']}-{run['seed']}" / "learning_curves.json"
    if not curve.exists():
        return
    rows = json.loads(curve.read_text())
    if not rows:
        return
    style = "-" if run["arm"] == "uniform" else "--"
    for ax, key in zip(axes, ["train_loss", "validation_primary"]):
        if key not in rows[0]:
            continue
        ax.plot(
            [r["epoch"] for r in rows],
            [r[key] for r in rows],
            color=seedcolors[run["seed"]],
            linestyle=style,
            label=f"{run['arm']} {run['seed']}",
        )
        ax.set_xlabel("Epoch")
        ax.spines[["top", "right"]].set_visible(False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    plot_results(parser.parse_args().output)
