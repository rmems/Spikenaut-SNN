"""Isolated Python compute stages, terminated by the parent shared deadline."""

import argparse
import json
from pathlib import Path

from .campaign import write_json
from .report import baselines, comparison


def run(stage, prepared, output):
    if stage == "baselines":
        baselines(json.loads(prepared.read_text()), output)
        complete = True
    else:
        baseline_results = json.loads((output / "baselines.json").read_text())
        complete = comparison(prepared, output, baseline_results)["complete"]
    write_json(output / f"{stage}-status.json", {"complete": complete})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("baselines", "comparison"))
    parser.add_argument("prepared", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    run(args.stage, args.prepared, args.output)
