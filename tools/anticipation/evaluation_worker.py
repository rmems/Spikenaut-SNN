"""Isolated Python compute stages, terminated by the parent shared deadline."""

import argparse
import json
from pathlib import Path

from .campaign import write_json
from .report import baselines, comparison


def run(stage, prepared, output):
    # prepared/output are internal orchestrator arguments (parsed from the
    # `-m tools.anticipation.evaluation_worker` argv spawned by evaluate.
    # _python_stage), not request data. Resolve them and confine every write to
    # the resolved output tree so no path can escape the intended directory.
    prepared = Path(prepared).resolve()
    output = Path(output).resolve()
    status_path = (output / f"{stage}-status.json").resolve()
    if not status_path.is_relative_to(output):
        raise ValueError("status path escapes output directory")
    if stage == "baselines":
        baselines(json.loads(prepared.read_text()), output)
        complete = True
    else:
        baseline_results = json.loads((output / "baselines.json").read_text())
        complete = comparison(prepared, output, baseline_results)["complete"]
    write_json(status_path, {"complete": complete})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("baselines", "comparison"))
    parser.add_argument("prepared", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    run(args.stage, args.prepared, args.output)
