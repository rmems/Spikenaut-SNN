# SPDX-License-Identifier: MIT OR Apache-2.0
"""exp-026 JSON documents: ``comparison.json``, ``timing.json`` and the
variant bundle's ``variant`` provenance block.

Pure views over a stepped and scored ``recurrence_eval.Comparison``;
nothing here computes a number. ``comparison.json`` carries no wall-clock
field, so on the same input it is identical across runs apart from
``source`` (the checkout's commit and branch, recorded for provenance).
``timing.json`` is host wall-clock and never a deterministic artifact.
"""

from __future__ import annotations

import platform
import sys
from pathlib import Path
from typing import Any

try:  # package import: `python3 -m tools.recurrence_eval`
    from .hamming_const import FROZEN_LINEAGE, FROZEN_MINMAX, LIVE_COLUMNS
    from .recurrence_bundle import EXPERIMENT_ID
    from .recurrence_core import (
        LAMBDA_GRID,
        PRIMARY_DEFINITION,
        READOUT_FIT_METHOD,
        SPLITS,
    )
    from .recurrent_lif import (
        ORDER_SEMANTICS,
        RECURRENCE_DELAY_TICKS,
        RECURRENCE_SCHEMA,
        STATE_SEMANTICS,
    )
    from .replay_core import REPO_ROOT, git_revision
except ImportError:  # direct script: `python3 tools/recurrence_eval.py`
    from hamming_const import FROZEN_LINEAGE, FROZEN_MINMAX, LIVE_COLUMNS
    from recurrence_bundle import EXPERIMENT_ID
    from recurrence_core import (
        LAMBDA_GRID,
        PRIMARY_DEFINITION,
        READOUT_FIT_METHOD,
        SPLITS,
    )
    from recurrent_lif import (
        ORDER_SEMANTICS,
        RECURRENCE_DELAY_TICKS,
        RECURRENCE_SCHEMA,
        STATE_SEMANTICS,
    )
    from replay_core import REPO_ROOT, git_revision

COMPARISON_SCHEMA = "spikenaut.recurrence-comparison.v1"


def display_path(path: Path) -> str:
    """Repo-relative POSIX path when inside the checkout, else as given."""
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _model_block(cmp: Any) -> dict:
    entry = cmp.entry
    return {
        "id": entry.id,
        "checkpoint": entry.checkpoint_relative,
        "checkpoint_digest": entry.checkpoint_digest,
        "feature_map_id": entry.feature_map_id,
        "output_contract_id": entry.output_contract_id,
        "numeric_format": entry.numeric_format,
        "training_dataset_digest": entry.training_dataset_digest,
        "bank_manifest": (
            display_path(cmp.bank.manifest_path)
            if cmp.bank.manifest_path is not None else None
        ),
    }


def _input_block(cmp: Any) -> dict:
    return {
        "path": (
            cmp.input_label if cmp.input_label is not None
            else display_path(cmp.jsonl)
        ),
        "sha256": cmp.input_sha256,
        "splits": {
            s: {
                "n_ticks": len(cmp.splits[s]),
                "sessions": [span.session for span in cmp.spans[s]],
            }
            for s in SPLITS
        },
        "encoder": {
            "kind": "analog-current, frozen minmax (not spikes)",
            "columns": list(LIVE_COLUMNS),
            "frozen_minmax_lineage": FROZEN_LINEAGE,
            "frozen_minmax": {k: list(v) for k, v in FROZEN_MINMAX.items()},
            "missing_policy": "encode-zero (missing live sensor encodes to 0)",
        },
    }


def _protocol_block(cmp: Any) -> dict:
    return {
        "k": cmp.k,
        "i_drive": cmp.i_drive,
        "gains": list(cmp.gains),
        "lambda_grid_rel": list(LAMBDA_GRID),
        "readout_fit": READOUT_FIT_METHOD,
        "primary": PRIMARY_DEFINITION,
        "target": (
            "Distill sample_readout_target: (comfort, temp, power) with "
            "temp = gpu_temp_c axon, power = power_w axon, "
            "comfort = clamp(1 - 0.5*temp - 0.5*power, 0, 1), binary32"
        ),
        "stepper": (
            "tools/hamming_lif.py keep_lif_step on the Q8.8 grid; "
            "recurrent arms via tools/recurrent_lif.py recurrent_lif_step"
        ),
        "recurrence_contract": {
            "schema": RECURRENCE_SCHEMA,
            "delay_ticks": RECURRENCE_DELAY_TICKS,
            "family": "I->E lateral inhibition at one gain; E->E, E->I, I->I, self = 0",
            "state": STATE_SEMANTICS,
            "order": ORDER_SEMANTICS,
        },
        "session_reset": "membrane and previous-spike state reset at each episode_id boundary",
    }


def comparison_json(cmp: Any) -> dict:
    """The ``comparison.json`` document (deterministic apart from ``source``)."""
    return {
        "schema_version": COMPARISON_SCHEMA,
        "experiment": EXPERIMENT_ID,
        "model": _model_block(cmp),
        "input": _input_block(cmp),
        "protocol": _protocol_block(cmp),
        "arm_order": list(cmp.arm_order),
        "arms": cmp.arms,
        "baselines": cmp.baselines(),
        "selection": cmp.selection(),
        "source": git_revision(REPO_ROOT),
    }


def timing_json(cmp: Any) -> dict:
    """The ``timing.json`` document: host stepper wall-clock, never FPGA latency."""
    ff = cmp.timing["ff"]["test"]["us_per_tick"]
    return {
        "note": (
            "Wall-clock of the Python binary32 reference stepper on this host "
            "(tools/hamming_lif.py / tools/recurrent_lif.py). Host latency only: "
            "not FPGA latency, not a deterministic artifact, excluded from "
            "comparison.json."
        ),
        "python": sys.version,
        "platform": platform.platform(),
        "arms": cmp.timing,
        "test_split_ratio_vs_ff": {
            name: cmp.timing[name]["test"]["us_per_tick"] / ff for name in cmp.timing
        },
        "structural_cost_per_tick": (
            "feed-forward: 16x16 MAC on stim; recurrent: + up to 16x16 MAC on "
            "s(t-1) (48 nonzero for I->E lateral inhibition) and one 16-bit "
            "register for s(t-1); one tick of added algorithmic delay"
        ),
    }


def provenance_json(cmp: Any, selected: str) -> dict:
    """The emitted bundle's ``variant`` block for the selected recurrent arm."""
    arm = cmp.arms[selected]
    train = cmp.spans["train"]
    return {
        "experiment": EXPERIMENT_ID,
        "issue": "rmems/Spikenaut-SNN#3",
        "derived_from": {
            "model_id": cmp.entry.id,
            "checkpoint": cmp.entry.checkpoint_relative,
            "checkpoint_digest": cmp.entry.checkpoint_digest,
            "unchanged_fields": [
                "threshold", "decay_rate", "weights", "inhibitory",
                "membrane_potential", "last_spike",
            ],
        },
        "gain": arm["gain"],
        "readout_fit": {
            "method": READOUT_FIT_METHOD,
            "lambda_rel": arm["lambda_rel"],
            "lambda_abs": arm["lambda_abs"],
            "k": cmp.k,
            "i_drive": cmp.i_drive,
            "train_sessions": f"{train[0].session}..{train[-1].session} ({len(train)} sessions)",
            "train_ticks": len(cmp.splits["train"]),
            "input_sha256": cmp.input_sha256,
            "selected_on": "validation primary",
        },
        "status": (
            "research variant for GH #3; not the shipped bank, not loaded by "
            "the Rust crate, no FPGA/board evidence (silicon-hdl#76)"
        ),
    }
