# SPDX-License-Identifier: MIT OR Apache-2.0
"""exp-026 artifacts: the recurrent checkpoint bundle and the markdown report.

A recurrent variant is a *separately named* checkpoint, never an edit of
``dataset/merged_v2``. The bundle this writes is the same shape a reader
already knows -- ``snn_model.json`` + ``model_bank.json`` + the four
``.mem`` images -- plus the versioned extras the ``spikenaut.recurrence.v1``
contract adds: a ``recurrence`` block in the checkpoint and
``parameters_recurrent_weights.mem``. The three images that carry the
unchanged hidden parameters are byte-identical to the shipped ones, which
``tests/test_recurrence.py`` asserts.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

try:  # package import: `python3 -m tools.recurrence_eval`
    from .hamming_const import N_NEURONS
    from .model_bank import (
        FEATURE_MAP_LIVE_EXP_025,
        NUMERIC_FORMAT_Q88,
        OUTPUT_CONTRACT_SUPERVISOR_V3_RM1150,
        dumps_manifest,
    )
    from .q88_core import ParseError, Q88RangeError, encode_q88_hex
    from .recurrence_core import HEADS, N_HEADS
    from .recurrent_lif import (
        RECURRENCE_KEY,
        RECURRENT_MEM_NAME,
        Recurrence,
        recurrence_block,
        recurrent_mem_words,
    )
    from .replay_inputs import sha256_bytes
except ImportError:  # direct script: `python3 tools/recurrence_eval.py`
    from hamming_const import N_NEURONS
    from model_bank import (
        FEATURE_MAP_LIVE_EXP_025,
        NUMERIC_FORMAT_Q88,
        OUTPUT_CONTRACT_SUPERVISOR_V3_RM1150,
        dumps_manifest,
    )
    from q88_core import ParseError, Q88RangeError, encode_q88_hex
    from recurrence_core import HEADS, N_HEADS
    from recurrent_lif import (
        RECURRENCE_KEY,
        RECURRENT_MEM_NAME,
        Recurrence,
        recurrence_block,
        recurrent_mem_words,
    )
    from replay_inputs import sha256_bytes

MEM_THRESHOLDS = "parameters.mem"
MEM_DECAY = "parameters_decay.mem"
MEM_HIDDEN = "parameters_weights.mem"
MEM_OUTPUT = "parameters_output_weights.mem"
UNCHANGED_MEM_IMAGES = (MEM_THRESHOLDS, MEM_DECAY, MEM_HIDDEN)
BUNDLE_FILES = UNCHANGED_MEM_IMAGES + (MEM_OUTPUT, RECURRENT_MEM_NAME)
EXPERIMENT_ID = "exp-026"


def build_checkpoint(
    base_model: dict,
    recurrence: Recurrence,
    readout: list[list[float]],
    provenance: dict[str, Any],
) -> dict:
    """The variant checkpoint: base neurons, refit head, recurrence block.

    Every neuron field other than ``output_weights`` is copied verbatim
    from the base checkpoint so the hidden ``.mem`` images cannot drift.
    """
    if RECURRENCE_KEY in base_model:
        raise ParseError("base checkpoint already declares recurrence")
    if len(readout) != N_NEURONS or any(len(row) != N_HEADS for row in readout):
        raise ParseError(f"readout must be {N_NEURONS}x{N_HEADS}")
    neurons = []
    for i, neuron in enumerate(base_model["neurons"]):
        copy = dict(neuron)
        copy["output_weights"] = [float(v) for v in readout[i]]
        neurons.append(copy)
    out = dict(base_model)
    out["neurons"] = neurons
    out[RECURRENCE_KEY] = recurrence_block(recurrence)
    out["variant"] = dict(provenance)
    return out


def checkpoint_bytes(model: dict) -> bytes:
    """Compact, ASCII, LF-terminated -- the bytes the bank digest covers."""
    try:
        text = json.dumps(model, ensure_ascii=True, allow_nan=False, separators=(",", ":"))
    except (TypeError, ValueError) as exc:
        raise ParseError(f"checkpoint is not serializable: {exc}") from exc
    return (text + "\n").encode("ascii")


def _hex_words(values: list[float], where: str) -> list[str]:
    try:
        return [encode_q88_hex(v) for v in values]
    except Q88RangeError as exc:
        raise ParseError(f"{where}: {exc}") from exc


def mem_images(model: dict, recurrence: Recurrence) -> dict[str, bytes]:
    """The five ``$readmemh`` images, one 4-digit word per LF line."""
    neurons = model["neurons"]
    images = {
        MEM_THRESHOLDS: _hex_words([n["threshold"] for n in neurons], MEM_THRESHOLDS),
        MEM_DECAY: _hex_words([n["decay_rate"] for n in neurons], MEM_DECAY),
        MEM_HIDDEN: _hex_words([w for n in neurons for w in n["weights"]], MEM_HIDDEN),
        MEM_OUTPUT: _hex_words(
            [w for n in neurons for w in n["output_weights"]], MEM_OUTPUT
        ),
        RECURRENT_MEM_NAME: recurrent_mem_words(recurrence),
    }
    return {name: ("\n".join(words) + "\n").encode("ascii") for name, words in images.items()}


def bank_manifest(
    model_id: str, digest: str, training_dataset_digest: str | None
) -> bytes:
    entry: dict[str, Any] = {
        "id": model_id,
        "checkpoint": "snn_model.json",
        "checkpoint_digest": digest,
        "feature_map_id": FEATURE_MAP_LIVE_EXP_025,
        "output_contract_id": OUTPUT_CONTRACT_SUPERVISOR_V3_RM1150,
        "numeric_format": NUMERIC_FORMAT_Q88,
    }
    if training_dataset_digest is not None:
        entry["training_dataset_digest"] = training_dataset_digest
    return dumps_manifest({"schema_version": 1, "models": [entry]}).encode("utf-8")


def write_bundle(
    out_dir: Path,
    *,
    model_id: str,
    base_model: dict,
    recurrence: Recurrence,
    readout: list[list[float]],
    provenance: dict[str, Any],
    training_dataset_digest: str | None,
) -> dict[str, str]:
    """Write the bundle; returns ``{relative path: sha256}``."""
    model = build_checkpoint(base_model, recurrence, readout, provenance)
    payload = checkpoint_bytes(model)
    digest = sha256_bytes(payload)
    files = {"snn_model.json": payload}
    files.update(mem_images(model, recurrence))
    files["model_bank.json"] = bank_manifest(model_id, digest, training_dataset_digest)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, str] = {}
    for name, data in files.items():
        (out_dir / name).write_bytes(data)
        written[name] = sha256_bytes(data)
    return written


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------


def _fmt(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


def _arm_row(name: str, arm: dict) -> str:
    test = arm["test"]
    act = arm["activity_test"]
    heads = " | ".join(_fmt(v) for v in test["session_weighted_mae"])
    gain = "-" if arm.get("gain") is None else repr(arm["gain"])
    lam = "-" if arm.get("lambda_rel") is None else repr(arm["lambda_rel"])
    silent = ",".join(str(i) for i in act["silent_neurons"]) or "none"
    return (
        f"| `{name}` | {gain} | {lam} | {_fmt(arm['val']['primary'])} | "
        f"{_fmt(test['primary'])} | {heads} | {act['spikes_per_tick']:.3f} | "
        f"{act['patterns']} | {act['cofire']:.3f} | {act['inhibitory_spikes']} | "
        f"{silent} | {arm['decision_agreement_vs_ff_shipped']:.3f} |"
    )


def render_report(comparison: dict) -> str:
    """Deterministic markdown view of ``comparison.json`` (no timing)."""
    inputs = comparison["input"]
    sel = comparison["selection"]
    lines = [
        f"# {EXPERIMENT_ID}: recurrence vs the frozen exp-025 baseline",
        "",
        f"Input `{inputs['path']}` ({inputs['sha256']}); k = "
        f"{comparison['protocol']['k']}, I-drive = {comparison['protocol']['i_drive']}. "
        f"Splits: train {inputs['splits']['train']['n_ticks']} ticks / "
        f"{len(inputs['splits']['train']['sessions'])} sessions, "
        f"val {inputs['splits']['val']['n_ticks']} / "
        f"{len(inputs['splits']['val']['sessions'])}, "
        f"test {inputs['splits']['test']['n_ticks']} / "
        f"{len(inputs['splits']['test']['sessions'])}.",
        "",
        f"Primary = {comparison['protocol']['primary']}.",
        "",
        "| arm | gain | lambda_rel | val primary | test primary | "
        + " | ".join(f"test MAE {h}" for h in HEADS)
        + " | spk/tick | patterns | cofire | I spikes | silent | decision agreement vs ff-shipped |",
        "|---|---:|---:|---:|---:|" + "---:|" * N_HEADS + "---:|---:|---:|---:|---|---:|",
    ]
    for name in comparison["arm_order"]:
        lines.append(_arm_row(name, comparison["arms"][name]))
    lines.extend(["", "## Baselines (float, not quantized)", "",
                  "| baseline | val primary | test primary | "
                  + " | ".join(f"test MAE {h}" for h in HEADS) + " |",
                  "|---|---:|---:|" + "---:|" * N_HEADS])
    for name, base in comparison["baselines"].items():
        heads = " | ".join(_fmt(v) for v in base["test"]["session_weighted_mae"])
        lines.append(
            f"| `{name}` | {_fmt(base['val']['primary'])} | "
            f"{_fmt(base['test']['primary'])} | {heads} |"
        )
    verdict = sel["verdict_vs_ff_refit"]
    lines.extend(
        [
            "",
            "## Verdict",
            "",
            f"Validation-selected recurrent arm: `{sel['selected_rec_arm']}`.",
            f"Against `ff-refit`: primary improvement {verdict['primary_improvement']:+.2%}, "
            f"worst head degradation {verdict['worst_head_degradation']:+.2%}; "
            f"promising = **{verdict['promising']}**.",
            f"Against `ff-shipped`: primary improvement "
            f"{sel['verdict_vs_ff_shipped']['primary_improvement']:+.2%}.",
            _constant_baseline_line(comparison),
            "",
            f"Criterion (predeclared): {verdict['criterion']}.",
            "",
        ]
    )
    return "\n".join(lines)


def _constant_baseline_line(comparison: dict) -> str:
    """The relative verdict is not the whole story: say where the arms sit
    against predicting the train mean, which needs no network at all."""
    const = comparison["baselines"]["constant-train-mean"]["test"]["primary"]
    best = min(
        comparison["arm_order"],
        key=lambda name: comparison["arms"][name]["test"]["primary"],
    )
    best_primary = comparison["arms"][best]["test"]["primary"]
    relation = "beats" if best_primary < const else "does not beat"
    return (
        f"Best arm on test is `{best}` at {best_primary:.4f}; it {relation} the "
        f"`constant-train-mean` baseline ({const:.4f})."
    )
