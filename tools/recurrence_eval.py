#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""exp-026: evaluate recurrence against the frozen exp-025 baseline (GH #3).

One offline command that, on identical held-out sessions:

1. attests the shipped bank exactly as ``replay_frozen`` does and refuses a
   base checkpoint that already declares recurrence;
2. steps the feed-forward exp-025 bank (``ff``) and the
   ``spikenaut.recurrence.v1`` I->E lateral-inhibition variant at each gain
   (``rec-g<gain>``) over the train, val and test splits, resetting at every
   ``episode_id`` boundary, with the checkpoint's K-WTA and I-drive;
3. scores the shipped 16x3 head as-is (``ff-shipped``) and refits a
   Dale-constrained ridge head for every arm on the train split only,
   choosing the ridge strength on the validation split only;
4. scores the test split: session-weighted MAE per head and the primary
   score, pooled MAE/RMSE, health-style activity, decision agreement with
   the frozen reference;
5. scores three baselines (train-mean constant, persistence, memoryless
   ridge on the five inputs), selects the recurrent gain on validation and
   applies the predeclared verdict (``recurrence_core.promising``);
6. writes ``comparison.json`` and ``report.md`` (deterministic: no
   wall-clock fields) plus ``timing.json`` (host stepper wall-clock, never
   FPGA latency); ``--emit-checkpoint DIR`` additionally writes the
   validation-selected variant as a separately named checkpoint bundle.

The default input is the committed synthetic method fixture
``tools/fixtures/recurrence/telemetry.jsonl``; it proves the pipeline
composes and is not a research result. The recorded exp-026 run used the
Hub v3 ``state_telemetry`` shards (``docs/recurrence/``).

Standard library only. Exit codes match ``replay_frozen.py``: 0 = the
comparison completed, 2 = an artifact could not be consumed.

Run from anywhere::

    python3 tools/recurrence_eval.py
    python3 -m tools.recurrence_eval
    python3 tools/recurrence_eval.py --jsonl PATH/state_telemetry.jsonl \\
        --out-dir out/ --emit-checkpoint out/bundle/
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:  # package import: `python3 -m tools.recurrence_eval`
    from .decision_core import DecisionError
    from .hamming_banks import bank_from_model, model_from_bytes
    from .hamming_encode import Sample, parse_jsonl_text, select_samples
    from .model_bank import AttestedEntry, BankError, ModelBank, load_model_bank
    from .q88_core import ParseError
    from .recurrence_bundle import render_report, write_bundle
    from .recurrence_core import (
        GAINS_DEFAULT,
        LAMBDA_GRID,
        READOUT_FIT_METHOD,
        SPLITS,
        ReadoutScorer,
        Span,
        activity,
        constant_predictor,
        decision_agreement,
        fit_dale_readout,
        fit_linear_inputs,
        flatten_readout,
        fmt_ratio,
        linear_predictions,
        metrics_from_predictions,
        normal_equations,
        pattern_stats,
        persistence_predictions,
        promising,
        readout_predictions,
        require_split,
        select_by_val,
        session_spans,
        sign_census,
        step_arm,
        target_argmax_agreement,
        targets_for,
    )
    from .recurrence_documents import comparison_json, provenance_json, timing_json
    from .recurrent_lif import (
        Recurrence,
        dale_flags,
        describe_recurrence,
        lateral_inhibition,
        recurrence_from_model,
    )
    from .replay_core import (
        REPO_ROOT,
        default_i_drive,
        default_k,
        manifest_json,
        model_on_numeric_grid,
        readout_from_model,
    )
    from .replay_inputs import select_entry, sha256_bytes
except ImportError:  # direct script: `python3 tools/recurrence_eval.py`
    from decision_core import DecisionError
    from hamming_banks import bank_from_model, model_from_bytes
    from hamming_encode import Sample, parse_jsonl_text, select_samples
    from model_bank import AttestedEntry, BankError, ModelBank, load_model_bank
    from q88_core import ParseError
    from recurrence_bundle import render_report, write_bundle
    from recurrence_core import (
        GAINS_DEFAULT,
        LAMBDA_GRID,
        READOUT_FIT_METHOD,
        SPLITS,
        ReadoutScorer,
        Span,
        activity,
        constant_predictor,
        decision_agreement,
        fit_dale_readout,
        fit_linear_inputs,
        flatten_readout,
        fmt_ratio,
        linear_predictions,
        metrics_from_predictions,
        normal_equations,
        pattern_stats,
        persistence_predictions,
        promising,
        readout_predictions,
        require_split,
        select_by_val,
        session_spans,
        sign_census,
        step_arm,
        target_argmax_agreement,
        targets_for,
    )
    from recurrence_documents import comparison_json, provenance_json, timing_json
    from recurrent_lif import (
        Recurrence,
        dale_flags,
        describe_recurrence,
        lateral_inhibition,
        recurrence_from_model,
    )
    from replay_core import (
        REPO_ROOT,
        default_i_drive,
        default_k,
        manifest_json,
        model_on_numeric_grid,
        readout_from_model,
    )
    from replay_inputs import select_entry, sha256_bytes

FIXTURE_JSONL = REPO_ROOT / "tools" / "fixtures" / "recurrence" / "telemetry.jsonl"
SHIPPED_MANIFEST = REPO_ROOT / "dataset" / "merged_v2" / "model_bank.json"
DEFAULT_VARIANT_ID = "merged_v2_rec_v1"
_K_UNSET = object()


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


def _read_bytes(path: Path) -> bytes:
    # NOSONAR pythonsecurity:S8707 -- --jsonl is the tool's explicit
    # user-chosen input, as for replay_frozen's inputs.
    if not path.is_file():
        raise ParseError(f"missing file: {path}")
    try:
        return path.read_bytes()  # NOSONAR
    except OSError as exc:
        raise ParseError(f"unreadable file {path}: {exc}") from exc


def load_splits(jsonl: Path) -> tuple[dict[str, list[Sample]], str]:
    """All three built-in splits, in file order, plus the input digest."""
    payload = _read_bytes(jsonl)
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ParseError(f"{jsonl.name}: file is not UTF-8 ({exc})") from exc
    records = parse_jsonl_text(text, jsonl.name)
    splits: dict[str, list[Sample]] = {}
    for split in SPLITS:
        samples = select_samples(records, split)
        require_split(samples, split)
        splits[split] = samples
    return splits, sha256_bytes(payload)


def prepare_base(entry: AttestedEntry) -> tuple[dict, dict]:
    """Raw and Q8.8-grid views of the attested feed-forward checkpoint."""
    raw = model_from_bytes(entry.checkpoint_bytes, entry.checkpoint_relative)
    if recurrence_from_model(raw, entry.checkpoint_relative) is not None:
        raise ParseError(
            f"{entry.id}: base checkpoint already declares recurrence; the "
            "comparison derives its variants from a feed-forward base"
        )
    grid = model_on_numeric_grid(raw, entry.numeric_format, entry.checkpoint_relative)
    return raw, grid


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AttestedBase:
    """The attested feed-forward checkpoint every arm derives from."""

    bank: ModelBank
    entry: AttestedEntry
    raw: dict
    grid: dict


@dataclass(frozen=True)
class EvalInput:
    """The encoded splits of one JSONL input, its digest and its label."""

    splits: dict[str, list[Sample]]
    sha256: str
    jsonl: Path
    label: str | None = None

    def __post_init__(self) -> None:
        # Stepping, timing and scoring divide by every split's tick count.
        for split in SPLITS:
            require_split(self.splits.get(split, []), split)


@dataclass(frozen=True)
class Protocol:
    """K-WTA, I-drive and the recurrent gains to compare."""

    k: int | None
    i_drive: float
    gains: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.gains:
            raise ParseError("the comparison needs at least one recurrent gain")


class Comparison:
    """Holds the stepped traces and scored arms for one configuration."""

    def __init__(self, base: AttestedBase, data: EvalInput, protocol: Protocol) -> None:
        self.bank = base.bank
        self.entry = base.entry
        self.raw = base.raw
        self.grid = base.grid
        self.splits = data.splits
        self.input_sha256 = data.sha256
        self.jsonl = data.jsonl
        self.input_label = data.label
        self.k = protocol.k
        self.i_drive = protocol.i_drive
        self.gains = protocol.gains
        self.flags = dale_flags(self.grid, self.entry.checkpoint_relative)
        self.lif = bank_from_model(self.grid, self.entry.id, self.i_drive)
        self.targets = {s: targets_for(self.splits[s]) for s in SPLITS}
        self.spans: dict[str, list[Span]] = {s: session_spans(self.splits[s]) for s in SPLITS}
        self.traces: dict[str, dict[str, Any]] = {}
        self.recurrences: dict[str, Recurrence | None] = {}
        self.arms: dict[str, dict] = {}
        self.arm_order: list[str] = []
        self.fitted: dict[str, list[list[float]]] = {}
        self.timing: dict[str, dict] = {}

    # -- stepping ---------------------------------------------------------

    def step_all(self) -> None:
        plan: list[tuple[str, float | None]] = [("ff", None)]
        plan.extend((f"rec-g{gain!r}", gain) for gain in self.gains)
        for name, gain in plan:
            recurrence = (
                None if gain is None
                else lateral_inhibition(self.grid, gain, self.entry.checkpoint_relative)
            )
            self.recurrences[name] = recurrence
            self.traces[name] = {}
            self.timing[name] = {}
            for split in SPLITS:
                trace = step_arm(self.lif, self.splits[split], self.k, recurrence)
                self.traces[name][split] = trace.masks
                self.timing[name][split] = {
                    "ticks": len(trace.masks),
                    "seconds": trace.seconds,
                    "us_per_tick": 1e6 * trace.seconds / len(trace.masks),
                }

    # -- scoring ----------------------------------------------------------

    def _metrics(self, masks: list[int], scorer: ReadoutScorer, split: str) -> dict:
        return metrics_from_predictions(
            readout_predictions(masks, scorer), self.targets[split], self.spans[split]
        )

    def _arm_entry(
        self,
        kind: str,
        stepped: str,
        readout: list[list[float]],
        scorer: ReadoutScorer,
        shipped_scorer: ReadoutScorer,
    ) -> dict:
        masks = self.traces[stepped]
        return {
            "kind": kind,
            "stepped_as": stepped,
            "gain": None if self.recurrences[stepped] is None else self._gain_of(stepped),
            "recurrence": describe_recurrence(self.recurrences[stepped]),
            "readout": {
                "weights_neuron_major": flatten_readout(readout),
                "sign_census": sign_census(readout, self.flags),
            },
            "val": self._metrics(masks["val"], scorer, "val"),
            "test": self._metrics(masks["test"], scorer, "test"),
            "activity_val": activity(masks["val"]),
            "activity_test": activity(masks["test"]),
            "decision_agreement_vs_ff_shipped": decision_agreement(
                masks["test"], scorer, self.traces["ff"]["test"], shipped_scorer
            ),
            "target_argmax_agreement_test": target_argmax_agreement(
                masks["test"], scorer, self.targets["test"]
            ),
        }

    def _gain_of(self, name: str) -> float:
        return self.gains[[f"rec-g{g!r}" for g in self.gains].index(name)]

    def score_all(self) -> None:
        shipped_flat = readout_from_model(self.grid)
        shipped = [shipped_flat[i * 3:(i + 1) * 3] for i in range(16)]
        shipped_scorer = ReadoutScorer(shipped_flat)
        entry = self._arm_entry("ff-shipped", "ff", shipped, shipped_scorer, shipped_scorer)
        entry["lambda_rel"] = None
        entry["readout"]["source"] = "shipped merged_v2 output_weights on the Q8.8 grid"
        self.arms["ff-shipped"] = entry
        self.arm_order.append("ff-shipped")
        n_train = len(self.splits["train"])
        for stepped in self.traces:
            name = "ff-refit" if stepped == "ff" else stepped
            gram, rhs = normal_equations(
                pattern_stats(self.traces[stepped]["train"], self.targets["train"])
            )
            sweep: dict[float, dict] = {}
            for lam_rel in LAMBDA_GRID:
                weights = fit_dale_readout(gram, rhs, lam_rel * n_train, self.flags)
                scorer = ReadoutScorer(flatten_readout(weights))
                sweep[lam_rel] = {
                    "weights": weights,
                    "scorer": scorer,
                    "val": self._metrics(self.traces[stepped]["val"], scorer, "val"),
                }
            lam_rel = select_by_val(sweep, key=lambda c: c["val"]["primary"])
            chosen = sweep[lam_rel]
            entry = self._arm_entry(
                "ff-refit" if stepped == "ff" else "rec",
                stepped, chosen["weights"], chosen["scorer"], shipped_scorer,
            )
            entry["lambda_rel"] = lam_rel
            entry["lambda_abs"] = lam_rel * n_train
            entry["lambda_sweep_val_primary"] = {
                repr(lam): sweep[lam]["val"]["primary"] for lam in LAMBDA_GRID
            }
            entry["readout"]["source"] = READOUT_FIT_METHOD
            self.arms[name] = entry
            self.arm_order.append(name)
            self.fitted[name] = chosen["weights"]

    def baselines(self) -> dict:
        train_t, val_t, test_t = (self.targets[s] for s in SPLITS)
        const = constant_predictor(train_t)
        out: dict[str, dict] = {
            "constant-train-mean": {
                "val": metrics_from_predictions([const] * len(val_t), val_t, self.spans["val"]),
                "test": metrics_from_predictions([const] * len(test_t), test_t, self.spans["test"]),
                "prediction": const,
            },
            "persistence": {
                "val": metrics_from_predictions(
                    persistence_predictions(val_t, self.spans["val"], const), val_t, self.spans["val"]
                ),
                "test": metrics_from_predictions(
                    persistence_predictions(test_t, self.spans["test"], const), test_t, self.spans["test"]
                ),
                "note": "previous tick's target within the session; first tick uses the train mean",
            },
        }
        n_train = len(train_t)
        sweep: dict[float, dict] = {}
        for lam_rel in LAMBDA_GRID:
            weights = fit_linear_inputs(self.splits["train"], train_t, lam_rel * n_train)
            sweep[lam_rel] = {
                "weights": weights,
                "val": metrics_from_predictions(
                    linear_predictions(self.splits["val"], weights), val_t, self.spans["val"]
                ),
            }
        lam_rel = select_by_val(sweep, key=lambda c: c["val"]["primary"])
        out["linear-inputs-ridge"] = {
            "lambda_rel": lam_rel,
            "weights_head_major": sweep[lam_rel]["weights"],
            "val": sweep[lam_rel]["val"],
            "test": metrics_from_predictions(
                linear_predictions(self.splits["test"], sweep[lam_rel]["weights"]),
                test_t, self.spans["test"],
            ),
            "note": (
                "memoryless ridge on the five live axons plus bias, float; the "
                "target is a function of two of those axons, so this is the ceiling"
            ),
        }
        return out

    def selection(self) -> dict:
        rec = {n: a for n, a in self.arms.items() if a["kind"] == "rec"}
        selected = min(rec, key=lambda n: (rec[n]["val"]["primary"], rec[n]["gain"]))
        return {
            "selected_rec_arm": selected,
            "selected_gain": rec[selected]["gain"],
            "selected_on": "validation primary (ties: smaller gain)",
            "verdict_vs_ff_refit": promising(rec[selected]["test"], self.arms["ff-refit"]["test"]),
            "verdict_vs_ff_shipped": promising(rec[selected]["test"], self.arms["ff-shipped"]["test"]),
            "val_primary_by_arm": {n: self.arms[n]["val"]["primary"] for n in self.arm_order},
            "test_primary_by_arm": {n: self.arms[n]["test"]["primary"] for n in self.arm_order},
        }

    # -- documents --------------------------------------------------------

    def comparison_document(self) -> dict:
        return comparison_json(self)

    def timing_document(self) -> dict:
        return timing_json(self)

    def bundle_provenance(self, selected: str) -> dict:
        return provenance_json(self, selected)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _gains(text: str) -> tuple[float, ...]:
    try:
        values = tuple(float(part) for part in text.split(",") if part.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"--gains: {exc}") from exc
    if not values:
        raise argparse.ArgumentTypeError("--gains must list at least one gain")
    return values


def _kwta(text: str) -> int | None:
    if text == "none":
        return None
    try:
        value = int(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"--k must be an integer or 'none', got {text!r}") from exc
    if value < 0:
        raise argparse.ArgumentTypeError(f"--k must be non-negative, got {text!r}")
    return value


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="recurrence_eval", description=__doc__.split("\n\n", 1)[0]
    )
    parser.add_argument("--bank-manifest", type=Path, default=None,
                        help="model_bank.json to attest (default: shipped dataset/merged_v2)")
    parser.add_argument("--model-id", default=None,
                        help="feed-forward base entry (default: the sole attested entry)")
    parser.add_argument("--jsonl", type=Path, default=None,
                        help="v3 state_telemetry JSONL with train, val and test episodes "
                             "(default: the synthetic tools/fixtures/recurrence fixture)")
    parser.add_argument("--input-label", default=None, metavar="TEXT",
                        help="how comparison.json names the input (default: its repo-relative "
                             "or absolute path); the sha256 is recorded regardless")
    parser.add_argument("--gains", type=_gains, default=GAINS_DEFAULT, metavar="G,G,...",
                        help=f"I->E inhibition gains on the Q8.8 grid (default {','.join(map(repr, GAINS_DEFAULT))})")
    parser.add_argument("--k", type=_kwta, default=_K_UNSET, metavar="N|none",
                        help="K-WTA width (default: the checkpoint's k_wta; 'none' disables)")
    parser.add_argument("--i-drive", type=float, default=None,
                        help="Dale I bias (default: the checkpoint's exp023_knobs.I_DRIVE)")
    parser.add_argument("--out-dir", type=Path, default=Path("recurrence_out"),
                        help="directory for comparison.json, report.md, timing.json")
    parser.add_argument("--emit-checkpoint", type=Path, default=None, metavar="DIR",
                        help="also write the validation-selected variant as a checkpoint bundle")
    parser.add_argument("--variant-id", default=DEFAULT_VARIANT_ID,
                        help=f"model id for the emitted bundle (default {DEFAULT_VARIANT_ID})")
    return parser


def _publish(out_dir: Path, files: dict[str, bytes]) -> None:
    """Write ``files`` into ``out_dir`` as one set: stage under temp names,
    then rename, so an interrupted run never leaves a new ``comparison.json``
    beside a stale ``report.md`` (the ``replay_frozen`` discipline). On
    failure the whole set is removed rather than left half-new."""
    # NOSONAR pythonsecurity:S8707 -- --out-dir is the tool's explicit
    # user-chosen destination, as for replay_frozen.
    out_dir.mkdir(parents=True, exist_ok=True)  # NOSONAR
    staged = {name: out_dir / f"{name}.tmp" for name in files}
    try:
        for name, data in files.items():
            staged[name].write_bytes(data)  # NOSONAR
        for name in files:
            os.replace(staged[name], out_dir / name)
    except OSError:
        for name in files:
            staged[name].unlink(missing_ok=True)
            (out_dir / name).unlink(missing_ok=True)
        raise


def run(args: argparse.Namespace) -> tuple[Comparison, dict]:
    if args.emit_checkpoint is not None and (args.k is not _K_UNSET or args.i_drive is not None):
        # The bundle keeps the base checkpoint's k_wta and I_DRIVE, and the
        # replay steps it with those, so its readout must be fit with them.
        raise ParseError(
            "--emit-checkpoint fits the bundle's readout with the checkpoint's "
            "own k_wta and I_DRIVE; drop --k and --i-drive to emit a bundle"
        )
    manifest = args.bank_manifest if args.bank_manifest is not None else SHIPPED_MANIFEST
    jsonl = args.jsonl if args.jsonl is not None else FIXTURE_JSONL
    bank = load_model_bank(manifest)
    entry = select_entry(bank, args.model_id)
    raw, grid = prepare_base(entry)
    splits, input_sha256 = load_splits(jsonl)
    k = default_k(grid) if args.k is _K_UNSET else args.k
    i_drive = default_i_drive(grid) if args.i_drive is None else args.i_drive
    comparison = Comparison(
        AttestedBase(bank=bank, entry=entry, raw=raw, grid=grid),
        EvalInput(splits=splits, sha256=input_sha256, jsonl=jsonl, label=args.input_label),
        Protocol(k=k, i_drive=i_drive, gains=args.gains),
    )
    comparison.step_all()
    comparison.score_all()
    document = comparison.comparison_document()
    return comparison, document


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        comparison, document = run(args)
        _publish(args.out_dir, {
            "comparison.json": manifest_json(document),
            "report.md": render_report(document).encode("utf-8"),
            "timing.json": manifest_json(comparison.timing_document()),
        })
        if args.emit_checkpoint is not None:
            selected = document["selection"]["selected_rec_arm"]
            written = write_bundle(
                args.emit_checkpoint,
                model_id=args.variant_id,
                base_model=comparison.raw,
                recurrence=comparison.recurrences[selected],
                readout=comparison.fitted[selected],
                provenance=comparison.bundle_provenance(selected),
                training_dataset_digest=comparison.input_sha256,
            )
            _publish(args.emit_checkpoint, {"bundle_digests.json": manifest_json(written)})
    except (BankError, ParseError, DecisionError) as exc:
        print(f"recurrence_eval: {exc}", file=sys.stderr)
        return 2
    sel = document["selection"]
    verdict = sel["verdict_vs_ff_refit"]
    print(
        f"recurrence_eval: {len(document['arm_order'])} arms on "
        f"{document['input']['splits']['test']['n_ticks']} test ticks; selected "
        f"{sel['selected_rec_arm']} (val); vs ff-refit primary "
        f"{fmt_ratio(verdict['primary_improvement'])}, worst head "
        f"{fmt_ratio(verdict['worst_head_degradation'])}, promising={verdict['promising']}; "
        f"wrote {args.out_dir}/comparison.json + report.md + timing.json"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
