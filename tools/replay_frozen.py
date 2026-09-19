#!/usr/bin/env python3
"""Deterministic frozen model-bank replay (Linear RM-1692 / GH #60).

One offline command that composes, in order:

1. attested model-bank selection (``tools/model_bank.py`` -- the
   checkpoint is consumed as the bytes its digest covered, never
   re-read);
2. five-sensor analog encoding (``tools/hamming_encode.py`` -- the
   exp-025 live map, frozen minmax lineage 74acdd0f, axons 5-15 at 0);
3. reference keep-LIF stepping (``tools/hamming_lif.py``
   ``keep_lif_step``, resetting at each ``episode_id`` boundary);
4. readout scoring and decision diagnostics (``tools/decision_core.py``
   ``replay_output_row`` over the shipped comfort/temp/power head).

Outputs (under ``--out-dir``):

* ``trace.jsonl`` -- one JSON object per step: session, source line,
  per-sensor ``missing`` names, stim, fired neurons, scores, decision
  diagnostics;
* ``manifest.json`` -- reproducibility manifest: checkpoint and input
  digests, feature/output contract identifiers, frozen-minmax
  normalization reference, session identifiers, configuration, source
  revision, and the trace digest. No wall-clock fields: both artifacts
  are byte-identical across runs on the same inputs.

Default invocation replays the committed method fixture
(``tools/fixtures/replay_frozen/telemetry.jsonl`` -- synthetic v3-shaped
rows, labeled synthetic; not measured hardware, not a holdout result)
against the shipped ``dataset/merged_v2`` bank. For real-session
qualification pass explicit ``--jsonl`` (and optionally
``--split-manifest``) paths.

Missing / stale inputs: a live column absent or null encodes to 0 (the
Distill ``T=0 stays 0`` semantics the bank was trained with) *and* is
named per step in the trace's ``missing`` list, so it stays
distinguishable from an observed zero; ``--missing-policy reject``
refuses such rows instead. Staleness is undetectable on v3
``state_telemetry`` (``ts_utc`` is null) -- rows replay in file order.
See ``tools/replay_core.py`` for the full policy text.

Standard library only. Exit codes match ``verify_q88.py``:
0 = replay completed, 2 = an artifact could not be consumed
(attestation, parse, or contract refusal), so nothing was replayed.

Run from anywhere::

    python3 tools/replay_frozen.py
    python3 -m tools.replay_frozen
    python3 tools/replay_frozen.py --jsonl PATH/state_telemetry.jsonl \\
        --split test --out-dir out/
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

try:  # package import: `python3 -m tools.replay_frozen`
    from .decision_core import DecisionError
    from .model_bank import AttestedEntry, BankError
    from .q88_core import ParseError
    from .replay_core import (
        MISSING_POLICIES,
        ReplayConfig,
        build_manifest,
        default_k,
        fixture_jsonl,
        load_replay_inputs,
        manifest_json,
        model_from_bytes,
        replay,
        shipped_manifest_path,
        trace_jsonl,
    )
except ImportError:  # direct script: `python3 tools/replay_frozen.py`
    from decision_core import DecisionError
    from model_bank import AttestedEntry, BankError
    from q88_core import ParseError
    from replay_core import (
        MISSING_POLICIES,
        ReplayConfig,
        build_manifest,
        default_k,
        fixture_jsonl,
        load_replay_inputs,
        manifest_json,
        model_from_bytes,
        replay,
        shipped_manifest_path,
        trace_jsonl,
    )


def _finite_float(text: str) -> float:
    """argparse type: a finite float. Rejects nan / inf / -inf."""
    try:
        value = float(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"invalid float value: {text!r}"
        ) from exc
    if not math.isfinite(value):
        raise argparse.ArgumentTypeError(
            f"--i-drive must be a finite number, got {text!r}"
        )
    return value


def _kwta(text: str) -> int | None:
    """argparse type: a non-negative K-WTA width, or ``none`` to disable."""
    if text == "none":
        return None
    try:
        value = int(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"--k must be a non-negative integer or 'none', got {text!r}"
        ) from exc
    if value < 0:
        raise argparse.ArgumentTypeError(
            f"--k must be a non-negative integer or 'none', got {text!r}"
        )
    return value


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="replay_frozen",
        description=__doc__.split("\n\n", 1)[0],
    )
    _add_input_args(parser)
    _add_run_args(parser)
    return parser


def _add_input_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--bank-manifest",
        type=Path,
        default=None,
        help=(
            "model_bank.json to attest (default: the shipped "
            "dataset/merged_v2 bank)"
        ),
    )
    parser.add_argument(
        "--model-id",
        default=None,
        help="bank entry to replay (default: the sole attested entry)",
    )
    parser.add_argument(
        "--jsonl",
        type=Path,
        default=None,
        help=(
            "v3 state_telemetry JSONL (default: the committed "
            "tools/fixtures/replay_frozen/telemetry.jsonl method fixture -- "
            "synthetic rows, not measured hardware)"
        ),
    )
    parser.add_argument(
        "--split",
        choices=("train", "val", "test", "all"),
        default="all",
        help=(
            "episodes to replay: built-in ranges, or the named split when "
            "--split-manifest is supplied (default: all)"
        ),
    )
    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=None,
        help=(
            "optional spikenaut.split-manifest.v1 JSON assigning episodes to "
            "splits for recorded sessions outside the built-in ranges; an "
            "episode in two splits is rejected"
        ),
    )


def _add_run_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--k",
        type=_kwta,
        default=None,
        metavar="N|none",
        help=(
            "K-WTA width (default: the checkpoint's recorded k_wta; "
            "'none' disables it)"
        ),
    )
    parser.add_argument(
        "--i-drive",
        type=_finite_float,
        default=0.0,
        help="Dale I bias added to neurons 12-15 (default 0.0)",
    )
    parser.add_argument(
        "--missing-policy",
        choices=MISSING_POLICIES,
        default="encode-zero",
        help=(
            "'encode-zero' (default): missing/null sensors encode to 0 and "
            "are named per step; 'reject': refuse rows with any missing "
            "live sensor"
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("replay_out"),
        help="directory for trace.jsonl + manifest.json (default replay_out/)",
    )


def _resolve_k(args: argparse.Namespace, entry: AttestedEntry) -> int | None:
    if args.k is not None:
        return args.k
    model = model_from_bytes(entry.checkpoint_bytes, entry.checkpoint_relative)
    return default_k(model)


def _write_artifacts(
    out_dir: Path, trace_bytes: bytes, manifest: dict
) -> None:
    # NOSONAR pythonsecurity:S8707 -- --out-dir is the tool's explicit
    # user-chosen destination; constraining it would break the documented
    # explicit-path qualification workflow.
    out_dir.mkdir(parents=True, exist_ok=True)  # NOSONAR
    (out_dir / "trace.jsonl").write_bytes(trace_bytes)  # NOSONAR
    (out_dir / "manifest.json").write_bytes(manifest_json(manifest))  # NOSONAR


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    bank_manifest = (
        args.bank_manifest
        if args.bank_manifest is not None
        else shipped_manifest_path()
    )
    jsonl = args.jsonl if args.jsonl is not None else fixture_jsonl()
    try:
        bank, entry, samples = load_replay_inputs(
            bank_manifest, args.model_id, jsonl, args.split,
            args.split_manifest,
        )
        config = ReplayConfig(
            split=args.split,
            k=_resolve_k(args, entry),
            i_drive=args.i_drive,
            missing_policy=args.missing_policy,
        )
        result = replay(entry, samples, config)
        trace_bytes = trace_jsonl(result.trace_rows)
        manifest = build_manifest(
            entry=entry,
            bank=bank,
            jsonl=jsonl,
            split_manifest_path=args.split_manifest,
            config=config,
            result=result,
            trace_bytes=trace_bytes,
        )
        _write_artifacts(args.out_dir, trace_bytes, manifest)
    except (BankError, ParseError, DecisionError) as exc:
        print(f"replay_frozen: {exc}", file=sys.stderr)
        return 2
    print(
        f"replay_frozen: {len(result.trace_rows)} steps over "
        f"{len(result.sessions)} session(s), {result.spikes_fired} spikes; "
        f"model {entry.id} ({entry.checkpoint_digest}); "
        f"k={'none' if config.k is None else config.k}, "
        f"missing_policy={config.missing_policy}; "
        f"wrote {args.out_dir}/trace.jsonl + manifest.json"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
