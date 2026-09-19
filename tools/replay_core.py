"""Frozen model-bank replay core (Linear RM-1692 / GH #60).

Composes the pieces that already exist -- attested model-bank selection
(``model_bank``), the exp-025 five-sensor analog encoder
(``hamming_encode``), the reference keep-LIF stepper (``hamming_lif``) and
the output-row decision contract (``decision_core``) -- into one offline
replay that emits a per-step trace plus a reproducibility manifest.

Determinism contract
--------------------
* The checkpoint is consumed as the exact bytes the bank attested
  (``AttestedEntry.checkpoint_bytes``); nothing re-reads the file, so a
  swapped checkpoint cannot bypass the digest.
* All float math runs through ``f32`` (binary32) inside the existing
  stepper; this module adds no new arithmetic.
* Manifest and trace contents depend only on inputs. Wall-clock timing is
  not recorded in either artifact; callers that want timing keep it
  outside the deterministic documents.

Missing / stale input policy
----------------------------
A live column absent or ``null`` on a v3 row encodes to ``0.0`` under the
frozen-minmax contract (``frozen_unit01``), matching the Distill ``T=0
stays 0`` semantics the bank was trained with. That zero is a *missing
measurement*, not an observed idle sensor, so each trace row carries the
per-sensor ``missing`` names -- a count alone would make missing values
indistinguishable from observed zeros. ``missing_policy='reject'``
refuses such rows instead. The v3 schema has no usable timestamp
(``ts_utc`` is null), so *staleness* is undetectable in replay: rows are
consumed in file order and the policy is documented as none. Callers
needing staleness must supply data whose schema supports it.

Sessions
--------
The session key is ``episode_id`` (``gpu-######``), same as the rest of
the harness: membrane state resets at every episode boundary, and
interleaved episodes are refused because a reset needs grouped rows.
An optional split manifest declares episodes per split explicitly, for
recorded sessions outside the built-in train/val/test episode ranges;
an episode listed in two splits is an overlap and is rejected.

Standard library only.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:  # package import: `python3 -m tools.replay_frozen`
    from .decision_core import (
        OUTPUT_WIDTH,
        SHIPPED_VOCABULARY,
        decision_as_dict,
        replay_output_row,
        score_readout,
    )
    from .hamming_banks import bank_from_model, model_from_bytes
    from .hamming_const import (
        FROZEN_LINEAGE,
        FROZEN_MINMAX,
        I_DRIVE_EXP024,
        LIVE_COLUMNS,
        N_NEURONS,
        SHIPPED_DIR,
        f32,
    )
    from .hamming_encode import (
        Sample,
        episode_index,
        parse_jsonl_text,
        select_samples,
    )
    from .hamming_lif import LifBank, keep_lif_step
    from .model_bank import (
        FEATURE_MAP_LIVE_EXP_025,
        OUTPUT_CONTRACT_SUPERVISOR_V3_RM1150,
        AttestedEntry,
        ModelBank,
        load_model_bank,
    )
    from .model_bank_json import _parse_json, _read_utf8
    from .q88_core import (
        ParseError,
        Q88RangeError,
        as_finite_float,
        decode_q88,
        encode_q88,
    )
except ImportError:  # direct script: `python3 tools/replay_frozen.py`
    from decision_core import (
        OUTPUT_WIDTH,
        SHIPPED_VOCABULARY,
        decision_as_dict,
        replay_output_row,
        score_readout,
    )
    from hamming_banks import bank_from_model, model_from_bytes
    from hamming_const import (
        FROZEN_LINEAGE,
        FROZEN_MINMAX,
        I_DRIVE_EXP024,
        LIVE_COLUMNS,
        N_NEURONS,
        SHIPPED_DIR,
        f32,
    )
    from hamming_encode import (
        Sample,
        episode_index,
        parse_jsonl_text,
        select_samples,
    )
    from hamming_lif import LifBank, keep_lif_step
    from model_bank import (
        FEATURE_MAP_LIVE_EXP_025,
        OUTPUT_CONTRACT_SUPERVISOR_V3_RM1150,
        AttestedEntry,
        ModelBank,
        load_model_bank,
    )
    from model_bank_json import _parse_json, _read_utf8
    from q88_core import (
        ParseError,
        Q88RangeError,
        as_finite_float,
        decode_q88,
        encode_q88,
    )

REPO_ROOT = Path(__file__).resolve().parent.parent
REPLAY_FIXTURE_DIR = REPO_ROOT / "tools" / "fixtures" / "replay_frozen"

TRACE_SCHEMA = "spikenaut.replay-trace.v1"
MANIFEST_SCHEMA = "spikenaut.replay-manifest.v1"
SPLIT_MANIFEST_SCHEMA = "spikenaut.split-manifest.v1"

MISSING_POLICY_ENCODE_ZERO = "encode-zero"
MISSING_POLICY_REJECT = "reject"
MISSING_POLICIES = (MISSING_POLICY_ENCODE_ZERO, MISSING_POLICY_REJECT)

SPLIT_NAMES = ("train", "val", "test")


def sha256_bytes(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def readout_from_model(model: dict) -> list[float]:
    """Neuron-major 16x3 readout weights for ``decision_core.score_readout``.

    ``_validate_neuron`` (run by ``model_from_bytes``) guarantees the
    neuron container shape and that every listed field is a finite real;
    this layer additionally refuses a missing or wrongly-sized
    ``output_weights`` row rather than letting a short row shift the
    channel assignment.
    """
    flat: list[float] = []
    for i, neuron in enumerate(model["neurons"]):
        row = neuron.get("output_weights")
        if not isinstance(row, list) or len(row) != OUTPUT_WIDTH:
            got = (
                type(row).__name__ if not isinstance(row, list) else len(row)
            )
            raise ParseError(
                f"neurons[{i}].output_weights: {got} entries, "
                f"expected {OUTPUT_WIDTH}"
            )
        for j, value in enumerate(row):
            flat.append(_readout_weight(i, j, value))
    return flat


def _readout_weight(i: int, j: int, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ParseError(
            f"neurons[{i}].output_weights[{j}]: expected a finite "
            f"number, got {type(value).__name__} {value!r}"
        )
    if not math.isfinite(value):
        raise ParseError(
            f"neurons[{i}].output_weights[{j}]: non-finite {value!r}"
        )
    return float(value)


def load_split_manifest(path: Path) -> dict[str, frozenset[str]]:
    """Parse an explicit episode->split declaration.

    Shape::

        {"schema_version": "spikenaut.split-manifest.v1",
         "splits": {"train": ["gpu-000001", ...], "val": [...], "test": [...]}}

    Every entry must be a ``gpu-######`` episode id. An episode appearing
    in two splits is a split overlap and is refused -- assigning one
    session to both train and test would quietly contaminate the holdout.
    """
    if not path.is_file():
        raise ParseError(f"missing split manifest: {path}")
    return _split_manifest_entries(path, _read_utf8(path, kind="split manifest"))


def _split_manifest_entries(path: Path, text: str) -> dict[str, frozenset[str]]:
    """Validate the already-read manifest text into split -> episodes."""
    splits = _split_manifest_doc(path, text)
    owner: dict[str, str] = {}
    out: dict[str, frozenset[str]] = {}
    for name, episodes in splits.items():
        if name not in SPLIT_NAMES:
            raise ParseError(
                f"{path.name}: unknown split {name!r} "
                f"(expected one of {', '.join(SPLIT_NAMES)})"
            )
        if not isinstance(episodes, list):
            raise ParseError(
                f"{path.name}: splits[{name!r}] must be an array, got "
                f"{type(episodes).__name__}"
            )
        out[name] = _split_members(path, name, episodes, owner)
    return out


def _split_manifest_doc(path: Path, text: str) -> dict:
    """Validate the manifest text and return its ``splits`` object."""
    document = _parse_json(text, source=path)
    if not isinstance(document, dict):
        raise ParseError(
            f"{path.name}: top level must be an object, got "
            f"{type(document).__name__}"
        )
    version = document.get("schema_version")
    if version != SPLIT_MANIFEST_SCHEMA:
        raise ParseError(
            f"{path.name}: schema_version must be {SPLIT_MANIFEST_SCHEMA!r}, "
            f"got {version!r}"
        )
    splits = document.get("splits")
    if not isinstance(splits, dict) or not splits:
        raise ParseError(
            f"{path.name}: 'splits' must be a non-empty object, got "
            f"{type(splits).__name__}"
        )
    return splits


def _split_members(
    path: Path, name: str, episodes: list, owner: dict[str, str]
) -> frozenset[str]:
    members: set[str] = set()
    for raw in episodes:
        if not isinstance(raw, str) or episode_index(raw) is None:
            raise ParseError(
                f"{path.name}: splits[{name!r}] entry {raw!r} is not a "
                "gpu-###### episode id"
            )
        if raw in members:
            raise ParseError(
                f"{path.name}: episode {raw} listed twice in split "
                f"{name!r}"
            )
        previous = owner.get(raw)
        if previous is not None:
            raise ParseError(
                f"{path.name}: split overlap -- episode {raw} is in both "
                f"{previous!r} and {name!r}"
            )
        members.add(raw)
        owner[raw] = name
    return frozenset(members)


def select_replay_samples(
    records: list[tuple[int, dict]],
    split: str,
    split_manifest: dict[str, frozenset[str]] | None,
) -> list[Sample]:
    """Select the rows to replay, in file order.

    Without a manifest, ``split`` uses the built-in episode ranges via
    ``select_samples``. With a manifest, membership decides: ``split``
    names the manifest split to replay and ``all`` replays every listed
    episode. ``select_samples('all')`` still runs first, so the v3 row
    refusals (non-telemetry, forbidden derived sensors, malformed
    ``episode_id``, interleaved episodes) apply unchanged either way.
    """
    if split_manifest is None:
        samples = select_samples(records, split)
    else:
        members = _manifest_members(split_manifest, split)
        all_samples = select_samples(records, "all")
        samples = [s for s in all_samples if s.episode_id in members]
    if not samples:
        raise ParseError(
            f"NOTHING WAS REPLAYED: 0 steps after split={split!r}"
            + (" (split manifest supplied)" if split_manifest else "")
        )
    return samples


def _manifest_members(
    split_manifest: dict[str, frozenset[str]], split: str
) -> set[str]:
    """Episodes the manifest assigns to ``split``; ``all`` means every one."""
    if split == "all":
        return set().union(*split_manifest.values())
    if split not in split_manifest:
        raise ParseError(
            f"split {split!r} is not declared in the split manifest "
            f"(declares {', '.join(sorted(split_manifest))})"
        )
    return set(split_manifest[split])


@dataclass(frozen=True)
class ReplayConfig:
    """Knobs recorded on the manifest."""

    split: str
    k: int | None
    i_drive: float
    missing_policy: str


@dataclass
class ReplayResult:
    """The two deterministic artifacts plus a human-facing summary."""

    trace_rows: list[dict]
    sessions: list[str]
    steps_per_session: dict[str, int]
    missing_counts: dict[str, int]
    spikes_fired: int


def _require_missing_policy(policy: str) -> None:
    if policy not in MISSING_POLICIES:
        raise ParseError(
            f"unknown missing policy {policy!r} "
            f"(expected {'|'.join(MISSING_POLICIES)})"
        )


def _require_samples_missing_policy(
    samples: list[Sample], policy: str
) -> None:
    if policy != MISSING_POLICY_REJECT:
        return
    bad = [s for s in samples if s.missing]
    if not bad:
        return
    first = bad[0]
    raise ParseError(
        f"line {first.source_line}: missing live sensor(s) "
        f"{', '.join(first.missing)} and missing_policy='reject' -- "
        "a missing measurement is not an observed zero"
    )


def replay(
    entry: AttestedEntry,
    samples: list[Sample],
    config: ReplayConfig,
) -> ReplayResult:
    """Step the attested checkpoint through the encoded sessions.

    The bank is built from ``entry.checkpoint_bytes`` -- the bytes the
    digest covered -- so the replay is bound to attestation by
    construction, and the parameters are consumed on the grid
    ``entry.numeric_format`` declares (Q8.8 for the shipped bank), so
    the trace matches the deployed ``.mem`` decoding. ``entry`` bytes
    and parsed parameters are only read, never mutated.
    """
    _require_missing_policy(config.missing_policy)
    _require_samples_missing_policy(samples, config.missing_policy)
    model = model_on_numeric_grid(
        model_from_bytes(entry.checkpoint_bytes, entry.checkpoint_relative),
        entry.numeric_format,
        entry.checkpoint_relative,
    )
    bank = bank_from_model(model, entry.id, config.i_drive)
    return _run_steps(bank, readout_from_model(model), samples, config)


def _q88(value: Any, where: str) -> float:
    """Round-trip ``value`` through the Q8.8 codec (the .mem decode)."""
    try:
        return f32(decode_q88(encode_q88(as_finite_float(value, where))))
    except Q88RangeError as exc:
        raise ParseError(str(exc)) from exc


def model_on_numeric_grid(
    model: dict, numeric_format: str, source: str
) -> dict:
    """Parameters on the entry's declared consumption grid.

    The checkpoint stores float JSON; ``numeric_format`` names the grid
    the bank consumes them on. ``q8.8-fixed-point`` round-trips every
    hidden weight, keep-factor, threshold and readout weight through
    the codec the deployed ``.mem`` images decode with -- replaying the
    raw floats would produce a different model than the one the digest
    and format claim. An unknown format is refused rather than
    silently replayed in float.
    """
    if numeric_format != "q8.8-fixed-point":
        raise ParseError(
            f"{source}: numeric_format {numeric_format!r} has no replay "
            "implementation (supported: 'q8.8-fixed-point')"
        )
    neurons = []
    for i, neuron in enumerate(model["neurons"]):
        where = f"neurons[{i}]"
        q = dict(neuron)
        q["weights"] = [
            _q88(w, f"{where}.weights[{j}]")
            for j, w in enumerate(neuron["weights"])
        ]
        q["decay_rate"] = _q88(neuron["decay_rate"], f"{where}.decay_rate")
        q["threshold"] = _q88(neuron["threshold"], f"{where}.threshold")
        row = neuron.get("output_weights")
        if isinstance(row, list):
            q["output_weights"] = [
                _q88(v, f"{where}.output_weights[{j}]")
                for j, v in enumerate(row)
            ]
        neurons.append(q)
    out = dict(model)
    out["neurons"] = neurons
    return out


def _run_steps(
    bank: LifBank,
    readout: list[float],
    samples: list[Sample],
    config: ReplayConfig,
) -> ReplayResult:
    """The stepping loop: one trace row per sample, reset per session."""
    rows: list[dict] = []
    sessions: list[str] = []
    steps_per_session: dict[str, int] = {}
    missing_counts: dict[str, int] = dict.fromkeys(LIVE_COLUMNS, 0)
    fired_total = 0
    prev: str | None = None
    bank.reset()
    for step, sample in enumerate(samples):
        if sample.episode_id != prev:
            bank.reset()
            prev = sample.episode_id
            sessions.append(sample.episode_id)
            steps_per_session[sample.episode_id] = 0
        steps_per_session[sample.episode_id] += 1
        spikes = keep_lif_step(bank, list(sample.stim), config.k)
        scores = score_readout(readout, spikes)
        fired_total += sum(1 for s in spikes if s)
        for column in sample.missing:
            missing_counts[column] += 1
        rows.append(
            {
                "step": step,
                "session": sample.episode_id,
                "source_line": sample.source_line,
                "missing": list(sample.missing),
                "stim": list(sample.stim),
                "spikes": [i for i, s in enumerate(spikes) if s],
                "scores": list(scores),
                "decision": decision_as_dict(replay_output_row(scores)),
            }
        )
    return ReplayResult(
        trace_rows=rows,
        sessions=sessions,
        steps_per_session=steps_per_session,
        missing_counts=missing_counts,
        spikes_fired=fired_total,
    )


def trace_jsonl(rows: list[dict]) -> bytes:
    """Canonical trace bytes: one compact JSON object per step, LF endings."""
    lines = [json.dumps(row, sort_keys=True, ensure_ascii=True) for row in rows]
    return ("\n".join(lines) + "\n").encode("utf-8")


def _git_dir(root: Path) -> Path | None:
    """The real git directory, following a worktree ``.git`` file."""
    gitdir = root / ".git"
    try:
        if gitdir.is_file():
            text = gitdir.read_text(encoding="utf-8").strip()
            if not text.startswith("gitdir:"):
                return None
            gitdir = (root / text.split(":", 1)[1].strip()).resolve()
    except OSError:
        return None
    return gitdir if gitdir.is_dir() else None


def _common_dir(gitdir: Path) -> Path:
    """The shared git dir: in a linked worktree the ``commondir`` file
    names it; loose branch refs and ``packed-refs`` live there, not in
    the per-worktree dir that HEAD sits in."""
    try:
        rel = (gitdir / "commondir").read_text(encoding="utf-8").strip()
    except OSError:
        return gitdir
    common = (gitdir / rel).resolve()
    return common if common.is_dir() else gitdir


def _ref_target(gitdir: Path, ref: str) -> str | None:
    try:
        return (gitdir / ref).read_text(encoding="utf-8").strip()
    except OSError:
        pass
    try:
        packed = (gitdir / "packed-refs").read_text(encoding="utf-8")
    except OSError:
        return None
    for line in packed.splitlines():
        line = line.strip()
        if line and not line.startswith(("#", "^")) and line.endswith(" " + ref):
            return line.split(" ", 1)[0]
    return None


def git_revision(root: Path) -> dict[str, Any]:
    """HEAD commit + branch read from ``.git`` files; nulls where unavailable.

    No subprocess: replay must stay hermetic, and spawning ``git`` is a
    security-scanner hotspot. Loose refs are read directly; packed refs
    fall back to ``.git/packed-refs``. A worktree ``.git`` file is
    followed. The working-tree dirty flag is not derivable without git,
    so it is reported as ``None`` (unavailable), never guessed.
    """
    info: dict[str, Any] = {"commit": None, "branch": None, "dirty": None}
    gitdir = _git_dir(root)
    if gitdir is None:
        return info
    try:
        head = (gitdir / "HEAD").read_text(encoding="utf-8").strip()
    except OSError:
        return info
    commit = head
    if head.startswith("ref:"):
        ref = head.split(":", 1)[1].strip()
        if ref.startswith("refs/heads/"):
            info["branch"] = ref[len("refs/heads/"):]
        commit = _ref_target(_common_dir(gitdir), ref)
    if commit is not None and len(commit) == 40:
        info["commit"] = commit
    return info


def _display_path(path: Path) -> str:
    """Repo-relative when under the checkout, absolute otherwise."""
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def build_manifest(
    *,
    entry: AttestedEntry,
    bank: ModelBank,
    jsonl: Path,
    split_manifest_path: Path | None,
    input_sha256: dict[str, str | None],
    config: ReplayConfig,
    result: ReplayResult,
    trace_bytes: bytes,
) -> dict:
    """The reproducibility manifest. No wall-clock fields: this document is
    byte-identical between runs on the same inputs, which is the property
    the determinism criterion checks."""
    return {
        "schema_version": MANIFEST_SCHEMA,
        "trace_schema": TRACE_SCHEMA,
        "trace_sha256": sha256_bytes(trace_bytes),
        "model": _model_section(entry, bank),
        "input": _input_section(
            jsonl, split_manifest_path, config, result, input_sha256
        ),
        "encoder": {
            "kind": "analog-current, frozen minmax (not spikes)",
            "columns": list(LIVE_COLUMNS),
            "frozen_minmax_lineage": FROZEN_LINEAGE,
            "frozen_minmax": {k: list(v) for k, v in FROZEN_MINMAX.items()},
            "missing_policy": config.missing_policy,
            "missing_counts": result.missing_counts,
            "staleness": "undetectable on v3 state_telemetry (ts_utc is "
            "null); rows replay in file order",
        },
        "stepper": {
            "name": "tools/hamming_lif.py keep_lif_step "
            "(v = decay*v + W@stim; decay is KEEP)",
            "k": config.k,
            "i_drive": config.i_drive,
            "n_neurons": N_NEURONS,
            "session_reset": "LifBank.reset() at each episode_id boundary",
        },
        "output": {
            "vocabulary": list(SHIPPED_VOCABULARY),
            "contract": "tools/decision_core.py replay_output_row",
            "diagnostic_only": True,
            "note": "argmax over comfort/temp/power is a diagnostic, not a "
            "validated ALLOW/WARN/THROTTLE/PAUSE/YIELD_GPU policy",
        },
        "results": {
            "spikes_fired": result.spikes_fired,
            "sessions": len(result.sessions),
        },
        "source": git_revision(REPO_ROOT),
    }


def _model_section(entry: AttestedEntry, bank: ModelBank) -> dict[str, Any]:
    return {
        "id": entry.id,
        "checkpoint": entry.checkpoint_relative,
        "checkpoint_digest": entry.checkpoint_digest,
        "feature_map_id": entry.feature_map_id,
        "output_contract_id": entry.output_contract_id,
        "numeric_format": entry.numeric_format,
        "training_dataset_digest": entry.training_dataset_digest,
        "bank_manifest": (
            _display_path(bank.manifest_path)
            if bank.manifest_path is not None
            else None
        ),
    }


def _input_section(
    jsonl: Path,
    split_manifest_path: Path | None,
    config: ReplayConfig,
    result: ReplayResult,
    input_sha256: dict[str, str | None],
) -> dict[str, Any]:
    doc: dict[str, Any] = {
        "path": _display_path(jsonl),
        "sha256": input_sha256["jsonl"],
        "split": config.split,
        "sessions": result.sessions,
        "steps_per_session": result.steps_per_session,
        "n_steps": len(result.trace_rows),
    }
    if split_manifest_path is not None:
        doc["split_manifest"] = {
            "path": _display_path(split_manifest_path),
            "sha256": input_sha256["split_manifest"],
        }
    return doc


def manifest_json(manifest: dict) -> bytes:
    """Canonical manifest bytes (same rules as ``model_bank.dumps_manifest``)."""
    return (
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=True)
        + "\n"
    ).encode("utf-8")


def default_k(model: dict) -> int | None:
    """K-WTA from the checkpoint metadata; ``None`` when unrecorded."""
    raw = model.get("k_wta")
    if raw is None:
        return None
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise ParseError(f"checkpoint 'k_wta' is not an integer: {raw!r}")
    return raw


def default_i_drive(model: dict) -> float:
    """Dale I bias: the checkpoint's recorded ``exp023_knobs.I_DRIVE``,
    else the shipped protocol constant ``I_DRIVE_EXP024`` (0.05)."""
    knobs = model.get("exp023_knobs")
    if not isinstance(knobs, dict):
        return I_DRIVE_EXP024
    raw = knobs.get("I_DRIVE")
    if raw is None:
        return I_DRIVE_EXP024
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise ParseError(f"checkpoint 'exp023_knobs.I_DRIVE': {raw!r}")
    if not math.isfinite(raw):
        raise ParseError(f"checkpoint 'exp023_knobs.I_DRIVE': {raw!r}")
    return float(raw)


def select_entry(bank: ModelBank, model_id: str | None) -> AttestedEntry:
    """Pick the replayed entry. A multi-entry bank requires an explicit id.

    Attestation verifies digest and shape, not semantics: this replay
    implements exactly the exp-025 feature map and the supervisor-v3
    (comfort/temp/power) output contract, so an attested entry declaring
    any other contract is refused -- a digest-valid foreign-contract
    checkpoint would otherwise emit a semantically invalid trace.
    """
    entry = bank.select(model_id) if model_id is not None else _sole(bank)
    if entry.feature_map_id != FEATURE_MAP_LIVE_EXP_025:
        raise ParseError(
            f"{entry.id}: feature_map_id {entry.feature_map_id!r} is not "
            f"the replayed contract {FEATURE_MAP_LIVE_EXP_025!r}"
        )
    if entry.output_contract_id != OUTPUT_CONTRACT_SUPERVISOR_V3_RM1150:
        raise ParseError(
            f"{entry.id}: output_contract_id {entry.output_contract_id!r} "
            f"is not the replayed contract "
            f"{OUTPUT_CONTRACT_SUPERVISOR_V3_RM1150!r}"
        )
    return entry


def _sole(bank: ModelBank) -> AttestedEntry:
    if len(bank.entries) != 1:
        raise ParseError(
            "bank holds "
            f"{len(bank.entries)} attested entries ({', '.join(bank.ids())}); "
            "pass --model-id"
        )
    return bank.entries[0]


def _read_input_bytes(path: Path, kind: str) -> bytes:
    """Read an input once; the returned bytes are parsed *and* hashed, so
    the manifest can only describe the payload that was replayed."""
    if not path.is_file():
        raise ParseError(f"missing {kind}: {path}")
    try:
        return path.read_bytes()
    except OSError as exc:
        raise ParseError(f"unreadable {kind} {path}: {exc}") from exc


def _utf8_or_refuse(payload: bytes, path: Path, kind: str) -> str:
    try:
        return payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ParseError(
            f"{path.name}: {kind} is not UTF-8 ({exc})"
        ) from exc


def load_replay_inputs(
    manifest_path: Path,
    model_id: str | None,
    jsonl: Path,
    split: str,
    split_manifest_path: Path | None,
) -> tuple[ModelBank, AttestedEntry, list[Sample], dict[str, str | None]]:
    """Attest the bank, then select rows. The checkpoint bytes consumed
    downstream are the attested ones -- there is no unverified reload.
    Each input file is read once; its digest (returned in the fourth
    element) covers the same bytes that were parsed."""
    bank = load_model_bank(manifest_path)
    entry = select_entry(bank, model_id)
    split_doc = None
    split_sha256 = None
    if split_manifest_path is not None:
        payload = _read_input_bytes(split_manifest_path, "split manifest")
        split_sha256 = sha256_bytes(payload)
        split_doc = _split_manifest_entries(
            split_manifest_path,
            _utf8_or_refuse(payload, split_manifest_path, "split manifest"),
        )
    jsonl_bytes = _read_input_bytes(jsonl, "file")
    records = parse_jsonl_text(
        _utf8_or_refuse(jsonl_bytes, jsonl, "file"), jsonl.name
    )
    samples = select_replay_samples(records, split, split_doc)
    return bank, entry, samples, {
        "jsonl": sha256_bytes(jsonl_bytes),
        "split_manifest": split_sha256,
    }


def shipped_manifest_path() -> Path:
    return SHIPPED_DIR / "model_bank.json"


def fixture_jsonl() -> Path:
    return REPLAY_FIXTURE_DIR / "telemetry.jsonl"
