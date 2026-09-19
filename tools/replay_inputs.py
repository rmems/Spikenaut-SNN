"""Input loading and selection for the frozen replay (RM-1692 / GH #60).

Split out of ``replay_core`` to keep each module under the repo's
file-size gate. Everything here runs *before* stepping: bank
attestation and contract validation, split-manifest parsing, sample
selection, and the single-read input bytes whose digests the manifest
records.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

try:  # package import: `python3 -m tools.replay_frozen`
    from .hamming_encode import (
        Sample,
        episode_index,
        parse_jsonl_text,
        select_samples,
    )
    from .model_bank import (
        FEATURE_MAP_LIVE_EXP_025,
        OUTPUT_CONTRACT_SUPERVISOR_V3_RM1150,
        AttestedEntry,
        ModelBank,
        load_model_bank,
    )
    from .model_bank_json import _parse_json, _read_utf8
    from .q88_core import ParseError
except ImportError:  # direct script: `python3 tools/replay_frozen.py`
    from hamming_encode import (
        Sample,
        episode_index,
        parse_jsonl_text,
        select_samples,
    )
    from model_bank import (
        FEATURE_MAP_LIVE_EXP_025,
        OUTPUT_CONTRACT_SUPERVISOR_V3_RM1150,
        AttestedEntry,
        ModelBank,
        load_model_bank,
    )
    from model_bank_json import _parse_json, _read_utf8
    from q88_core import ParseError

SPLIT_MANIFEST_SCHEMA = "spikenaut.split-manifest.v1"
SPLIT_NAMES = ("train", "val", "test")
_SPLIT_MANIFEST_KIND = "split manifest"


def sha256_bytes(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


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
    return _split_manifest_entries(
        path, _read_utf8(path, kind=_SPLIT_MANIFEST_KIND)
    )


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
        payload = _read_input_bytes(split_manifest_path, _SPLIT_MANIFEST_KIND)
        split_sha256 = sha256_bytes(payload)
        split_doc = _split_manifest_entries(
            split_manifest_path,
            _utf8_or_refuse(payload, split_manifest_path, _SPLIT_MANIFEST_KIND),
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
