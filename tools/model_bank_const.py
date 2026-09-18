# SPDX-License-Identifier: MIT OR Apache-2.0
"""Shared model-bank constants (script vs ``-m``).

Identifiers, schema keys, and the shipped-bundle paths live here so the
JSON, I/O, and field-check modules do not each re-declare the contract.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_FILENAME = "model_bank.json"
SHIPPED_MANIFEST = REPO_ROOT / "dataset" / "merged_v2" / MANIFEST_FILENAME
SHIPPED_CHECKPOINT = REPO_ROOT / "dataset" / "merged_v2" / "snn_model.json"

SCHEMA_VERSION = 1
SUPPORTED_SCHEMA_VERSIONS = frozenset({SCHEMA_VERSION})

DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
MSG_MISSING_FIELD = "missing required field"

# Canonical identifiers this repository ships. Unknown-but-well-formed IDs
# still attest -- the bank is not a registry -- but missing/empty values do not.
FEATURE_MAP_LIVE_EXP_025 = "spikenaut.feature-map.live-exp-025.v1"
OUTPUT_CONTRACT_SUPERVISOR_V3_RM1150 = (
    "spikenaut.output-contract.supervisor-v3.rm-1150"
)
NUMERIC_FORMAT_Q88 = "q8.8-fixed-point"
HUB_V3_JSONL_DIGEST = (
    "sha256:26d7d7442605a32b11330f53f55e621750f31e775465209991f73f94a6c72c09"
)

MANIFEST_KEYS = frozenset({"schema_version", "models"})
ENTRY_REQUIRED = (
    "id",
    "checkpoint",
    "checkpoint_digest",
    "feature_map_id",
    "output_contract_id",
    "numeric_format",
)
ENTRY_OPTIONAL = ("training_dataset_digest",)
ENTRY_KEYS = frozenset(ENTRY_REQUIRED + ENTRY_OPTIONAL)
