"""Shared Hamming constants (script vs ``-m``).

Kept small and import-cycle-free so encode / LIF / measure can all
depend on it. ``SelfTestFailure`` is not imported here -- only the CLI
and self-test raise or catch it.
"""

from __future__ import annotations

import struct
from pathlib import Path

try:  # package import: `python3 -m tools.measure_hamming`
    from .q88_core import N_INPUTS, N_NEURONS
except ImportError:  # direct script: `python3 tools/measure_hamming.py`
    from q88_core import N_INPUTS, N_NEURONS

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURE_DIR = REPO_ROOT / "tools" / "fixtures" / "hamming_method"
SHIPPED_DIR = REPO_ROOT / "dataset" / "merged_v2"

# Legal 5-ch train-scaled encoder (exp-008 / exp-024). Order is the contract.
LIVE_COLUMNS: tuple[str, ...] = (
    "mem_util_pct",
    "power_w",
    "gpu_temp_c",
    "sm_clock_mhz",
    "mem_clock_mhz",
)
N_LIVE_AXONS = len(LIVE_COLUMNS)
# Unused encoder width: axons 5-15 stay 0. Ties the live-column count
# to the 16-wide bank so N_INPUTS is not a re-export-only import.
UNUSED_AXONS = tuple(range(N_LIVE_AXONS, N_INPUTS))

# Frozen minmax from v3 state_telemetry train, sha lineage 74acdd0f.
# Do not refit on val/test. Copied from SynapticDistill.jl FROZEN_MINMAX.
FROZEN_MINMAX: dict[str, tuple[float, float]] = {
    "mem_util_pct": (0.0, 75.0),
    "power_w": (8.527000427246094, 302.8450012207031),
    "gpu_temp_c": (0.0, 69.0),
    "sm_clock_mhz": (180.0, 2910.0),
    "mem_clock_mhz": (405.0, 14801.0),
}
FROZEN_LINEAGE = "74acdd0f"

# Episode holdout. Session key is episode_id (ts_utc is 100% null on v3).
TRAIN_EP_LO, TRAIN_EP_HI = 0, 138
VAL_EP_LO, VAL_EP_HI = 140, 168
TEST_EP_LO, TEST_EP_HI = 170, 198
EMBARGO_EPS = frozenset((139, 169))
EXP024_TEST_N_TICKS = 117653

# Distill Dale / K-WTA / I-drive. 0-based: excitatory 0..11, inhibitory 12..15.
N_EXC = 12
INHIB_ROWS = tuple(range(N_EXC, N_NEURONS))
I_DRIVE_EXP024 = 0.05
I_WTA_MAX = 2
E_WTA_MIN = 2

CONDITION_METHOD_FIXTURE = "method-fixture"
CONDITION_EXP024 = "exp-024"
CONDITION_SHIPPED = "shipped-merged-v2"

FORBIDDEN_SENSORS = (
    "hashrate_mh_derived",
    "power_w_derived",
    "gpu_temp_c_derived",
    "reward_hint_derived",
    "tick_rate",
    "fan_speed_pct",
    "vddcr_gfx_v",
    "vram_temp_c",
    "step_idx",
)


def f32(value: float) -> float:
    """Snap ``value`` onto IEEE-754 binary32, matching Julia ``Float32``.

    The Distill sidecar does every LIF update in Float32. Python's default
    float is binary64; leaving the extra bits in would invent a third
    arithmetic that neither bank used.
    """
    return struct.unpack("=f", struct.pack("=f", float(value)))[0]
