"""Single dual-import site for the Hamming CLI and self-test.

``measure_hamming`` and ``hamming_selftest`` used to carry identical
try/except import blocks. Both now load symbols from here.
"""

from __future__ import annotations

try:
    from .hamming_core import (
        CONDITION_EXP024,
        CONDITION_METHOD_FIXTURE,
        CONDITION_SHIPPED,
        FIXTURE_DIR,
        I_DRIVE_EXP024,
        KResult,
        LifBank,
        Measurement,
        ParseError,
        Protocol,
        SHIPPED_DIR,
        apply_kwta,
        encode_q88_hex,
        encode_record,
        keep_lif_step,
        load_expected,
        measure,
        measure_method_fixture,
        method_fixture_paths,
        pin_matches,
        select_samples,
        under_dir,
    )
    from .hamming_report import report
    from .q88_core import Q88RangeError, SelfTestFailure
except ImportError:
    from hamming_core import (
        CONDITION_EXP024,
        CONDITION_METHOD_FIXTURE,
        CONDITION_SHIPPED,
        FIXTURE_DIR,
        I_DRIVE_EXP024,
        KResult,
        LifBank,
        Measurement,
        ParseError,
        Protocol,
        SHIPPED_DIR,
        apply_kwta,
        encode_q88_hex,
        encode_record,
        keep_lif_step,
        load_expected,
        measure,
        measure_method_fixture,
        method_fixture_paths,
        pin_matches,
        select_samples,
        under_dir,
    )
    from hamming_report import report
    from q88_core import Q88RangeError, SelfTestFailure

__all__ = (
    "CONDITION_EXP024",
    "CONDITION_METHOD_FIXTURE",
    "CONDITION_SHIPPED",
    "FIXTURE_DIR",
    "I_DRIVE_EXP024",
    "KResult",
    "LifBank",
    "Measurement",
    "ParseError",
    "Protocol",
    "Q88RangeError",
    "SHIPPED_DIR",
    "SelfTestFailure",
    "apply_kwta",
    "encode_q88_hex",
    "encode_record",
    "keep_lif_step",
    "load_expected",
    "measure",
    "measure_method_fixture",
    "method_fixture_paths",
    "pin_matches",
    "report",
    "select_samples",
    "under_dir",
)
