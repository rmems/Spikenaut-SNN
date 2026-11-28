"""Verify the promoted exp-025 artifact bytes against their provenance record."""

import hashlib
import json
from pathlib import Path
import sys


BANK = Path(__file__).resolve().parents[1] / "dataset" / "merged_v2"
ARTIFACTS = (
    "snn_model.json",
    "parameters.mem",
    "parameters_weights.mem",
    "parameters_decay.mem",
    "parameters_output_weights.mem",
)


def verify(bank: Path) -> None:
    record = json.loads((bank / "provenance.json").read_text(encoding="utf-8"))
    artifacts = record["artifacts"]
    if set(artifacts) != set(ARTIFACTS):
        raise ValueError("expected the five promoted exp-025 artifacts")
    for name in ARTIFACTS:
        data = (bank / name).read_bytes()
        pin = artifacts[name]
        if len(data) != pin["bytes"]:
            raise ValueError(f"{name}: byte count differs from provenance record")
        if hashlib.sha256(data).hexdigest() != pin["sha256"]:
            raise ValueError(f"{name}: SHA-256 differs from provenance record")


def main() -> int:
    try:
        verify(BANK)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"exp-025 provenance verification failed: {exc}", file=sys.stderr)
        return 1
    print("Verified the five promoted exp-025 artifacts.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
