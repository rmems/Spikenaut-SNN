# Float-vs-Q8.8 Hamming protocol (issue #39)

This is a **measurement with its protocol**, not a pass/fail gate. How much
float-to-Q8.8 spike disagreement is acceptable is blocked on the
output/decision contract ([#20](https://github.com/rmems/Spikenaut-SNN/issues/20)).
This document must not grow a Hamming threshold.

The harness is `tools/measure_hamming.py`. It is standard-library Python.
The keep-LIF it steps is **harness-only** — `src/` still does not advance
membranes, and nothing here claims the Rust crate runs spikes.

## One-command runs

```text
python3 tools/measure_hamming.py
python3 tools/measure_hamming.py --self-test
```

Exit codes match `verify_q88.py`: `0` published / self-test passed, `1` a
method pin or self-test assertion failed, `2` an artifact could not be
parsed. An empty holdout is a hard failure, never `0.0%`.

The default command scores the in-repo **method fixture**
(`tools/fixtures/hamming_method/`). That proves the method and pins a
known Hamming so CI can notice a broken stepper. It is **not** exp-024.

## exp-024 reference (the published number)

| Field | Value |
|---|---|
| Weights | exp-023 PASS Distill knobs scratch (seed 123 / 5 ep) |
| Not | shipped `dataset/merged_v2` ramp — that is a different condition and must be labeled `shipped-merged-v2` |
| Encoder | legal 5-ch train-scaled: `mem_util_pct`, `power_w`, `gpu_temp_c`, `sm_clock_mhz`, `mem_clock_mhz` |
| Minmax | frozen v3 train split, sha lineage `74acdd0f` (do not refit on val/test) |
| Unused axons | 5–15 held at 0 (unused width, not fake channels) |
| Split | v3 `state_telemetry` test episodes `gpu-000170..198` |
| Ticks | n = 117653 |
| Seed | 123 (Distill training seed; analog current, Poisson unused when `learn=false`) |
| Stepper | keep-LIF: `v = decay * v + W @ stim`; decay is a **keep** factor |
| I-drive | `0.05` added to neurons 12–15 (Dale I bias; train and eval) |
| K-WTA | `k=none`: every suprathreshold spike kept. `k=4`: Distill mixed E/I quota among firers (`I_WTA_MAX=2`, `E_WTA_MIN=2`, 12 E + 4 I) |
| Compared | float bank (`snn_model.json` as written, binary32) vs Q8.8-decoded `.mem` |
| Claimed | `k=none` 13.187% / 0.1608 bits; `k=4` 56.188% / 1.697 bits; hidden json↔mem 0/256 |

Hamming **percent** is the fraction of ticks whose 16-bit spike vectors
differ. Hamming **bits** is the mean popcount of the per-tick XOR.

Those scratch weights and the 117653-tick JSONL are **not in this
repository**. This tool will not invent a Hugging Face download and must
not overwrite `dataset/merged_v2`.

```text
python3 tools/measure_hamming.py --condition exp-024 \
    --jsonl PATH/state_telemetry.jsonl \
    --float-json PATH/snn_model.json \
    --mem-dir PATH/
```

`--condition exp-024` refuses `dataset/merged_v2`. To score the shipped
ramp on an external JSONL, say so:

```text
python3 tools/measure_hamming.py --condition shipped-merged-v2 \
    --jsonl PATH/state_telemetry.jsonl
```

## Frozen minmax (lineage 74acdd0f)

Copied from SynapticDistill `scripts/spikenaut_train.jl`. Missing / null
encodes as 0. `gpu_temp_c == 0` **stays 0** on this protocol (Distill:
"T=0 stays 0"); it is not dropped as a dropout tick.

| Column | min | max |
|---|---:|---:|
| `mem_util_pct` | 0.0 | 75.0 |
| `power_w` | 8.527000427246094 | 302.8450012207031 |
| `gpu_temp_c` | 0.0 | 69.0 |
| `sm_clock_mhz` | 180.0 | 2910.0 |
| `mem_clock_mhz` | 405.0 | 14801.0 |

## Why exp-024 is not reproduced by `python3 tools/measure_hamming.py`

The default run has no path to the exp-023 scratch bank or the v3 test
JSONL. The method fixture is a 4-tick synthetic holdout with a
deliberate 1-neuron shift between the JSON float bank and the `.mem`
bank, so Hamming is large and pinned. That is a harness regression pin,
not a claim about FPGA parity and not a substitute for the 117653-tick
figure.

When the scratch paths are supplied, the printed report compares the
fresh numbers to the claimed exp-024 figures and reports the delta.
Until then the honest statement is: **protocol published; figures
recorded; in-repo reproduction blocked on artifacts this vault does not
ship.**

## Out of scope

- A Hamming pass/fail threshold (blocked on #20)
- Closing #2 or #13
- Retrain / Dale invent / HF publish / silicon weight-load
- Overwriting `dataset/merged_v2`
