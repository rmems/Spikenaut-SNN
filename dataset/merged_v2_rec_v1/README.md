# `merged_v2_rec_v1` — exp-026 recurrent research variant (GH #3)

**Not the shipped bank.** `dataset/merged_v2/` is still the live exp-025 artifact, the Rust
crate still loads only that, and nothing here is promoted, qualified, or board-measured. This
bundle exists so the recurrent variant evaluated in
[`docs/recurrence/`](../../docs/recurrence/README.md) is a separately named, attestable,
replayable checkpoint rather than a number in a table.

## What it is

The shipped `merged_v2` checkpoint (digest in `snn_model.json` → `variant.derived_from`)
with two changes and nothing else:

1. a `recurrence` block, schema `spikenaut.recurrence.v1` (`tools/recurrent_lif.py`): I→E
   lateral inhibition at gain **1/8**, one-tick delay, per-session state — `R[e][i] = -0.125`
   for every excitatory `e` (0–11) and inhibitory `i` (12–15), all other entries 0;
2. `output_weights` refit on the v3 train split (`gpu-000000..138`, 569,344 ticks) by the
   Dale-constrained ridge in `tools/recurrence_core.py`, `lambda_rel = 1e-4`, `k = 4`,
   I-drive 0.05, selected on the validation split. The fitted head keeps outgoing Dale with
   zero sign violations.

Every neuron's `threshold`, `decay_rate`, `weights`, `inhibitory`, `membrane_potential` and
`last_spike` are copied verbatim from `merged_v2`, so `parameters.mem`,
`parameters_decay.mem` and `parameters_weights.mem` are **byte-identical** to the shipped
images (`tests/test_recurrence.py` asserts it). `parameters_output_weights.mem` holds the
refit head and `parameters_recurrent_weights.mem` holds `R` as 256 signed Q8.8 words,
post-major (`post * 16 + pre`).

## Files

| File | Role |
|---|---|
| `snn_model.json` | the checkpoint; `recurrence` block plus `variant` provenance |
| `model_bank.json` | one attested entry, id `merged_v2_rec_v1`; `training_dataset_digest` is the sha256 of the projected v3 JSONL the head was fit on (`docs/recurrence/README.md`) |
| `parameters.mem`, `parameters_decay.mem`, `parameters_weights.mem` | identical to `dataset/merged_v2` |
| `parameters_output_weights.mem` | the refit 16×3 head, neuron-major, 48 words |
| `parameters_recurrent_weights.mem` | the recurrence matrix, 256 words |
| `bundle_digests.json` | sha256 of each file as written |

## How it was made and how to regenerate it

Written by `tools/recurrence_eval.py --emit-checkpoint` at the end of the recorded `k=4`
run; regenerate only from the same input (the Hub v3 `state_telemetry` shards projected as
`docs/recurrence/README.md` describes), which reproduces these bytes exactly. Do not edit
the files by hand and do not regenerate from a different input without changing the id.

```bash
python3 tools/verify_model_bank.py dataset/merged_v2_rec_v1/model_bank.json --select merged_v2_rec_v1
python3 tools/replay_frozen.py --bank-manifest dataset/merged_v2_rec_v1/model_bank.json
```

The replay applies the recurrence and records it under `stepper.recurrence` in its manifest.

## Status

Measured on the recorded held-out test split: 9.5% lower session-weighted readout error than
the feed-forward bank with an identically refit head, no head more than 0.3% worse, 418
distinct spike patterns against 268 — and still worse than predicting the train-mean target.
Hardware: none. The FPGA SoC has no neuron-to-neuron feedback path and the parity harness
([silicon-hdl#76](https://github.com/rmems/silicon-hdl/issues/76)) has not run, so
`parameters_recurrent_weights.mem` currently has no consumer outside `tools/replay_core.py`.
