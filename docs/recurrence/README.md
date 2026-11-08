# exp-026: recurrence against the frozen exp-025 baseline

Issue [#3](https://github.com/rmems/Spikenaut-SNN/issues/3) asks whether recurrence buys
anything beyond the signed, feed-forward exp-025 bank. This directory holds the answer as
measured on the recorded held-out sessions, the tool that produced it, and the separately
named checkpoint it evaluated.

**Result in one paragraph.** With the checkpoint's own K-WTA (`k=4`), I→E lateral
inhibition at gain 1/8 lowers the held-out readout error by 9.5% against the feed-forward
bank with an identically refit head and by 8.4% against the shipped head, with no head
more than 0.3% worse, so it meets the criterion predeclared in `tools/recurrence_core.py`.
It does so by breaking the lockstep of the excitatory population (268 → 418 distinct spike
patterns at the same spike count), not by adding memory. The same comparison also shows
that **no network arm, recurrent or not, beats a constant predictor** of the train-mean
target on the test split (0.0816 versus 0.0967 for the best arm), and that a memoryless
linear read of the five inputs is ten times better than any of them (0.0090). With K-WTA
off (the health protocol) inhibition only hurts, monotonically with gain, and gains of 3/8
and above silence the excitatory population. Recurrence is therefore a small, real,
relative improvement on a readout contract that the architecture does not solve; it is not
a promotion candidate and nothing here changes `dataset/merged_v2`.

## The variant

`tools/recurrent_lif.py` defines the one recurrent extension evaluated here and the
checkpoint block that declares it, `spikenaut.recurrence.v1`:

| Aspect | Definition |
|---|---|
| Topology | Dense `16 × 16` matrix `R[post][pre]` on the Q8.8 grid, post-major like `weights`. Dale on the presynaptic side: inhibitory columns ≤ 0, excitatory columns ≥ 0, zero diagonal. The exp-026 family is I→E lateral inhibition only: `R[e][i] = -gain` for every excitatory `e` (0–11) and inhibitory `i` (12–15); E→E, E→I, I→I and self connections are 0 (48 nonzero entries). |
| Delay | Exactly one tick. The feedback current at tick `t` is `R @ s(t-1)`, where `s(t-1)` is the spike vector the bank *emitted* at `t-1`, after K-WTA. There is no same-tick recurrence. |
| State | The membrane `v` plus `s(t-1)`. Both reset at every `episode_id` boundary; the first tick of a session has zero feedback. |
| Order inside a tick | `input = W @ stim`, `+ I_DRIVE` on neurons 12–15, `+ R @ s(t-1)`; `v = keep * v + input`; fire at `v >= threshold`; optional K-WTA; reset winners. Every term is accumulated in binary32 by the same `tools/hamming_lif.py` stepper the frozen replay uses. |
| Export | `parameters_recurrent_weights.mem`: 256 signed Q8.8 words, post-major. The four feed-forward images keep their layout. |
| Identity | With `R = 0` the step is bit-identical to the feed-forward reference (`tests/test_recurrence.py`). |

A checkpoint that carries the block is replayed recurrently by `tools/replay_frozen.py`,
whose manifest records the block under `stepper.recurrence`; a feed-forward checkpoint
records `null` there. A block that violates the contract (shape, schema, delay, Dale, grid)
makes the checkpoint unconsumable (exit 2). Nothing may apply recurrent weights a checkpoint
did not spell out.

The checkpoint for the validation-selected gain is [`dataset/merged_v2_rec_v1/`](../../dataset/merged_v2_rec_v1/README.md):
the shipped hidden parameters, byte-identical `.mem` images for them, the recurrence block,
and a readout refit on the train split.

## Held-out data

The recorded exp-025 holdout is the v3 `state_telemetry` **test** split, episodes
`gpu-000170..198`, 117,653 ticks. This repository does not ship it; the Hub does, as Parquet:

| Shard | Rows | Episodes | sha256 (Hub LFS oid) |
|---|---:|---|---|
| `v3/state_telemetry/train-00000.parquet` | 569,344 | `gpu-000000..138` (139) | `27f15ef1e9b11ac163300d4581d357b05b812a045bd6789e8fb3a657aa014b40` |
| `v3/state_telemetry/validation-00000.parquet` | 118,784 | `gpu-000140..168` (29) | `f842013642780cba49dc9d8f0247c1658d417230c75a1a9f5a574e3837241eeb` |
| `v3/state_telemetry/test-00000.parquet` | 117,653 | `gpu-000170..198` (29) | `8b186b4f4d3090dc6fb05bed3da2aeb23f04ec507dd3ce96e92c66e36c9ce6a6` |

from `https://huggingface.co/datasets/rmems/Spikenaut-SNN-Telemetry/resolve/main/<shard>`.
The tools here are standard-library and refuse Parquet, so the shards were projected to one
JSONL in file order (train, validation, test), keeping `episode_id` and the five live columns
`mem_util_pct`, `power_w`, `gpu_temp_c`, `sm_clock_mhz`, `mem_clock_mhz` and dropping every
other column (the tools refuse rows that carry `step_idx`, `vram_temp_c`, `fan_speed_pct` or
`vddcr_gfx_v`). Nulls stay `null`; nothing is imputed and no timestamp is invented. The
projection is reproducible with `pyarrow` outside this repository:

```python
import json, pyarrow.parquet as pq
LIVE = ["mem_util_pct", "power_w", "gpu_temp_c", "sm_clock_mhz", "mem_clock_mhz"]
with open("state_telemetry.jsonl", "w", encoding="utf-8", newline="\n") as out:
    for shard in ("train", "validation", "test"):
        d = pq.read_table(f"{shard}-00000.parquet").to_pydict()
        for i in range(len(d["episode_id"])):
            rec = {"episode_id": d["episode_id"][i]}
            rec.update((c, None if d[c][i] is None else float(d[c][i])) for c in LIVE)
            out.write(json.dumps(rec, ensure_ascii=True) + "\n")
```

The result is 805,781 rows, 119,734,661 bytes, sha256
`b1e601077100aabcc24021bcf86b8eeabbd9f5c1140623cdbc4ea08d946372da`; every episode is
contiguous, `step_idx` is monotonic within each, no live column is null, and 11,997 test
rows carry `gpu_temp_c == 0` (13 in train, 0 in validation), the dropout sentinel the state
contract in #20 says must be masked and which the exp-025 protocol encodes as 0.

**This is the recorded holdout, not a lookalike.** Stepping the float `snn_model.json` bank
through `tools/hamming_lif.py` with `k=none` over these test rows reproduces the README health
figures exactly: cofire 0.863, all-16 0.000, 223,511 inhibitory spikes, 206 distinct
patterns. (The comparison itself runs on the Q8.8 grid the replay contract uses, where the
same pass gives 0.852 / 0.000 / 223,637 / 165.) The bank's recorded
`training_dataset_digest` names the training-time JSONL, a different file that is not on the
Hub; the variant bundle records the projection's digest instead.

## Protocol

`tools/recurrence_eval.py` ran once per K-WTA setting on that JSONL (the two labelled runs
are what is committed under `exp-026/`):

- **Arms.** `ff` is the shipped bank on the Q8.8 grid. `rec-g<gain>` adds lateral inhibition
  at gain 1/16, 1/8, 1/4, 3/8, 1/2 (the design pass showed 1/2 silences every excitatory
  neuron; the grid brackets that collapse on purpose). All arms step the same encoded rows
  with the checkpoint's I-drive 0.05, resetting at every `episode_id`.
- **Target.** Distill's `sample_readout_target`: `(comfort, temp, power)` with `temp` and
  `power` the frozen-minmax axons and `comfort = clamp(1 - 0.5·temp - 0.5·power, 0, 1)`,
  binary32. It is a function of the current input, which is why the linear baseline is a
  ceiling.
- **Readouts.** `ff-shipped` scores the shipped head as-is. Every arm, `ff` included, also
  gets `ff-refit`-style head: ridge on the emitted spike vectors, no bias (the shipped head
  has none), the exp-025 outgoing-Dale sign box (excitatory columns ≥ 0, inhibitory ≤ 0)
  solved exactly by Lawson–Hanson NNLS on the normal equations, weights snapped to the Q8.8
  grid, `lambda = lambda_rel · n_train` with `lambda_rel ∈ {1e-4, 1e-3, 1e-2, 1e-1, 1}` chosen
  on validation only. Refitting `ff` too is what separates the recurrence effect from the
  effect of a fresh fit.
- **Metrics.** Per head, MAE per session averaged with equal session weight; the primary is
  the mean of the three. Pooled MAE/RMSE are kept alongside. Activity follows Distill
  `health_eval` (spikes/tick, all-16, mean pairwise cofire, patterns, inhibitory spikes,
  silent neurons). Decision agreement uses `tools/decision_core.py`, the replay's own argmax.
- **Baselines.** Train-mean constant; persistence (previous tick's target within the session);
  memoryless ridge on the five live axons plus bias (float, selected on validation).
- **Verdict (predeclared in code before the first real run).** The validation-selected
  recurrent arm is *promising* only if its test primary is at least 5% below `ff-refit`'s
  and no head is more than 5% worse. Selection never sees the test split.

## Results on the test split (117,653 ticks, 29 sessions)

### `k = 4` (the checkpoint's K-WTA; the replay default) — `exp-026/comparison.json`

| arm | λ_rel | val primary | **test primary** | comfort | temp | power | spk/tick | patterns | cofire |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `ff-shipped` | — | 0.0226 | 0.1055 | 0.1741 | 0.0693 | 0.0731 | 3.591 | 268 | 0.264 |
| `ff-refit` | 1e-4 | 0.0369 | 0.1068 | 0.1984 | 0.0556 | 0.0664 | 3.591 | 268 | 0.264 |
| `rec-g0.0625` | 1e-4 | 0.0359 | 0.1061 | 0.1971 | 0.0541 | 0.0672 | 3.591 | 273 | 0.264 |
| **`rec-g0.125`** (val-selected) | 1e-4 | **0.0191** | **0.0967** | 0.1724 | 0.0511 | 0.0665 | 3.568 | 418 | 0.262 |
| `rec-g0.25` | 1e-4 | 0.0388 | 0.1487 | 0.2640 | 0.1138 | 0.0683 | 3.265 | 345 | 0.247 |
| `rec-g0.375` | 1e-2 | 0.2519 | 0.3602 | 0.6596 | 0.3471 | 0.0738 | 2.180 | 246 | 0.169 |
| `rec-g0.5` | 1e-4 | 0.3998 | 0.4114 | 0.7564 | 0.4060 | 0.0717 | 1.904 | 110 | 0.140 |
| `constant-train-mean` | — | 0.0118 | **0.0816** | 0.0791 | 0.1014 | 0.0645 | | | |
| `persistence` | — | 0.0005 | 0.0006 | 0.0006 | 0.0006 | 0.0007 | | | |
| `linear-inputs-ridge` | 1e-4 | 0.0012 | 0.0090 | 0.0043 | 0.0099 | 0.0129 | | | |

Inhibitory spikes are 207,005 in every arm (the feedback is I→E only, and `I_WTA_MAX = 2`
caps the inhibitory winners, so inhibitory activity is unchanged by construction); neurons
6–9 are silent in every arm (their incoming weights are negative on every live axon, as in
exp-025). Verdict against `ff-refit`: primary −9.48%, worst head +0.23% → **promising = true**.
Against `ff-shipped`: −8.40%.

### `k = none` (the health protocol) — `exp-026/comparison-k-none.json`

| arm | λ_rel | val primary | **test primary** | comfort | temp | power | spk/tick | patterns | cofire |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `ff-shipped` | — | 0.9222 | 0.8538 | 1.4013 | 0.9004 | 0.2596 | 5.628 | 165 | 0.852 |
| `ff-refit` | 1e-1 | 0.2233 | 0.2617 | 0.4755 | 0.2373 | 0.0724 | 5.628 | 165 | 0.852 |
| **`rec-g0.0625`** (val-selected) | 1e-1 | 0.2274 | 0.2822 | 0.5116 | 0.2626 | 0.0725 | 5.171 | 194 | 0.803 |
| `rec-g0.125` | 1e-1 | 0.2611 | 0.3132 | 0.5724 | 0.2947 | 0.0725 | 4.500 | 328 | 0.746 |
| `rec-g0.25` | 1e-1 | 0.3171 | 0.3476 | 0.6212 | 0.3471 | 0.0746 | 3.588 | 354 | 0.758 |
| `rec-g0.375` | 1e-1 | 0.3622 | 0.3893 | 0.6923 | 0.3999 | 0.0757 | 2.602 | 273 | 0.569 |
| `rec-g0.5` | 1e-2 | 0.4111 | 0.4186 | 0.7372 | 0.4408 | 0.0778 | 1.931 | 86 | 0.422 |

Inhibitory spikes 223,637 in every arm. Verdict against `ff-refit`: primary improvement −7.82%
(7.8% worse), worst head +10.66% → **promising = false**. The shipped head was trained with `k=4`, so at
`k=none` it sums over 5.6 spikes per tick and overshoots the `[0, 1]` targets (comfort MAE 1.40);
`ff-refit` is the fair feed-forward reference here.

### What the numbers say

1. **The relative benefit is real but small, and only in the K-WTA regime.** At `k=4` the
   eight live excitatory neurons fire in lockstep (identical 0.23 rates) and K-WTA keeps two
   of them per tick, so the feed-forward bank exposes 268 distinct patterns to the readout.
   Gain 1/8 is enough to break those ties without changing who fires how often (spikes per
   tick 3.591 → 3.568), and the refit head uses the extra 150 patterns. From gain 1/4 the
   feedback starts suppressing excitatory firing and the error climbs; at 3/8 and 1/2 the
   excitatory rates fall to 0.05 and 0.02 and the readout is worse than predicting zero.
   With K-WTA off there are no ties to break, so every gain is pure suppression.
2. **No arm beats a constant.** Predicting the train mean scores 0.0816; the best arm scores
   0.0967. The exp-025 head has no bias and reads one tick's binary spike vector, so its
   prediction is one of a handful of levels (about 0.73 when two excitatory neurons fire,
   about 0 when none do) against a smooth target. The one-tick recurrence here cannot add
   the missing bias or the missing resolution; it was never going to.
3. **The test split is harder than validation, for everyone.** The constant baseline goes
   from 0.0118 (val) to 0.0816 (test) because the last six test sessions hold the 11,997
   `gpu_temp_c == 0` sentinel rows (per-session comfort MAE for `ff-refit`: 0.06–0.08 on the
   first 23 sessions, 0.23–0.97 on the last six). Every arm's test number carries that shift,
   which is why the comparison is relative to `ff-refit` on the same rows and not to a
   threshold.
4. **The three-head argmax is degenerate on this data.** Every arm agrees with `ff-shipped`'s
   decision on 100% of test ticks and with the target's argmax on 92.9%: the `comfort` head
   wins almost always. This confirms the README's warning that the argmax is a diagnostic,
   not a policy; recurrence does not change it.
5. **Signed weights and dense export survive.** Every fitted head satisfies outgoing Dale
   with zero violations; the variant bundle's three hidden `.mem` images are byte-identical
   to `dataset/merged_v2`'s; the recurrence lands in one additional 256-word image under a
   versioned block rather than a changed layout.

### Latency

Host latency of the Python binary32 reference stepper, measured alone on the host over the
test split with `--gains 0.125` (`exp-026/timing.json`); this is harness time on a laptop
CPU, not FPGA latency and not a deterministic artifact:

| arm | µs per tick (test split) | relative |
|---|---:|---:|
| `ff` | 73.9 | 1.00 |
| `rec-g0.125` | 87.2 | 1.18 |

(Windows 11, CPython 3.13.12; train and validation splits give the same ratio within 2%.
The earlier seven-arm runs shared the host with each other and with a cargo build, so their
`timing.json` files are not quoted.)

Structurally the variant adds, per tick, up to one `16 × 16` multiply-accumulate on `s(t-1)`
(48 nonzero terms for this family), one 16-bit register holding `s(t-1)`, and one tick of
algorithmic delay before an inhibitory spike can act.

## Hardware evidence

None, and this directory claims none. The parity owner is
[silicon-hdl#76](https://github.com/rmems/silicon-hdl/issues/76) (open: the first
deliverable, a reproducible Basys 3 reference path, is pending; the 2026-09-13 board smoke is
a programmed-board heartbeat). silicon-hdl closed its signed output-weight path
([#72](https://github.com/rmems/silicon-hdl/issues/72)) and signed inhibitory path
([#73](https://github.com/rmems/silicon-hdl/issues/73)) on the HDL side on 2026-09-13; that is
RTL work, not board-measured parity for inhibition. Its
`docs/lif-array-connectivity-model.md` states that the wired SoC has **no feedback path from
the spike bitmap back into the input index**, so the FPGA cannot execute this variant today
and `parameters_recurrent_weights.mem` has no hardware consumer. The Rust crate does not
implement the block either: `tools/replay_core.py` is the only recurrent executor, and the
tests in `tests/test_recurrence.py` are host tests, not board proof.

## Reproduce

```bash
python3 tools/recurrence_eval.py --jsonl PATH/state_telemetry.jsonl \
    --input-label "hub:rmems/Spikenaut-SNN-Telemetry v3/state_telemetry/{train,validation,test}-00000.parquet projected to episode_id + five live columns, file order (see docs/recurrence/README.md)" \
    --out-dir out/k4 --emit-checkpoint out/k4/bundle
python3 tools/recurrence_eval.py --jsonl PATH/state_telemetry.jsonl --input-label "..." \
    --k none --out-dir out/k-none
sha256sum out/k4/comparison.json docs/recurrence/exp-026/comparison.json   # identical
python3 tools/replay_frozen.py --bank-manifest dataset/merged_v2_rec_v1/model_bank.json \
    --jsonl PATH/state_telemetry.jsonl --split test --out-dir out/replay
```

`comparison.json` carries no wall-clock field and is byte-identical across runs on the same
input; `report.md` is `recurrence_bundle.render_report` applied to it. The default
invocation without `--jsonl` runs the synthetic `tools/fixtures/recurrence/` method fixture,
which proves the pipeline and is not evidence.

## Files

| Path | Contents |
|---|---|
| `exp-026/comparison.json` | the `k=4` run (deterministic) |
| `exp-026/report.md` | its rendered table |
| `exp-026/comparison-k-none.json`, `exp-026/report-k-none.md` | the `k=none` run |
| `exp-026/timing.json` | host stepper wall-clock, `ff` vs `rec-g0.125`, measured alone |
| `../../dataset/merged_v2_rec_v1/` | the validation-selected variant checkpoint bundle |
