---
language:
- code
license:
- mit
- apache-2.0
tags:
- spiking-neural-networks
- neuromorphic
- fpga
- q88-fixed-point
- leaky-integrate-and-fire
- e-prop
- ottt
- interoception
- telemetry
pipeline_tag: other
model_name: Spikenaut-SNN-v2
---

# Spikenaut-SNN-v2

A 16-neuron Leaky-Integrate-and-Fire (LIF) spiking neural network **designed to learn** a compact temporal representation of machine state from live hardware and node telemetry, targeting a Xilinx Artix-7 FPGA. That is the intended architecture; nothing here demonstrates that the shipped weights learned such a representation — see Status below.

Spikenaut is the small-supervisor layer of a wider research program, **Artificial Interoception / Neuromorphic Supervisor** ([#7](https://github.com/rmems/Spikenaut-SNN/issues/7)). The name comes from "spike" (neural firing) and "naut" (navigator). This repository holds the model artifact — the shipped weight files and the Q8.8 export contract. Two caveats the name invites: the weights are not established as trained (Status, below), and the three output rows are Distill regression channels `(comfort, temp, power)`, not a bound RM-1150 action list.

## Status — read this first

This is a **research artifact, not a validated supervisor.** Five things a reader should know before using the shipped weights:

| | |
|---|---|
| **The live bank is exp-025, not the #2 ramp.** | `dataset/merged_v2/` is the Distill sidecar Dale health-PASS export (protocol pin [`a1fa491`](https://github.com/rmems/SynapticDistill.jl/commit/a1fa491c70397b96967ba6cf8f08c2ce2fbb2fb7), seed 123, 20 epochs, Hub v3 JSONL sha `26d7d744…`, legal 5-ch, lineage `74acdd0f`). Hidden weights are mixed-sign. Outgoing Dale is 12:4 (`inhibitory` on neurons 12–15). Health k=none on `gpu-000170..198`: cofire **0.863**, all-16 **0.000**, I_spikes **223511** (exp-009 PASS). The monotonic ramp [#2](https://github.com/rmems/Spikenaut-SNN/issues/2) tracked is the bank this promote replaces. |
| **Decay is Distill keep=0.85, not the old linspace.** | All 16 `parameters_decay.mem` words are `00DA` (0.8515625). That is sidecar keep semantics, not `torch.linspace(0.8, 0.95, 16)`. `tau = -dt / ln(decay)` is therefore the same 6.21 ms on every unit. |
| **The output layer is the Distill readout with a documented decision contract.** | 48 signed Q8.8 words, also present as per-neuron `output_weights` in `snn_model.json`, neuron-major 16×3. Distill trained the three rows as regression onto `(comfort, temp, power)` (`sample_readout_target` at sidecar `a1fa491`). `src/decision.rs` / `tools/decision_core.py` convert one finite score row through `replay_output_row`: argmax, lowest-index ties, fail-closed on NaN/empty/wrong-width and on overflowing derived margin/confidence. RM-1150's five-wide `ALLOW/WARN/THROTTLE/PAUSE/YIELD_GPU` list is unbound — mapping it onto these three channels would be a guess. Scratch Hamming on the published harness: k=none **14.960%**, k=4 **49.095%**, json↔mem hidden **0/256** — measurement only; no pass/fail threshold ([#20](https://github.com/rmems/Spikenaut-SNN/issues/20) still open). |
| **Weight provenance is the Distill sidecar pin.** | Trainer is `scripts/spikenaut_train.jl` at Distill `a1fa491`. This repository still does not run that script. The 8-record `fresh_sync_data.jsonl` import is the *previous* bank's story, not this one. Closed [#13](https://github.com/rmems/Spikenaut-SNN/issues/13) (2026-09-13) tracked that sidecar pin; it landed via Distill (exp-025 promote). |
| **FPGA parity is unproven; live smoke is not parity.** | [silicon-hdl#68](https://github.com/rmems/silicon-hdl/issues/68) is CLOSED: Phase C live smoke **PASS** on Basys 3 — PROGRAM_OK plus `step_en` ~heartbeat after BTNC in SW15 status mode. That is a programmed-board heartbeat, not software↔FPGA spike/action/membrane agreement. Spike, action, membrane-potential, and quantization error vs the software model have not been measured; [#6](https://github.com/rmems/Spikenaut-SNN/issues/6) stays open. The LUT/power/register/WNS rows below remain Vivado synthesis estimates, not board-measured. |

The architecture and the Q8.8 export contract are verifiable from the artifacts in this repository. The FPGA resource and power figures are **externally reported** — no RTL, constraints, Vivado project or synthesis reports are checked in here, so a reader cannot reproduce them from this artifact. Health bars and Hamming figures above are Distill-sidecar measurements on the pinned v3 holdout, not FPGA parity, and Hamming is not a gate.

## The research question

> Can an event-driven SNN maintain a useful temporal representation of an AI system's internal computational state and learn bounded supervisory behavior, while deterministic software remains responsible for hard safety constraints?

A second question: can a larger GPU temporal model teach useful supervisory behavior into a small SNN that eventually runs on dedicated FPGA hardware?

The program is explicitly designed to be able to fail. Demonstrating that a simple baseline beats the SNN, that online plasticity destabilizes it, or that FPGA quantization loses key behavior are all useful outcomes. The requirement is measured evidence, not preserving the hypothesis.

### System model

```text
ENVIRONMENT / BODY          hardware + workloads + market simulation
        ↓
SENSORY DATA                telemetry, node sync, paper trajectories
        ↓
TEMPORAL RESEARCH MODEL     LiquidCortex.jl and simple baselines
        ↓
DISTILLATION                SynapticDistill.jl
        ↓
SMALL SUPERVISOR            Spikenaut-SNN   ← this repository
        ↓
HARDWARE DEPLOYMENT         silicon-bridge → silicon-hdl / FPGA
        ↓
ACTION PROPOSAL
        ↓
HARD SAFETY SHIELD          deterministic Rust governor
        ↓
BOUNDED ACTUATION / SHADOW EVALUATION
        └──────────────────────────────────↺
```

### Core safety principle

**Learned systems propose; deterministic safety rules constrain.**

No learned SNN, LLM, FPGA controller, or online-training loop may disable or raise hard thermal thresholds, override a deterministic emergency pause, silently continue after invalid or NaN model state, or gain unrestricted host control before shadow-mode evidence exists.

## Architecture

| Spec | Value |
|------|-------|
| Neuron model | Leaky-Integrate-and-Fire (LIF) |
| Neurons | 16 |
| Input channels | 16 wide; **live** exp-025 map is five legal sensors on axons 0–4 with axons 5–15 held at zero (game-blind). Front end is `stim::LiveStimAdapter` (analog current). Public `TelemetryEncoder` / `CHANNEL_MAP` is a deprecated historical coin proposal and **must not** be paired with this bank — see [Inputs](#inputs--live-exp-025-map-primary) |
| Weight format | Q8.8 fixed-point |
| Learning rules | E-prop, OTTT, reward-modulated STDP — externally reported, see [Training provenance](#training-provenance) |
| Clock | 1 kHz (1 ms resolution) |
| Training speed | 35 µs/tick — externally reported |
| Memory footprint | 672 bytes (336 Q8.8 codes) |
| FPGA target | Xilinx Artix-7 xc7a35tcpg236-1 (Basys3) |

## Inputs — live exp-025 map (PRIMARY)

This is the mapping the live exp-025 `merged_v2` bank was trained on. Sidecar metadata names these five `legal_columns` and holds axons 5–15 at zero (`unused_axons`). It is **game-blind**: no game-id / title channels.

| Axon | Signal |
|---|---|
| 0 | `mem_util_pct` |
| 1 | `power_w` |
| 2 | `gpu_temp_c` |
| 3 | `sm_clock_mhz` |
| 4 | `mem_clock_mhz` |
| 5–15 | unused / held at zero in train |

Public `TelemetryEncoder` / `CHANNEL_MAP` is **not** this adapter. It is a deprecated 16-column coin proposal (DNX/Quai/Qubic/Kaspa/Monero/Ocean/Verus/Thermal). Pairing that encoder with the shipped weights is wrong: training held axons 5–15 at zero, while that encoder maps unrelated blockchain sources across all 16 channels and still emits its nonzero base rate on the unused axons. `TelemetryEncoder::for_shipped_merged_v2` refuses construction as a live adapter.

Rust names the live sensors as `encode::LIVE_COLUMNS` and now ships an encoder that matches their **columns**. `encode::LiveTelemetryEncoder` is a real, non-deprecated 5-column rate encoder: it takes a `[f32; 5]`, so a 16-wide coin-shaped frame is a compile error, and it drives axons 0–4 only — axons 5–15 are never written.

**It is still not this bank's front end, and it says so.** `merged_v2` was trained and evaluated on *analog current*, not on spikes: `tools/hamming_core.py` steps it as `input = W @ stim` and labels that line "analog current, not Poisson", and `tools/HAMMING_PROTOCOL.md` records the exp-024 condition as "analog current, Poisson unused when `learn=false`". Rate-coding the same five sensors changes the magnitude and temporal distribution of every input — most concretely, a normalized `0.0` encodes at the non-zero `BASE_RATE_HZ`, so an idle sensor stops reading as idle and zero stimulus is not representable at all.

So the crate refuses **two** wrong pairings, with two different diagnoses, and accepts exactly one:

| Constructor | Columns | Modality | Result |
|---|---|---|---|
| `TelemetryEncoder::for_shipped_merged_v2` | wrong (coin, 16-wide) | spikes | `Err(LiveMapMismatch)` |
| `LiveTelemetryEncoder::for_shipped_merged_v2` | right (`LIVE_COLUMNS`) | wrong (spikes) | `Err(SpikeModalityMismatch)` |
| **`LiveStimAdapter::for_shipped_merged_v2`** | right (`LIVE_COLUMNS`) | right (analog) | **`Ok`** |

`LiveTelemetryEncoder::new` still builds, because the encoder is correct for a consumer that actually eats spikes — the FPGA path — on the live five-sensor map.

### The bank's front end: `stim::LiveStimAdapter`

`stim::LiveStimAdapter` is the analog adapter for these weights. It takes the five raw sensors above — percent, watts, degrees Celsius, megahertz — and returns the 16-wide `[f32; 16]` **analog** `stim` vector that `v = decay * v + W @ stim` consumes. It is stateless: analog current is not a spike train, so there is no clock, no accumulator, and no `reset`.

- **Normalization** is per sensor, through the `frozen_minmax` spans the shipped sidecar records (`kinetic::LIVE_RAW_RANGES`, pinned to `dataset/merged_v2/snn_model.json`), clamped into `[0, 1]`. A GPU hotter than anything in training is a saturated axon, not a fault.
- **Raw samples are snapped to binary32 before the affine map**, because the bank's arithmetic is `f32` and the reference encoder snaps (`hamming_const.f32`). Skipping that snap lands the result on a different `f32` for 27–36% of uniform in-span readings, depending on the sensor — a one-ulp error, about `3e-8` at mid-span, which any tolerance looser than that absorbs silently. The parity tests compare exact bit patterns for this reason.
- **Axons 5–15 are exactly `0.0`**, matching the sidecar's `unused_axons: "5:15"`. The adapter only ever writes axons 0–4, so an `UNUSED-AXON LEAK` is unrepresentable rather than merely detectable. A missing or null sensor, and a reading at or below its frozen minimum, encode to the same literal zero — which is the value the contract is written in terms of, and the one value the rate encoder cannot express at all.
- **Non-finite readings are refused whole**, naming every offending sensor (`NonFiniteLiveFrame`), along with a finite `f64` too large for the `f32` grid. Nothing is substituted.
- **`Input → Linear` is the node that consumes it.** The graph's `Linear` node carries the learned 16×16 matrix and computes `I = W @ stim`; because axons 5–15 are exactly zero, the five live axons account for every bit of that current. `tests/live_stim.rs` asserts the round trip. On this bank the unused axons are inert twice over: the `snn_model.json` weights on columns 5–15 are training residue far below the Q8.8 grid — none is exactly zero, the largest is ≈ `8e-15` against a grid step of `1/256` — so the decode snaps every one to zero, and the `Linear` node could not be moved by those axons even if something wrote to them.

Parity with the reference is pinned across both languages: `tools/live_stim_parity.py` runs `tools/hamming_encode.py` over `tools/fixtures/live_stim/reading.jsonl` and writes `expected_stim.json`; `tests/live_stim.rs` reads the same two files and asserts the Rust adapter reproduces every value exactly — bit pattern for bit pattern, not within a tolerance. CI runs both halves, so for every input class the fixtures cover, neither implementation can move without the other failing.

Readings JSON cannot express — `NaN`, the infinities, an `f64` past the binary32 ceiling, negative zero, subnormals — travel in the pin as raw IEEE-754 bits, each recording whether the reference *refused* it. Without them the refusal contract would have no cross-language coverage at all, since a JSONL row carrying one would make the pin generator reject the whole file. The same tool also pins Python's `FROZEN_MINMAX` against `dataset/merged_v2/snn_model.json`, mirroring what `shipped_bank_frozen_minmax_matches_live_raw_ranges` already did for Rust. One divergence is known and uncovered: a degenerate span (`max <= min`) short-circuits in Rust and divides by a negative width in Python. It is unreachable on this bank — the shipped spans are strictly increasing and now pinned on both sides — and is documented rather than papered over.

This is the input side only. It is not a claim that this crate runs the shipped bank — `tools/hamming_lif.py` remains the only place in this repository that steps *these weights* through a membrane. `src/neuromod_host.rs` steps a default published `neuromod::LifNeuron` and a synthetic non-negative `SpikingNetwork`, neither of which sees the shipped weights. This also does not change FPGA parity ([#6](https://github.com/rmems/Spikenaut-SNN/issues/6)), where `LiveTelemetryEncoder` stays the right front end for a spike-consuming consumer.

A live 5-column **kinetic** encode path exists on top of `LiveTelemetryEncoder`: `kinetic::LiveKineticFrontEnd` runs one causal `kinetic-signals` pipeline per live sensor and encodes through `LiveTelemetryEncoder`, targeting axons 0–4 with axons 5–15 held at zero. Being a spike path, it inherits the modality caveat above — it is not the shipped bank's front end either. Its projection is the identity — each sensor's own raw value, normalised against the `frozen_minmax` span the sidecar records for it, lands on its own axon. The eleven kinetic features come back alongside the frame as audit data and **do not** reach an axon: which of them (if any) earns one is the RAW / KINETIC / HYBRID ablation, which remains open — [#14](https://github.com/rmems/Spikenaut-SNN/issues/14). The encode path is host-side only and does not block FPGA parity.

Non-finite samples are rejected whole and never substituted, by the analog adapter, by both rate encoders, and by the kinetic front end: a single `NaN` sensor rejects the five-wide reading before any estimator or accumulator moves. Dropout **sentinels** are a different problem and none of the three front ends solves it — `gpu_temp_c == 0` is a finite number and passes straight through. Masking it is the caller's job under the state contract in [#20](https://github.com/rmems/Spikenaut-SNN/issues/20).

Every figure quoted across [#2](https://github.com/rmems/Spikenaut-SNN/issues/2), [#3](https://github.com/rmems/Spikenaut-SNN/issues/3), [#4](https://github.com/rmems/Spikenaut-SNN/issues/4) and closed [#13](https://github.com/rmems/Spikenaut-SNN/issues/13) was measured on these five GPU sensors. A cofire or Hamming figure is unreadable without knowing it was five channels rather than sixteen.

Two consequences worth being explicit about. `vram_temp_c` is excluded because it is exactly `gpu_temp_c + 8` on every non-dropout row, so admitting it would leak the thermal signal into itself. And `gpu_temp_c == 0` is a dropout sentinel, not a cold GPU — it must be masked, which is the same rule the state adapter below states.

**The live map is not a claim that 16 axons must be filled.** The replacement state contract is being defined in [#20](https://github.com/rmems/Spikenaut-SNN/issues/20), under one governing rule: logical state variables are *not* equivalent to physical SNN axons. Signals are never invented or duplicated just to fill 16 slots. Instead an explicit adapter sits between them:

```text
raw state → state adapter (optional kinetic-signals) → encoder → fixed-width SNN stimuli
```

The adapter accepts a variable number of legitimate source signals, so the raw feature count can change without the input contract breaking. Every signal must declare its unit, source of truth, timestamp and sampling semantics, valid range, normalization, missing-value and staleness behavior, and provenance. Missing signals are masked, never silently zeroed. Axons 5–15 at zero on this bank are unused width, not a license to invent filler channels.

### Tier A stream-READY candidates (not shipped)

This is a **stream-readiness board**, not a new Inputs map and not an axon fill. The live bank is still axons 0–4 (`LIVE_COLUMNS`). Unused axons stay **0**. No Stage-1 EXP, no retrain, no bank rewrite.

Re-scored 2026-09-13 on [#20](https://github.com/rmems/Spikenaut-SNN/issues/20) after producer [gaming-telemetry#27](https://github.com/rmems/gaming-telemetry/pull/27) @ `07b19f3` (null-on-miss on `main`) and ETL fail-loud [spikenaut-telemetry-etl#21](https://github.com/rmems/spikenaut-telemetry-etl/pull/21) + [#23](https://github.com/rmems/spikenaut-telemetry-etl/pull/23) on `main`.

**READY** (stream / candidate for a future named Stage-1 slot, one-by-one):

| Axon | Signal | Verdict |
|---|---|---|
| 6 | `memory_used_mb` | **READY**. Distinct from live axon 0 `mem_util_pct`. `mem_bw_util` is **BLOCKED** (not collected — do not invent it). |
| 7 | `pcie_tx_kbps` / `pcie_rx_kbps` | **stream-candidate / projection TBD**. Each of `pcie_tx_kbps` and `pcie_rx_kbps` is independently **READY** to stream (null = miss; 0 = observed idle is OK). Axon 7 carries one scalar per frame, so axon fill needs a named Stage-1 projection (`tx`, `rx`, `max(tx,rx)`, or `tx+rx`; null if either or both missing -- fail-loud, never invent). Until that EXP, axon 7 stays unused (0) on the live bank. |
| 8 | `fan_speed_perc` | **CONDITIONAL READY**. Stream OK with nulls; axon fill still needs train/val/test variance (Hub mining fans were near-constant). |

**BLOCKED** (collector schema absent — **DENY inventing** these axons):

| Axon | Signal | Why |
|---|---|---|
| 5 | `gpu_util_pct` | **BLOCKED**. Schema gap. Producer has only `encoder_util_perc` / `decoder_util_perc`. Do not substitute encoder/decoder util. |
| 9 | `cpu_util_pct` | **BLOCKED**. Schema gap. Host thermal/power (`cpu_tctl_c`, CCD, `cpu_package_power_w`) are Tier B-ish, not util%. |

Implementation order stays: (1) this contract, (2) ETL availability, (3) named Stage-1 EXP adding READY slots one-by-one — **no silent bank overwrite**. Work metrics stay labels / Stage-4, not axons.

Context only (live 0–4 path, not this board): bank pin [#47](https://github.com/rmems/Spikenaut-SNN/pull/47) @ `6965e12a`; Inputs honesty [#49](https://github.com/rmems/Spikenaut-SNN/pull/49); analog front end [#52](https://github.com/rmems/Spikenaut-SNN/issues/52) / [#54](https://github.com/rmems/Spikenaut-SNN/pull/54) (`LiveStimAdapter` still writes axons 0–4 only).

### Historical / non-live proposal — coin CHANNEL_MAP

The table below is the proposed v2 layout implemented by public `TelemetryEncoder` / `CHANNEL_MAP`. It is **not live**. It disagrees with the deleted training-time encoder in every column, and neither historical map is the exp-025 train mapping.

| Channels | Data Source | Function |
|----------|------------|----------|
| 0-1 | DNX (Dynex) | PoUW solver health and neural baselines |
| 2-3 | Quai | Live on-chain reflex and sync confidence |
| 4-5 | Qubic | Epoch and tick cadence monitoring |
| 6-7 | Kaspa | High-frequency DAG settlement tracking |
| 8-9 | XMR (Monero) | Node stability and CPU L3 cache contention |
| 10-11 | Ocean | Data liquidity and staking prep |
| 12-13 | Verus | CPU-heavy validator tracking (AVX-512) |
| 14-15 | Thermal | Pain receptors — power and temperature |

Channels 14-15 are *intended* as the network's pain receptors. **This is design intent, not shipped behaviour.** The repository now carries code, but none of it is a runtime. `tools/` verifies the Q8.8 export, `src/` lifts the model into a NIR graph, and `src/encode.rs` does read channels 14-15 on this historical map — it turns them into spikes and can report that a rejected frame touched them. That is routing and diagnostics, not a response: there is still no reward signal, no online weight update, and nothing that acts on a temperature reading. The 85 °C threshold also belongs to `thalamic-relay`, a peer process listed below, not to the SNN. Wiring a thermal penalty into training is future work.

### Historical / non-live — deleted generate_spike_data.py encoder

A second, different layout is recorded in this repository's history, and the two historical maps disagree in every column — so at most one of them can describe a past encoder, and neither describes the live bank. The historical encoder — `dataset/generate_spike_data.py`, deleted in `50a2627` but recoverable from history — declares:

| Channels | Training-time signal | coin CHANNEL_MAP says |
|---|---|---|
| 0-3 | `kaspa_hashrate`, `kaspa_power`, `kaspa_temp`, `kaspa_qubic` | DNX, Quai |
| 4-7 | `monero_hashrate`, `monero_power`, `monero_temp`, `monero_qubic` | Qubic, Kaspa |
| 8-11 | `qubic_hashrate`, `qubic_power`, `qubic_temp`, `qubic_qubic` | XMR, Ocean |
| 12-15 | `thermal_stress`, `power_efficiency`, `network_health`, `composite_reward` | Verus, Thermal |

Every column is assigned differently. That disagreement is history about two non-live maps. No training run in this repository links that deleted encoder or `fresh_sync_data.jsonl` to the live matrix; the live mapping is the five-column Distill layout above, not this table and not `TelemetryEncoder`. The coin table is not a safe guide to the shipped weights.

Running that encoder end to end over the eight records — including `create_spike_train`, which allocates `np.zeros(16)` per record and overwrites only the channels that event touches — gives its **`normalized_values` matrix**. This is the rate the encoder assigns each channel, not the spikes it emits; the distinction matters and is taken up below the table.

| ch | name | normalized value (0-1), records 1-8 | |
|---|---|---|---|
| 0 | `kaspa_hashrate` | 0.420, 0.450, 0.480, 0.500, **0, 0, 0, 0** | live |
| 1 | `kaspa_power` | 0.380, 0.403, 0.438, 0.458, **0, 0, 0, 0** | live |
| 2 | `kaspa_temp` | 0.883, 0.850, 0.817, 0.783, **0, 0, 0, 0** | live |
| 3 | `kaspa_qubic` | 1.0, 1.0, 1.0, 1.0, **0, 0, 0, 0** | binary flag |
| 4-6 | `monero_*` | **0, 0, 0, 0**, then 0.30 – 0.70 | live |
| 7 | `monero_qubic` | **0, 0, 0, 0**, 0.800, 0.900, 0.950, 1.000 | live |
| **8-11** | `qubic_*` | **all zero — no `qubic` record exists** | **dead** |
| **12** | `thermal_stress` | **all zero** | **dead** |
| 13 | `power_efficiency` | 0, 0, 0.006, 0.015, 0, 0, 0, 0 | barely live |
| 14 | `network_health` | 1.0 ×4, then 0.900, 0.950, 0.975, 1.000 | live |
| 15 | `composite_reward` | 0.9991, 0.9998, 0.9999, 1.0, 0.9999, 0.99996, 0.999997, 1.0 | near-constant |

Two things about this table are easy to get wrong, and I got both wrong before review caught them.

**It is rates, not spikes.** `temporal_encoding` turns a normalized value into `spike_rate = normalized * 100` Hz, then `spike_prob = spike_rate / 1000` per 1 ms tick, then draws `1 if np.random.random() < spike_prob else 0`. So the emitted `spike_vector` is a Bernoulli sample, and even a channel pinned at normalized `1.0` fires with probability **0.1** per tick. Channel 3 sitting at 1.0 across its four kaspa records yields roughly **0.4 expected spikes**, not four. Nothing here licenses a claim about the realized spike train.

There is also **no `np.random.seed` anywhere in the encoder**, so rerunning it reproduces the rate table above exactly and its `spike_vector` never. Stated narrowly on purpose: that is a fact about this script, not about the shipped weights. Since nothing here links this encoder to `merged_v2`, it does not establish that these were the training stimuli, nor that the real ones are lost — they may have been generated or retained somewhere outside this repository. What it does mean is that re-deriving the stimuli *from this script* is not a route to reproducing them.

**The zero-fill still holds, and holds for both representations.** A kaspa event never writes channels 4-11, a monero event never writes 0-3 or 8-11, so **every one of channels 0-7 is zero on half the records** — and a normalized 0 gives `spike_prob` 0, meaning those halves emit no spikes at all, deterministically. Channel 3 is the extreme case: `1,1,1,1,0,0,0,0` in rate terms, so it can only ever fire on a kaspa record, though on any given run it will mostly not fire at all. Asymmetric, not a clean flag. Channels 0-2 and 4-7 carry the same asymmetry folded into their magnitude.

**Five of sixteen channels are dead, not four.** Channels 8-11 have no source records at all. Channel 12 is dead for a different and more interesting reason: it is computed as `(gpu_temp_c - 40) / 6`, giving roughly 0.30-0.88, and then passed to `temporal_encoding(..., 'temp')`, which normalizes with `(value - 40) / 6` *again* and clips at zero. The double subtraction lands every record at zero. Channel 13 carries the same second normalization — `power_eff / 5` lands near 0.5, and the `'hashrate'` branch then subtracts exactly 0.5 — so six of the eight records clip to zero. It survives on records 3 and 4 alone, where `power_eff / 5` is 0.505806 and 0.515066 and genuinely clears the threshold. That residual activity is real signal squeezed through a wrong subtraction, not a rounding artifact.

Channels 14 and 15 do carry signal that 0-11 do not: channels 0-11 read only `hashrate_mh`, `power_w`, `gpu_temp_c` and `qubic_tick_trace`, while 14 additionally consumes `qubic_epoch_progress` and 15 consumes `reward_hint`. Those two fields reach the network **only** through 14 and 15, so neither channel is redundant — a point that matters when deciding what to keep or repair in a retrain.

What is left, then, is a 16-wide input in which five channels are always zero — deterministically, in rates and spikes alike — one is near-constant, **eight (channels 0-7) are silent on exactly half the records** — each is written on 4 of the 8, since `blockchain` routes 0-3 to kaspa records and 4-7 to monero — and every record makes exactly eight *attempted* writes of the sixteen — but two of those encode to zero, so the actual count of nonzero normalized inputs is **six** on records 1, 2 and 5-8 and **seven** on records 3-4. The emitted spike vector is sparser still, since each nonzero value is only a per-tick probability. That is consistent with the degenerate ramp described below, though — like the ramp's cause — not a link this repository can demonstrate, since no training run here connects the dataset to the parameters.

That deleted encoder is historical only. The live mapping is the five-column table at the top of this section.

## Merged v2 parameters

| Parameter | Source | Values |
|-----------|--------|--------|
| Thresholds (16) | Distill sidecar (exp-025) | Twelve cells ≈1.60 (`019A`/`0199`); four cells 0.45 (`0073`, neurons 6–9) |
| Decay rates (16) | Distill keep=0.85 | All `00DA` = 0.8515625 |
| Hidden weights (256) — JSON float census | Distill sidecar — **mixed-sign, not a ramp** | Unquantized JSON: range −1.0 to +1.5999; 116 negative / 140 positive |
| Hidden weights (256) — shipped Q8.8 census | `parameters_weights.mem` (FPGA / Rust graph) | 19 negative / 50 positive / **187 zero**. Do not treat the float census as FPGA sparsity. |
| Output weights (48) | Distill sidecar readout (also in JSON) | Signed; neurons 12–15 inhibitory (`0000`/`FFF7`/`FFE9` family) |

The hidden matrix is no longer the #2 linear ramp. The *previous* bank increased each neuron's 16 weights by exactly one Q8.8 step (`0x0001`, 0.0039):

```text
Neuron 0:  00C0, 00C1, 00C2, 00C3, ... 00CF   (+1 each)   # old bank only
Neuron 1:  00C4, 00C5, 00C6, 00C7, ... 00D3   (+1 each)   # old bank only
```

That formula does **not** reproduce the live files. [#2](https://github.com/rmems/Spikenaut-SNN/issues/2) attributed the ramp to degenerate training convergence — too few samples, no inhibitory connections, and identical E-prop/OTTT gradients across neurons. **Externally reported, like the rest of that diagnosis of the old bank.**

One clause of that diagnosis **cannot be assessed from this repository**, independently of provenance. No learning rule is implemented here, so nothing in this artifact executes e-prop or OTTT. What the repository *does* establish is the topology the clause turns on: `src/graph.rs` builds three edges, `Input → Linear → LIF → Output`, with no back edge and fixed thresholds.

What nothing here establishes is that the two rules *coincide* on a layer of that shape. The published work reports a similar descent direction for particular spike-representation mappings and variants, under stated assumptions about feedback routing, surrogate gradient, loss and time horizon — which is narrower than an identity of eligibility trace, surrogate factor and learning-signal modulation. This card cites no primary source for the stronger form and implements neither rule to test it, so it should not assert one. "Identical E-prop/OTTT gradients" is therefore a clause this card can neither confirm as a check that **passed** nor accept as a cause of the ramp; it stays recorded as [#2](https://github.com/rmems/Spikenaut-SNN/issues/2)'s claim, unassessed.

What *is* verifiable here is the ramp itself and that it is not an export bug. The export path was independently cross-validated and confirmed correct ([#4](https://github.com/rmems/Spikenaut-SNN/issues/4)).

### Q8.8 fixed-point format

All `.mem` files use Q8.8 fixed-point encoding. Each line is one 4-digit hex value:

```text
Hex: 0100  →  Decimal: 256  →  Float: 256/256 = 1.0
Hex: 00DA  →  Decimal: 218  →  Float: 218/256 = 0.852
Hex: 00CC  →  Decimal: 204  →  Float: 204/256 = 0.797
```

Negative values use two's complement: `FFF9` = -0.027.

## Files

```text
config.json                        # Machine-readable header: neuron and
                                   # channel counts, weight format, clock

dataset/merged_v2/
├── model_bank.json                # One-entry attested bank wrapping this checkpoint (RM-1327)
├── parameters.mem                 # 16 neuron thresholds (Q8.8 hex)
├── parameters_decay.mem           # 16 decay rates (Q8.8 hex)
├── parameters_weights.mem         # 16x16 weight matrix (Q8.8 hex)
├── parameters_output_weights.mem  # Output layer weights (signed Q8.8)
└── snn_model.json                 # Full model definition (float values)

tools/                             # Python package, standard library only
├── verify_q88.py                  # Q8.8 encoding verifier (#4)
├── verify_model_bank.py           # Manifest-backed model-bank attestation (RM-1327)
├── measure_hamming.py             # float-vs-Q8.8 Hamming holdout (#39):
                                   # CLI; keep-LIF stepper is hamming_core.py;
                                   # --self-test proves it can fail.
                                   # Measurement, not a pass/fail gate.
├── live_stim_parity.py            # Cross-language pin for the analog stim
                                   # contract (#52): reference encoder vs
                                   # src/stim.rs, on a shared fixture. CLI;
                                   # the pin itself is live_stim_pin.py and
                                   # --self-test is live_stim_selftest.py.
├── decision_parity.py             # Cross-language pin for the output-row
                                   # decision contract (RM-1328):
                                   # decision_core.py vs src/decision.rs.
                                   # Distill (comfort, temp, power); RM-1150
                                   # five-wide list unbound.
├── fixtures/live_stim/            # reading.jsonl + expected_stim.json, read
                                   # by both live_stim_parity.py and
                                   # tests/live_stim.rs
├── fixtures/model_bank/           # Golden valid + negative bank fixtures (RM-1327)
└── fixtures/decision/             # expected.json golden pin for
                                   # replay_output_row (RM-1328)

src/                               # Rust, `spikenaut-snn`
├── lib.rs                         # Crate root: what the library exposes
├── model.rs                       # Decodes snn_model.json, validated
├── graph.rs                       # Builds the NIR graph
├── encode.rs                      # LIVE_COLUMNS + LiveTelemetryEncoder
                                   # (exp-025 5-col, PRIMARY, axons 0-4);
                                   # refuses the shipped bank twice over --
                                   # wrong columns (coin) and wrong modality
                                   # (spikes vs analog current); not a runtime
├── stim.rs                        # LiveStimAdapter: the shipped bank's
                                   # front end. Raw 5 sensors -> the 16-wide
                                   # analog stim vector `W @ stim` consumes,
                                   # axons 5-15 exactly 0.0; pinned against
                                   # the Python reference encoder (#52)
├── decision.rs                    # Output-row -> decision contract
                                   # (RM-1328): replay_output_row, lowest-
                                   # index ties, fail-closed NaN/width
                                   # and overflowing derived confidence
├── kinetic.rs                     # Host-side kinetic-signals front end
                                   # upstream of encode.rs; encodes against
                                   # the live 5-col contract (axons 0-4,
                                   # 5-15 at zero); host-side only, FPGA
                                   # parity not blocked; does not replace
                                   # axon-encoder
├── neuromod_host.rs               # Host-side neuromod 0.6 experiments:
                                   # LifNeuron compatibility, seeded
                                   # non-negative R-STDP network, sparse GIF
├── critic.rs                      # Checked limbic-critic TD adapter
├── wiring.rs                      # Deterministic 12:4 Dale recurrent
                                   # topology experiment; not the bank matrix
├── ipc.rs                         # Validated corpus-ipc messages;
                                   # typed JSON only, no transport
├── silicon.rs                     # Checked silicon-bridge 0.3 signed Q8.8
                                   # export; KxN exporter to NxK HDL adapter
├── training.rs                    # Optional plasticity-lab 0.2 session over
                                   # the synthetic seeded HostNetwork only
└── json.rs                        # Strict reader, so the dependency list
                                   # stays at what Cargo.toml declares
```

The artifacts are the product; the code exists to check them and to hand them
to consumers in a standard form. The Rust crate still does not run the
**shipped exp-025 bank** — `Neuron::membrane_potential` is decoded and never
advanced. `HostNetwork` and `HostGifLayer` are explicitly separate experiments
with synthetic non-negative weights / GIF dynamics; neither is the bank.
The optional `HostTrainingSession` applies `plasticity-lab` only to that
synthetic `HostNetwork`; its in-memory weight deltas are not a new exp-025
checkpoint and are not written into `dataset/merged_v2/`.
`stim::LiveStimAdapter` builds the input the network would eat; it does not
step the shipped weights. `decision::replay_output_row` converts one
already-scored Distill row into a shadow-policy decision; it does not
actuate the host.
`tools/measure_hamming.py` is the documented exception: it publishes
float-vs-Q8.8 Hamming on a holdout via a standard-library **keep-LIF**
stepper in `tools/hamming_core.py`
([#39](https://github.com/rmems/Spikenaut-SNN/issues/39)). That is not a
claim `src/` executes spikes, and it is not a Hamming pass/fail gate — the
tolerance is deferred to [#20](https://github.com/rmems/Spikenaut-SNN/issues/20). Protocol:
`tools/HAMMING_PROTOCOL.md`.

### Loading on FPGA

```verilog
// Load thresholds from Q8.8 hex file
reg [15:0] threshold_ram [0:15];
initial $readmemh("dataset/merged_v2/parameters.mem", threshold_ram);

// Load weights from Q8.8 hex file
reg [15:0] weight_ram [0:255];
initial $readmemh("dataset/merged_v2/parameters_weights.mem", weight_ram);
```

### Checked FPGA parameter export

[`silicon-bridge`](https://crates.io/crates/silicon-bridge) 0.3.0 now validates
and encodes the full shipped bank through its rejecting **Checked signed Q8.8**
path. `export_shipped_fpga_image()` returns all four memory images without
overwriting the vault:

```rust
use spikenaut_snn::{FPGA_MEM_FILENAMES, export_shipped_fpga_image};

let image = export_shipped_fpga_image()?;
let files = image.mem_files();
assert_eq!(files.each_ref().map(|file| file.name), FPGA_MEM_FILENAMES);
assert_eq!(image.weights[6 * 16] as u16, 0xFF00); // signed -1.0 survives
# Ok::<(), spikenaut_snn::SiliconExportError>(())
```

The crate validates readout weights as KxN (outputs by neurons), while the
checked-in vault and silicon-hdl `OutputLayer` consume NxK (neuron by output).
The adapter performs both transposes explicitly; its contract test regenerates
all 336 words and matches the four committed `.mem` files byte-for-byte. The
dependency uses default features disabled, so the optional UART feature stays
disabled. This deterministic export does not prove live UART or FPGA parity;
the connected-board protocol and software-vs-hardware outputs remain separate
evidence gates.

### Optional host training experiment

The `training` Cargo feature adopts [`plasticity-lab`](https://crates.io/crates/plasticity-lab)
0.2 from crates.io. It wraps the existing caller-seeded, synthetic
`HostNetwork`, uses one persistent RNG stream across the whole batch, and
returns the published `TrainingSummary`, including the exact per-weight deltas
that were applied:

```rust
use spikenaut_snn::{HostTrainingSession, TrainingConfig, TrainingExample};

let mut session = HostTrainingSession::new(99, TrainingConfig::default());
let summary = session.run_session(&[
    TrainingExample { stimuli: vec![1.0; 16], reward: 0.0 },
    TrainingExample { stimuli: vec![0.0; 16], reward: 10.0 },
])?;
assert!(summary.weight_drifts.iter().flatten().any(|&delta| delta != 0.0));
# Ok::<(), spikenaut_snn::TrainerError>(())
```

Run it with `cargo test --locked --features training`. The feature is off by
default: this is an M3 host experiment, not a runtime for the signed exp-025
bank. It neither exports Q8.8 nor hands a checkpoint to `silicon-bridge`, so
the Julia Distill sidecar remains the only artifact-producing trainer until a
separately validated parity/export path exists.

## Training provenance

**Live bank (exp-025).** Distill sidecar `scripts/spikenaut_train.jl` at `a1fa491`, seed **123**, **20** epochs, Hub v3 JSONL sha `26d7d7442605a32b11330f53f55e621750f31e775465209991f73f94a6c72c09` (805781 lines), legal 5 columns, frozen minmax lineage `74acdd0f`, episode split train `gpu-000000..138` / test `gpu-000170..198`. This repository still does not run that script; the five files under `dataset/merged_v2/` are the exported bank.

**Historical record of the previous bank** (the #2 ramp this promote replaces). The figures below come from that earlier import; they are recorded here for continuity, not as the live provenance.

| Metric | Value |
|--------|-------|
| Architecture | Julia-Rust hybrid |
| Algorithm | E-prop + OTTT |
| Convergence | 20 epochs |
| Training speed | 35 µs/tick |
| IPC overhead | 0.8 µs |
| Memory usage | 1.6 KB — **disagrees with the shipped artifact**, which is 336 Q8.8 codes = 672 bytes |
| Training date | 2026-03-22 |
| Training data | `fresh_sync_data.jsonl` — **8 records**, Kaspa + Monero mainnet sessions |

**The learning rules disagree with each other.** The spec table at the top of this card lists three — E-prop, OTTT and reward-modulated STDP — while the **Algorithm** row above records only the first two. The STDP claim traces to an earlier revision of this document ([`8676f56`](https://github.com/rmems/Spikenaut-SNN/commit/8676f56), `| Learning | Reward-Modulated STDP |`) rather than to the record reproduced here. Gradient-based e-prop/OTTT and reward-modulated STDP are different training regimes; nothing in this repository reconciles them, and no training run here links either to the shipped weights. Both are left standing because an unresolved conflict in the historical record is itself part of the provenance — deleting one side would make the record look settled.

**Externally reported diagnosis, not established here.** [#2](https://github.com/rmems/Spikenaut-SNN/issues/2) attributes the ramp to the 8-record set: two of its six features (`qubic_epoch_progress`, `reward_hint`) are effectively constant — range 0.0009, standard deviation 0.000284 — while dominating spike encoding at a reported 87.5% spike rate each. **Both halves of that last clause are wrong against the encoder**, and are recorded here only because #2 states them: `qubic_epoch_progress` has no channel of its own — it reaches the network only after being averaged with `qubic_tick_trace` on channel 14 — and neither field produces 87.5%. The `'qubic'` branch sets `spike_rate = clip(value, 0, 1) * 100`, giving channel 14 a mean of **97.80 Hz** and channel 15 **99.98 Hz**, which at the 1 ms event step is a spike probability near 0.10, not 0.875. The constancy claim does hold. But since no training run in this repository links that dataset to the shipped matrix, the causal claim cannot be checked from here. What *is* verifiable from the artifact is the ramp itself.

The issue also cites monotonically converging sync data (`0.999912 → 1.0`) as producing single-attractor weights. That part does not survive checking, though not because the data is missing: `sync_percent` **is** present, on records 5-8, carrying exactly `0.999912, 0.999967, 0.999997, 1.0`. It is excluded because `encode_single_event` never references `sync_percent`: it binds `timestamp`, `telemetry` and `blockchain`, and every channel *value* comes from a field inside `telemetry`. Note that top-level `blockchain` **does** reach the network — not as a channel value but as routing, selecting whether channels 0-3, 4-7 or 8-11 are written at all, which is what produces the zero-fill pattern above. `sync_percent`, `block_rate` and `blocks_accepted` are the ones never read. The nearest thing that *did* — `reward_hint` on channel 15, ranging 0.9991 to 1.0000 — is a real near-constant input and is a better candidate for the same argument.

A replacement corpus, `qubic_ticks_snn.jsonl` (~27,430 records), and a data adapter are reported in [#2](https://github.com/rmems/Spikenaut-SNN/issues/2), but **neither is present in this repository or anywhere in its history** — treat both as external and currently uninspectable from the model card. The live retrain used the Hub v3 JSONL pin above, not that 27k file. Distill sidecar code (#34 / #37) no longer pins `W_MIN = 0` or writes unsigned Q8.8. Closed [#13](https://github.com/rmems/Spikenaut-SNN/issues/13) tracked that sidecar work; the pin is the live bank above, not an open trainer ticket.

## Known limitations

- **#2 ramp is no longer the live bank.** The linear-ramp matrix was verifiable on the previous artifact and is gone from `dataset/merged_v2/`. Attribution of that ramp to an 8-record set remains **externally reported** history. Recurrence ([#3](https://github.com/rmems/Spikenaut-SNN/issues/3)) is still out of scope. [#2](https://github.com/rmems/Spikenaut-SNN/issues/2). Closed [#13](https://github.com/rmems/Spikenaut-SNN/issues/13) recorded the Distill sidecar path that replaced the ramp.
- **Decay is uniform keep=0.85.** The old `torch.linspace(0.8, 0.95, 16)` placeholders are not the live file. All 16 words are `00DA`; `tau = -dt/ln(decay_rate)` at `dt = 1 ms` is **6.21 ms** on every unit.
- **Outgoing Dale, no recurrence.** Hidden weights are mixed-sign; sidecar `inhibitory` marks neurons 12–15 on the readout (12:4). That is not incoming-Dale recurrence and not K-WTA in the NIR graph — train-time K-WTA is sidecar metadata (`k_wta: 4`). The gap that remains is **recurrent** memory, not temporal state as such: each LIF still keeps a decaying membrane. [#3](https://github.com/rmems/Spikenaut-SNN/issues/3)
- **FPGA spike/action/membrane parity vs software is not done.** Phase C live smoke ([silicon-hdl#68](https://github.com/rmems/silicon-hdl/issues/68), CLOSED) is **PASS**: PROGRAM_OK + `step_en` ~heartbeat after BTNC in SW15 status mode on Basys. That is not FPGA parity, not a Dale inhibitory proof on board, and not a measurement of the power/LUT rows below. Spike agreement, action agreement, membrane-potential error, and quantization error against the software model have not been measured. [#6](https://github.com/rmems/Spikenaut-SNN/issues/6) stays open.
- **Float-vs-Q8.8 Hamming is published as a measurement, not a gate.** `tools/measure_hamming.py` reports per-tick Hamming (%) and mean bits for `k=none` and `k=4` with the full protocol (weights, encoder, episodes, seed). exp-025 scratch (this bank): k=none **14.960%**, k=4 **49.095%**, json↔mem hidden **0/256**. exp-024 claimed `k=none` 13.187% / 0.1608 bits and `k=4` 56.188% / 1.697 bits on the exp-023 PASS Distill knobs scratch (seed 123 / 5 ep), legal 5-ch train-scaled encoder, frozen minmax lineage `74acdd0f`, v3 test `gpu-000170..198` (n=117653). The in-repo harness run is a method fixture, not a reproduction of either scratch. A pass threshold is deferred to [#20](https://github.com/rmems/Spikenaut-SNN/issues/20). [#39](https://github.com/rmems/Spikenaut-SNN/issues/39), [#4](https://github.com/rmems/Spikenaut-SNN/issues/4)
- **Checked export is not hardware parity.** `silicon-bridge` 0.3.0 now rejects malformed or out-of-range parameters and preserves signed hidden/readout words; the in-repo adapter reproduces all four committed `.mem` files byte-for-byte. Its UART feature stays disabled here, and an exact parameter image does not establish spike/action/membrane agreement on the connected FPGA. [#15](https://github.com/rmems/Spikenaut-SNN/issues/15), [#6](https://github.com/rmems/Spikenaut-SNN/issues/6)
- **The output layer has a JSON source and a documented decision contract.** The 48 signed values in `parameters_output_weights.mem` match per-neuron `output_weights` in `snn_model.json` (neuron-major). Distill row order is `(comfort, temp, power)`. `replay_output_row` is the software replay path: argmax, lowest-index ties, fail-closed on non-finite or malformed rows and on overflowing derived margin/confidence. JSON `output_weights` are snapped to Q8.8 against the neuron-major `.mem` image so value/order drift cannot pass. The model-bank entry for this checkpoint names the RM-1150 output-contract identifier; RM-1150 `ALLOW/WARN/THROTTLE/PAUSE/YIELD_GPU` stays unbound. `tools/verify_q88.py` still pins the `.mem` by canonical sha256 and gold hex. [#6](https://github.com/rmems/Spikenaut-SNN/issues/6), Linear RM-1328.
- **Tier A stream-READY is not axon fill.** After [gaming-telemetry#27](https://github.com/rmems/gaming-telemetry/pull/27), axon **6** (`memory_used_mb`) is **READY** to stream; each of `pcie_tx_kbps` and `pcie_rx_kbps` is independently **READY** to stream (axon **7** is **stream-candidate / projection TBD** and stays unused (0) until a named Stage-1 EXP); axon **8** (`fan_speed_perc`) is **CONDITIONAL READY** (variance-gated). Axons **5** (`gpu_util_pct`) and **9** (`cpu_util_pct`) stay **BLOCKED** (collector schema absent — do not invent them, and do not substitute encoder/decoder util). Live bank remains exp-025 axons 0-4; unused 5-15 stay 0 until a named Stage-1 EXP. Schema / acceptance on [#20](https://github.com/rmems/Spikenaut-SNN/issues/20) stay open.
- **Upstream dataset hygiene.** Sibling telemetry datasets still carry dead columns, schema drift, mixed timestamp formats, synthetic tail records, and stuck values. [#2](https://github.com/rmems/Spikenaut-SNN/issues/2), [#3](https://github.com/rmems/Spikenaut-SNN/issues/3)

## Hardware baseline

Vivado synthesis and implementation reports for the Basys3 target. **These power, LUT, register, and WNS figures are tool estimates, not board-measured.** A Basys live-smoke PASS exists ([silicon-hdl#68](https://github.com/rmems/silicon-hdl/issues/68)); it does not make these rows board-measured.

| Component | Spec |
|-----------|------|
| CPU | AMD Ryzen 9 9950X |
| GPU | NVIDIA RTX 5080 (Blackwell SM_120) |
| FPGA | Digilent Basys3 (Xilinx Artix-7 xc7a35tcpg236-1) |
| FPGA power | 97 mW total (25 mW dynamic, 72 mW static) |
| FPGA LUTs | 1,063 / 20,800 (5.11%) |
| FPGA registers | 1,091 / 41,600 (2.62%) |
| Timing WNS | 3.727 ns (37.27% margin) |
| OS | Fedora 44 |

Per the program's evidence rules, any efficiency claim must rest on measured system or hardware evidence rather than spike-operation counts alone. The power figure above does not yet meet that bar.

**Nor can it be checked from here.** `git ls-tree -r HEAD` returns no RTL, no constraints file, no Vivado project and no synthesis or implementation report — this repository ships the weights and the code that reads them, and nothing else. Every number in the table above is therefore not merely a tool estimate but an *unreproducible* one: a reader cannot regenerate it from this artifact, and neither can its author without the project that produced it.

Two things would have to change for the power figure to mean anything. The **72 mW static** share is the XC7A35T being powered on at 5.11% LUT utilization — it is a property of the part, not of this network. So the only component that says anything about *this network* is the **25 mW dynamic** figure. That is an argument about which number is relevant, not about whether it is trustworthy: 25 mW is one of the unreproducible estimates above, so it is what a measurement should target, not a headline to quote in the meantime. Better still would be energy per inference (dynamic power x latency), which at 1 kHz over 336 parameters should be small — and which, once measured, would be defensible in a way none of these numbers currently are. The second is disclosure: a Vivado power estimate made against default switching activity is a different claim from one made against a SAIF captured from simulating real telemetry, and nothing here records which was used.

## Roadmap

The program advances through a milestone ladder ([#7](https://github.com/rmems/Spikenaut-SNN/issues/7)). Current stage: **M0**.

| Stage | Goal | Exit criterion |
|---|---|---|
| **M0** | Data contracts, no learned control | One session traceable from raw telemetry to a deterministic training record, with hashes and no leakage |
| **M1** | Machine Interoception Benchmark v1 | Reproducible results table showing where temporal models help or fail, against persistence / linear / memoryless baselines |
| **M2** | Teacher → student distillation | Held-out teacher/student agreement plus safety-sensitive disagreement metrics |
| **M3** | Bounded online adaptation | Adaptation moves a predeclared metric without breaking safety-sensitive error bounds |
| **M4** | FPGA parity | Machine-readable spike/action parity report with documented fixed-point and timing differences |
| **M5** | Assisted supervisor under hard Rust shield | Deterministic-only vs assisted controller compared on held-out workloads |
| **M6** | LLM / coding-agent nervous-system experiment | Only after M0–M5 produce usable evidence |

M3 enforces a two-clock rule: a fast loop for telemetry → spikes → inference → proposal, and a slow loop for outcome → eligibility → bounded parameter update. Weights never change on every raw sensor sample.

## Ecosystem

Spikenaut-SNN is a weights and model repository that now also carries a thin Rust package. The table below is the dependency contract ([#5](https://github.com/rmems/Spikenaut-SNN/issues/5)), which deliberately distinguishes libraries this repo depends on — or will — from peer processes it must not. **Declared** marks what `Cargo.toml` actually resolves today. Everything else is either intent — a crate to adopt once it exists — or an explicit non-dependency, and the Relationship column says which.

| Component | Role | Relationship |
|---|---|---|
| [`nir-rs`](https://crates.io/crates/nir-rs) 0.4.3 | NIR graph interchange | **Declared** in `Cargo.toml`, resolved from crates.io — [#8](https://github.com/rmems/Spikenaut-SNN/issues/8) |
| [`kinetic-signals`](https://crates.io/crates/kinetic-signals) 0.4.0 | Causal temporal features (Hurst / Hawkes / surprise / volatility / entropy / EMA-SMA / Z-score / moments) | **Declared** in `Cargo.toml`, resolved from crates.io — host-side preprocessing **upstream of** `axon-encoder`; does not replace it. The kinetic path now encodes against the live 5-col contract (`LiveKineticFrontEnd` → `LiveTelemetryEncoder`, axons 0–4, axons 5–15 at zero), host-side only. FPGA parity is not blocked: software and FPGA should see the same encoded sequence. RAW / KINETIC / HYBRID ablation remains open — [#14](https://github.com/rmems/Spikenaut-SNN/issues/14) |
| [`axon-encoder`](https://crates.io/crates/axon-encoder) 0.4.0 | Telemetry → spike encoding | **Declared** in `Cargo.toml`, resolved from crates.io — downstream of `kinetic-signals` — [#9](https://github.com/rmems/Spikenaut-SNN/issues/9) |
| [`neuromod`](https://crates.io/crates/neuromod) 0.6.0 | LIF engine, seeded stepping, R-STDP, neuromodulators, sparse GIF | **Declared** from crates.io — `HostLif` compatibility plus parallel `HostNetwork` / `HostGifLayer` experiments; their synthetic non-negative weights are not exp-025 and do not rewrite Distill or FPGA artifacts — [#5](https://github.com/rmems/Spikenaut-SNN/issues/5) |
| [`limbic-critic`](https://crates.io/crates/limbic-critic) 0.3.0 | Checked TD critic → neuromodulator adapter | **Declared** from crates.io — `HostCritic` uses `try_assess`, preserves signed TD dopamine, and keeps domain reward collection outside — [#10](https://github.com/rmems/Spikenaut-SNN/issues/10) |
| [`synaptic-wiring`](https://crates.io/crates/synaptic-wiring) 0.3.0 | Deterministic topology, Dale polarity, delayed propagation | **Declared** from crates.io — parallel 16-neuron 12:4 recurrent proposal only; it does not reinterpret the shipped dense input matrix or change `.mem` layout — [#16](https://github.com/rmems/Spikenaut-SNN/issues/16) |
| [`corpus-ipc`](https://crates.io/crates/corpus-ipc) 0.1.0 | Versioned stimulus, spike, and modulator wire messages | **Declared** from crates.io with transport features disabled — validated typed JSON only; no ZMQ/server, and no invented mapping between the two crates' different modulator vocabularies |
| [`silicon-bridge`](https://crates.io/crates/silicon-bridge) 0.3.0 | Checked signed Q8.8 `.mem` export | **Declared** from crates.io with default features disabled — `export_shipped_fpga_image` rejects invalid shapes/ranges, preserves signed hidden and readout words, and explicitly adapts KxN exporter order to NxK silicon-hdl order; all four vault images match byte-for-byte. The UART feature stays disabled, and this does not prove live UART or FPGA parity — [#15](https://github.com/rmems/Spikenaut-SNN/issues/15) |
| [`plasticity-lab`](https://crates.io/crates/plasticity-lab) 0.2.0 | Reproducible reward-modulated training sessions | **Declared** from crates.io as optional feature `training` — `HostTrainingSession` uses the synthetic seeded `HostNetwork`, proves real in-memory weight deltas, and does not export or overwrite exp-025 artifacts — [#17](https://github.com/rmems/Spikenaut-SNN/issues/17) |
| `brainstem-daemon` | 1 kHz headless inference host | **Peer process, not a dependency** — [#11](https://github.com/rmems/Spikenaut-SNN/issues/11) |
| `thalamic-relay` | NVML supervisor, 85 °C / 350 W brake | **Peer process, not a dependency** — [#12](https://github.com/rmems/Spikenaut-SNN/issues/12) |
| `SynapticDistill.jl` | Training sidecar that writes the `.mem` artifacts | **Sidecar, not a Cargo dependency** — Distill pin landed; closed [#13](https://github.com/rmems/Spikenaut-SNN/issues/13) |
| `silicon-hdl` | FPGA RTL | **Consumer of `.mem`, not a dependency** |

Published crates are pinned from crates.io only — no `git` or `path` pins for adopted dependencies.

## The Story

In 2013, a severe concussion left me unable to process the world's data the way I used to. Without access to neuro-rehabilitation, I decided to research on my own, and I started building what would become Spikenaut -- a neuromorphic system that learns from the raw signals of the machines I run every day.  Originally inspired by the bottlenecks of my local GPU (RTX 5080), spent loads of money just to find out that I can't run massive LLM's on it for AI tutoring.  So naturally my curious mind went on the internet to find alternatives.  That is where I found Spiking Neural Networks, a low power alternative to traditional neural networks.

Unfortunately, the neuromorphic field is still in its early stages, and Spikenaut is just the beginning. I created Limen-Neural a GitHub organization over my experimental work in Neuromorphic computing.  Meantime I have been modularizing all my work into reusable components in Limen-Neural. Feel free to check it out use the code to your liking, copy and use it in your own projects or use git dependencies. I'm still far from where I want it to be but I can guarantee you in a near future I will be there with benchmarks, docs with wiki and performance improvements.

As of the right now the weights are a mess, merged_v2 is where I am going to continue improving, the rest are more artifacts than anything.  So expect updates over the time for new and improve SNN weights.

## Related

- **Limen-Neural** — [github.com/Limen-Neural](https://github.com/Limen-Neural) (runtime and learning-rule crates include `neuromod`, `nir-rs`, `axon-encoder`, `synaptic-wiring`, `limbic-critic`, `corpus-ipc`, `plasticity-lab`, and `brainstem-daemon`). The FPGA-export and training peers named elsewhere remain separate repositories.
- **Telemetry** — [rmems/Spikenaut-SNN-Telemetry](https://huggingface.co/datasets/rmems/Spikenaut-SNN-Telemetry)
- **Q8.8 export** — [silicon-bridge](https://github.com/rmems/silicon-bridge)
- **Research program** — [Artificial Interoception / Neuromorphic Supervisor](https://github.com/rmems/Spikenaut-SNN/issues/7)

## License

Dual-licensed under MIT and Apache-2.0. Developed independently by Raul Montoya Cardenas.
