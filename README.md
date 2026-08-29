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

A 16-neuron Leaky-Integrate-and-Fire (LIF) spiking neural network that learns a compact temporal representation of machine state from live hardware and node telemetry, targeting a Xilinx Artix-7 FPGA.

Spikenaut is the small-supervisor layer of a wider research program, **Artificial Interoception / Neuromorphic Supervisor** ([#7](https://github.com/rmems/Spikenaut-SNN/issues/7)). The name comes from "spike" (neural firing) and "naut" (navigator). This repository holds the brain — the trained weights and the export contract that turns raw telemetry into decisions.

## Status — read this first

This is a **research artifact, not a validated supervisor.** Three things a reader should know before using the shipped weights:

| | |
|---|---|
| **The hidden weight matrix is degenerate.** | All 256 hidden weights follow a linear ramp, `weight[n][i] = 0.75 + n×0.015625 + i×0.00390625`. This is real trained output, but from a degenerate convergence — not a healthy learned matrix. See [#2](https://github.com/rmems/Spikenaut-SNN/issues/2). |
| **It was trained on 8 records.** | The shipped `merged_v2` weights come from `fresh_sync_data.jsonl` — 8 records across 2 mining sessions, with 2 of 6 features effectively constant. The dataset has since been replaced with `qubic_ticks_snn.jsonl` (27,430 records), but **the model has not yet been retrained.** See [#13](https://github.com/rmems/Spikenaut-SNN/issues/13). |
| **FPGA parity is unproven.** | The hardware numbers below are Vivado synthesis and implementation reports, not board-measured results. Whether the deployed SNN behaves like the software model is an open question, not a settled one. See [#6](https://github.com/rmems/Spikenaut-SNN/issues/6). |

The architecture and the Q8.8 export contract are verifiable from the artifacts in this repository. The FPGA resource and power figures are **externally reported** — no RTL, constraints, Vivado project or synthesis reports are checked in here, so a reader cannot reproduce them from this artifact. The *weights* are the part that needs redoing.

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
| Input channels | 16 |
| Weight format | Q8.8 fixed-point |
| Learning rules | E-prop, OTTT, reward-modulated STDP |
| Clock | 1 kHz (1 ms resolution) |
| Training speed | 35 µs/tick |
| Memory footprint | 1.6 KB |
| FPGA target | Xilinx Artix-7 xc7a35tcpg236-1 (Basys3) |

## Inputs — current v2 layout

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

Channels 14-15 are the network's pain receptors. When the GPU crosses 85 °C, the SNN receives negative reward and learns to avoid states that could damage the hardware.

**This map is the current v2 layout, not a fixed contract.** The replacement state contract is being defined in [#20](https://github.com/rmems/Spikenaut-SNN/issues/20), under one governing rule: logical state variables are *not* equivalent to physical SNN axons. Signals are never invented or duplicated just to fill 16 slots. Instead an explicit adapter sits between them:

```text
raw state → state adapter → encoder → fixed-width SNN stimuli
```

The adapter accepts a variable number of legitimate source signals, so the raw feature count can change without the input contract breaking. Every signal must declare its unit, source of truth, timestamp and sampling semantics, valid range, normalization, missing-value and staleness behavior, and provenance. Missing signals are masked, never silently zeroed.

## Merged v2 parameters

| Parameter | Source | Values |
|-----------|--------|--------|
| Thresholds (16) | Trained | Graduated 1.125 to 1.594 per neuron |
| Decay rates (16) | Converted | Graduated 0.80 to 0.95 per neuron |
| Hidden weights (256) | Trained — **degenerate, see below** | Range 0.75 to 1.04, 76 unique values |
| Output weights (48) | Trained | Signed: -0.164 to +0.258 (inhibitory + excitatory) |

The hidden weight matrix is the known problem. Each neuron's 16 weights increase by exactly one Q8.8 step (`0x0001`, 0.0039):

```text
Neuron 0:  00C0, 00C1, 00C2, 00C3, ... 00CF   (+1 each)
Neuron 1:  00C4, 00C5, 00C6, 00C7, ... 00D3   (+1 each)
```

A healthy trained SNN shows weights moving up *and* down across a neuron's inputs. This ramp is caused by degenerate training convergence — too few samples, no inhibitory connections, and identical E-prop/OTTT gradients across neurons — not by an export bug. The export path was independently cross-validated and confirmed correct ([#4](https://github.com/rmems/Spikenaut-SNN/issues/4)).

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
dataset/merged_v2/
├── parameters.mem                 # 16 neuron thresholds (Q8.8 hex)
├── parameters_decay.mem           # 16 decay rates (Q8.8 hex)
├── parameters_weights.mem         # 16x16 weight matrix (Q8.8 hex)
├── parameters_output_weights.mem  # Output layer weights (signed Q8.8)
└── snn_model.json                 # Full model definition (float values)
```

### Loading on FPGA

```verilog
// Load thresholds from Q8.8 hex file
reg [15:0] threshold_ram [0:15];
initial $readmemh("dataset/merged_v2/parameters.mem", threshold_ram);

// Load weights from Q8.8 hex file
reg [15:0] weight_ram [0:255];
initial $readmemh("dataset/merged_v2/parameters_weights.mem", weight_ram);
```

## Training provenance

| Metric | Value |
|--------|-------|
| Architecture | Julia-Rust hybrid |
| Algorithm | E-prop + OTTT |
| Convergence | 20 epochs |
| Training speed | 35 µs/tick |
| IPC overhead | 0.8 µs |
| Memory usage | 1.6 KB |
| Training date | 2026-03-22 |
| Training data | `fresh_sync_data.jsonl` — **8 records**, Kaspa + Monero mainnet sessions |

The 8-record training set is the root cause of the degenerate weights. Two of its six features (`qubic_epoch_progress`, `reward_hint`) are effectively constant — range 0.0009, standard deviation 0.000284 — while dominating spike encoding at 87.5% spike rate each. Monotonically converging sync data (0.999912 → 1.0) produces single-attractor weights.

The training data has been replaced with `qubic_ticks_snn.jsonl` (27,430 records) and a data adapter added, but the retrain has not happened yet. The trainer itself also needs fixing first ([#13](https://github.com/rmems/Spikenaut-SNN/issues/13)): it currently pins `W_MIN = 0`, row L1-normalizes, writes unsigned Q8.8, and emits no output-weight file.

## Known limitations

- **Degenerate hidden weights.** Linear-ramp matrix from an 8-record training set. Fix path: retrain on the 27K-record dataset, add ~4 inhibitory neurons (80:20 E:I), implement K-WTA sparsity. [#2](https://github.com/rmems/Spikenaut-SNN/issues/2), [#3](https://github.com/rmems/Spikenaut-SNN/issues/3), [#13](https://github.com/rmems/Spikenaut-SNN/issues/13)
- **Purely excitatory hidden layer.** All 256 hidden weights are positive. Without inhibition and recurrence the network cannot do winner-take-all competition, suppress noise, or sharpen contrast. Note it is not memoryless: each LIF neuron's decaying membrane potential retains recent input, so the gap is **long-horizon and recurrent** memory, not temporal state as such. [#3](https://github.com/rmems/Spikenaut-SNN/issues/3)
- **No FPGA parity evidence.** Spike agreement, action agreement, membrane-potential error, and quantization error against the software model have not been measured. Hardware numbers below are synthesis reports. [#6](https://github.com/rmems/Spikenaut-SNN/issues/6)
- **Export tooling clamps negatives.** `silicon-bridge`'s `encode_q88` currently clamps negative values to zero, which would destroy the signed `parameters_output_weights.mem`. Signed Q8.8 is a hard requirement before that path is adopted. [#15](https://github.com/rmems/Spikenaut-SNN/issues/15)
- **Output weights have no float source.** `snn_model.json` records only the 16 hidden-layer neurons; the 48 signed values in `parameters_output_weights.mem` have no float counterpart in this repository, so they cannot be cross-validated against a source of truth. They are checked structurally instead (count, encoding, round-trip, sign integrity). A sign-preserving exporter regression would go undetected until the output layer is added to `snn_model.json`. [#4](https://github.com/rmems/Spikenaut-SNN/issues/4)
- **Upstream dataset hygiene.** Sibling telemetry datasets still carry dead columns, schema drift, mixed timestamp formats, synthetic tail records, and stuck values. [#2](https://github.com/rmems/Spikenaut-SNN/issues/2), [#3](https://github.com/rmems/Spikenaut-SNN/issues/3)

## Hardware baseline

Vivado synthesis and implementation reports for the Basys3 target. **These are tool estimates, not board-measured figures.**

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

Spikenaut-SNN is currently a **weights and model repository** — there is no `Cargo.toml` yet. The table below is the intended dependency contract ([#5](https://github.com/rmems/Spikenaut-SNN/issues/5)), which deliberately distinguishes libraries this repo will depend on from peer processes it must not.

| Component | Role | Relationship |
|---|---|---|
| [`nir-rs`](https://crates.io/crates/nir-rs) 0.4.2 | NIR graph interchange | crates.io dependency — [#8](https://github.com/rmems/Spikenaut-SNN/issues/8) |
| [`axon-encoder`](https://crates.io/crates/axon-encoder) 0.4.0 | Telemetry → spike encoding | crates.io dependency — [#9](https://github.com/rmems/Spikenaut-SNN/issues/9) |
| [`neuromod`](https://crates.io/crates/neuromod) 0.5.2 | LIF engine, learning rules, neuromodulators | crates.io dependency |
| `silicon-bridge` | Q8.8 `.mem` export | Dependency once published — [#15](https://github.com/rmems/Spikenaut-SNN/issues/15) |
| `kinetic-signals` | Feature math for channels 0–13 | Dependency once published — [#14](https://github.com/rmems/Spikenaut-SNN/issues/14) |
| `synaptic-mesh` | Dale 80:20 polarity, 16-channel router | Dependency once published — [#16](https://github.com/rmems/Spikenaut-SNN/issues/16) |
| `limbic-critic` | TD critic → neuromodulator adapter | Optional dependency once published — [#10](https://github.com/rmems/Spikenaut-SNN/issues/10) |
| `plasticity-lab` | Reproducible training loops | Only once it actually writes weight deltas — [#17](https://github.com/rmems/Spikenaut-SNN/issues/17) |
| `brainstem-daemon` | 1 kHz headless inference host | **Peer process, not a dependency** — [#11](https://github.com/rmems/Spikenaut-SNN/issues/11) |
| `thalamic-relay` | NVML supervisor, 85 °C / 350 W brake | **Peer process, not a dependency** — [#12](https://github.com/rmems/Spikenaut-SNN/issues/12) |
| `SynapticDistill.jl` | Training sidecar that writes the `.mem` artifacts | **Sidecar, not a Cargo dependency** — [#13](https://github.com/rmems/Spikenaut-SNN/issues/13) |
| `silicon-hdl` | FPGA RTL | **Consumer of `.mem`, not a dependency** |

Published crates are pinned from crates.io only — no `git` or `path` pins for adopted dependencies.

## The Story

In 2013, a severe concussion left me unable to process the world's data the way I used to. Without access to neuro-rehabilitation, I decided to research on my own, and I started building what would become Spikenaut -- a neuromorphic system that learns from the raw signals of the machines I run every day.  Originally inspired by the bottlenecks of my local GPU (RTX 5080), spent loads of money just to find out that I can't run massive LLM's on it for AI tutoring.  So naturally my curious mind went on the internet to find alternatives.  That is where I found Spiking Neural Networks, a low power alternative to traditional neural networks.

Unfortunately, the neuromorphic field is still in its early stages, and Spikenaut is just the beginning. I created Limen-Neural a GitHub organization over my experimental work in Neuromorphic computing.  Meantime I have been modularizing all my work into reusable components in Limen-Neural. Feel free to check it out use the code to your liking, copy and use it in your own projects or use git dependencies. I'm still far from where I want it to be but I can guarantee you in a near future I will be there with benchmarks, docs with wiki and performance improvements.

As of the right now the weights are a mess, merged_v2 is where I am going to continue improving, the rest are more artifacts than anything.  So expect updates over the time for new and improve SNN weights.

## Related

- **Limen-Neural** — [github.com/Limen-Neural](https://github.com/Limen-Neural) (runtime, FPGA export, training crates)
- **Telemetry** — [rmems/Spikenaut-SNN-Telemetry](https://huggingface.co/datasets/rmems/Spikenaut-SNN-Telemetry)
- **Q8.8 export** — [silicon-bridge](https://github.com/Limen-Neural/silicon-bridge)
- **Research program** — [Artificial Interoception / Neuromorphic Supervisor](https://github.com/rmems/Spikenaut-SNN/issues/7)

## License

Dual-licensed under MIT and Apache-2.0. Developed independently by Raul Montoya Cardenas.
