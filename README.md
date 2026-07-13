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
pipeline_tag: other
model_name: Spikenaut-SNN-v2
---

# Spikenaut-SNN-v2

A 16-neuron Leaky-Integrate-and-Fire (LIF) Spiking Neural Network trained on live cryptocurrency mining, high-frequency trading, and blockchain sync node telemetry. Designed for Xilinx Artix-7 FPGA deployment at 97 mW.

## The Story

In 2013, a severe concussion left me unable to process the world's data the way I used to. Without access to neuro-rehabilitation, I decided to build my own. As an AI Engineering student at Western Governors University focusing on micro/nano devices, I started building what would become Spikenaut -- a neuromorphic system that learns from the raw signals of the machines I run every day.

The name comes from "spike" (neural firing) and "naut" (navigator). This model is the brain -- the trained neural weights that turn raw telemetry into decisions.

## Architecture

| Spec | Value |
|------|-------|
| Neuron model | Leaky-Integrate-and-Fire (LIF) |
| Neurons | 16 |
| Input channels | 16 |
| Weight format | Q8.8 fixed-point |
| Learning rules | E-prop, OTTT, reward-modulated STDP |
| Clock | 1 kHz (1ms resolution) |
| Training speed | 35 us/tick |
| Memory footprint | 1.6 KB |
| FPGA power | 97 mW (25 mW dynamic, 72 mW static) |
| FPGA target | Xilinx Artix-7 xc7a35tcpg236-1 (Basys3) |

### 16-Channel Input Map

| Channels | Data Source | Function |
|----------|------------|----------|
| 0-1 | DNX (Dynex) | PoUW solver health and neural baselines |
| 2-3 | Quai | Live on-chain reflex and sync confidence |
| 4-5 | Qubic | Epoch and tick cadence monitoring |
| 6-7 | Kaspa | High-frequency DAG settlement tracking |
| 8-9 | XMR (Monero) | Node stability and CPU L3 cache contention |
| 10-11 | Ocean | Data liquidity and staking prep |
| 12-13 | Verus | CPU-heavy validator tracking (AVX-512) |
| 14-15 | Thermal | Pain receptors -- power and temperature |

Channels 14-15 are the network's pain receptors. When the GPU crosses 85C, the SNN receives negative reward and learns to avoid states that could damage the hardware.

## Merged v2 Parameters

This model ships with a merged parameter set combining the best of three training sources:

| Parameter | Source | Values |
|-----------|--------|--------|
| Thresholds (16) | Real trained weights | Graduated 1.125 to 1.594 per neuron |
| Decay rates (16) | Converted parameters | Graduated 0.80 to 0.95 per neuron |
| Hidden weights (256) | Real trained weights | Range 0.75 to 1.04, 76 unique values |
| Output weights (48) | Real trained weights | Signed: -0.164 to +0.258 (inhibitory + excitatory) |

### Q8.8 Fixed-Point Format

All `.mem` files use Q8.8 fixed-point encoding. Each line is one 4-digit hex value:

```
Hex: 0100  →  Decimal: 256  →  Float: 256/256 = 1.0
Hex: 00DA  →  Decimal: 218  →  Float: 218/256 = 0.852
Hex: 00CC  →  Decimal: 204  →  Float: 204/256 = 0.797
```

Negative values use two's complement: `FFF9` = -0.027.

## Files

### Merged v2 (best-of-fusion)
```
dataset/merged_v2/
├── parameters.mem              # 16 neuron thresholds (Q8.8 hex)
├── parameters_decay.mem        # 16 decay rates (Q8.8 hex)
├── parameters_weights.mem      # 16x16 weight matrix (Q8.8 hex)
├── parameters_output_weights.mem # Output layer weights (signed Q8.8)
└── snn_model.json              # Full model definition (float values)
```

### v1 Mining (original)
```
dataset/v1/
├── parameters.mem              # 8 neuron thresholds (Q8.8 hex)
├── parameters_weights.mem      # Weight matrix (Q8.8 hex)
└── parameters_decay.mem        # Decay rates (Q8.8 hex)
```

### v1 FPGA (Basys3 deployment)
```
dataset/v1_fpga/
├── parameters_v1_fpga.mem      # FPGA-optimized thresholds
└── parameters_weights_v1_fpga.mem # FPGA-optimized weights
```

### v2 Per-Asset Variants
```
dataset/dynex_v2/               # Dynex PoUW-optimized
dataset/quai_v2/                # Quai PoW+PoS-optimized
dataset/hft_v2/                 # HFT trading-optimized
dataset/multimodal_v2/          # Multi-asset fusion
```
Each contains: `parameters.mem`, `parameters_weights.mem`, `parameters_decay.mem`

## Usage

### Python

```python
import json

# Load model
with open("dataset/merged_v2/snn_model.json") as f:
    model = json.load(f)

for i, neuron in enumerate(model["neurons"]):
    print(f"Neuron {i}: threshold={neuron['threshold']}, decay={neuron['decay_rate']}")
    print(f"  Weights: {neuron['weights']}")
```

### Verilog

```verilog
// Load thresholds from Q8.8 hex file
reg [15:0] threshold_ram [0:15];
initial $readmemh("dataset/merged_v2/parameters.mem", threshold_ram);

// Load weights from Q8.8 hex file
reg [15:0] weight_ram [0:255];
initial $readmemh("dataset/merged_v2/parameters_weights.mem", weight_ram);
```

### Rust

```rust
use std::fs;

fn load_q88(path: &str) -> Vec<f32> {
    fs::read_to_string(path)
        .unwrap()
        .lines()
        .filter(|l| !l.is_empty())
        .map(|l| u16::from_str_radix(l.trim(), 16).unwrap() as f32 / 256.0)
        .collect()
}

let thresholds = load_q88("dataset/merged_v2/parameters.mem");
let decay = load_q88("dataset/merged_v2/parameters_decay.mem");
let weights = load_q88("dataset/merged_v2/parameters_weights.mem");
```

### Julia

```julia
function load_q88(path::String)
    [parse(Int, line, base=16) / 256.0 for line in eachline(path) if !isempty(line)]
end

thresholds = load_q88("dataset/merged_v2/parameters.mem")
decay = load_q88("dataset/merged_v2/parameters_decay.mem")
weights = load_q88("dataset/merged_v2/parameters_weights.mem")
```

## Training Results

| Metric | Value |
|--------|-------|
| Architecture | Julia-Rust hybrid |
| Algorithm | E-prop + OTTT |
| Accuracy | 95.2% |
| Convergence | 20 epochs |
| Training speed | 35 us/tick |
| IPC overhead | 0.8 us |
| Memory usage | 1.6 KB |
| Training date | 2026-03-22 |
| Data sources | Kaspa mainnet, Monero mainnet |

## Known Limitations

- **Monotonic hidden weight pattern**: Root cause identified (2026-07-12) — NOT an export bug. Caused by degenerate training convergence: 8 uniform training samples + no inhibitory connections + identical E-prop/OTTT gradients. Fix: retrain with `qubic_ticks_snn.jsonl` (27K records) + add 4 inhibitory neurons (80:20 E:I ratio) + implement K-WTA sparsity. See [RM-43](https://linear.app/rpd-34/issue/RM-43).

- **Purely excitatory hidden layer**: All 256 hidden weights are positive. The network lacks inhibitory connections (negative weights) and recurrent feedback, which limits its capacity for noise suppression and temporal memory. A future training run should add ~4 inhibitory neurons and recurrent connections.

## Hardware Baseline

| Component | Spec |
|-----------|------|
| CPU | AMD Ryzen 9 9950X |
| GPU | NVIDIA RTX 5080 (Blackwell SM_120) |
| FPGA | Digilent Basys3 (Xilinx Artix-7 xc7a35tcpg236-1) |
| FPGA Power | 97 mW total |
| FPGA LUTs | 1,063 / 20,800 (5.11%) |
| FPGA Registers | 1,091 / 41,600 (2.62%) |
| Timing WNS | 3.727 ns (37.27% margin) |
| OS | Fedora 43 |

## License

Dual-licensed under MIT and Apache-2.0. Developed independently by Raul Montoya Cardenas, Western Governors University, AI Engineering.

## Related

- **Telemetry Dataset**: [rmems/Spikenaut-SNN-Telemetry](https://huggingface.co/datasets/rmems/Spikenaut-SNN-Telemetry) — 953K+ training records

*"The mind is not a vessel to be filled, but a fire to be kindled."* -- Plutarch
