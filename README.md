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

A 16-neuron Leaky-Integrate-and-Fire (LIF) Spiking Neural Network trained on live cryptocurrency telemetry mining, high-frequency trading, and blockchain sync node telemetry. Designed for Xilinx Artix-7 FPGA deployment at 97 mW.


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

This is bound to change in the future, not sure how I am going to change the input format.

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

### Merged v2
```
dataset/merged_v2/
├── parameters.mem              # 16 neuron thresholds (Q8.8 hex)
├── parameters_decay.mem        # 16 decay rates (Q8.8 hex)
├── parameters_weights.mem      # 16x16 weight matrix (Q8.8 hex)
├── parameters_output_weights.mem # Output layer weights (signed Q8.8)
└── snn_model.json              # Full model definition (float values)
```

Each contains: `parameters.mem`, `parameters_weights.mem`, `parameters_decay.mem`

### Verilog

```verilog
// Load thresholds from Q8.8 hex file
reg [15:0] threshold_ram [0:15];
initial $readmemh("dataset/merged_v2/parameters.mem", threshold_ram);

// Load weights from Q8.8 hex file
reg [15:0] weight_ram [0:255];
initial $readmemh("dataset/merged_v2/parameters_weights.mem", weight_ram);
```

## Training Results

| Metric | Value |
|--------|-------|
| Architecture | Julia-Rust hybrid |
| Algorithm | E-prop + OTTT |
| Convergence | 20 epochs |
| Training speed | 35 us/tick |
| IPC overhead | 0.8 us |
| Memory usage | 1.6 KB |
| Training date | 2026-03-22 |
| Data sources | Kaspa mainnet, Monero mainnet |

## Known Limitations

- **Monotonic hidden weight pattern**: Root cause identified (2026-07-12) — NOT an export bug. Caused by degenerate training convergence: 8 uniform training samples + no inhibitory connections + identical E-prop/OTTT gradients. Fix: retrain with `qubic_ticks_snn.jsonl` (27K records) + add 4 inhibitory neurons (80:20 E:I ratio) + implement K-WTA sparsity.

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
| OS | Fedora 44 |

## The Story

In 2013, a severe concussion left me unable to process the world's data the way I used to. Without access to neuro-rehabilitation, I decided to research on my own, and I started building what would become Spikenaut -- a neuromorphic system that learns from the raw signals of the machines I run every day.  Originally inspired by the bottlenecks of my local GPU (RTX 5080), spent loads of money just to find out that I can't run massive LLM's on it for AI tutoring.  So naturally my curious mind went on the internet to find alternatives.  That is where I found Spiking Neural Networks, a low power alternative to traditional neural networks. 

Unfortunately, the neuromorphic field is still in its early stages, and Spikenaut is just the beginning. I created Limen-Neural a GitHub organization over my experimental work in Neuromorphic computing.  Meantime I have been modularizing all my work into reusable components in Limen-Neural. Feel free to check it out use the code to your liking, copy and use it in your own projects or use git dependencies. I'm still far from where I want it to be but I can guarantee you in a near future I will be there with benchmarks, docs with wiki and performance improvements.

As of the right now the weights are a mess, merged_v2 is where I am going to continue improving, the rest are more artifacts than anything.  So expect updates over the time for new and improve SNN weights.

The name comes from "spike" (neural firing) and "naut" (navigator). This model is the brain -- the trained neural weights that turn raw telemetry into decisions.

## Related
- **Limen-Neural** — [github.com/Limen-Neural](https://github.com/Limen-Neural) (runtime, FPGA export, training crates)
- **Telemetry** — [rmems/Spikenaut-SNN-Telemetry](https://huggingface.co/datasets/rmems/Spikenaut-SNN-Telemetry)
- **Q8.8 export** — [silicon-bridge](https://github.com/Limen-Neural/silicon-bridge)


## License

Dual-licensed under MIT and Apache-2.0. Developed independently by Raul Montoya Cardenas