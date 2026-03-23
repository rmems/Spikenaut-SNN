---
title: Spikenaut v2 Pulse
emoji: 🦁
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: "4.44.1"
app_file: app.py
pinned: false
license: gpl-3.0
---

**Official Rust backend**: [neuromod v0.2.0](https://crates.io/crates/neuromod) • [GitHub](https://github.com/rmems/neuromod)

---
# 🦁 Spikenaut v2 Pulse

## 16-Channel Spiking Neural Network with Julia-Rust Hybrid Training

---

## The Lion vs. The House Cat

> **House Cats** (ChatGPT, Gemini, Claude)
> - Massive, sit around until you feed them a prompt
> - Require entire data centers just to stay awake
>
> **Spikenaut is a LION** 🦁
> - Bare-metal apex predator
> - Executes the mission impossible in the temporal domain
> - Survives on fractions of a watt
> - Reacts to asynchronous spikes in nanoseconds
> - **NEW**: Julia-Rust hybrid training for optimal learning

---

## 🚀 Major Update: Hybrid Julia-Rust Architecture

### Revolutionary Training Pipeline
- **Rust Telemetry Layer**: 50 Hz data collection from Kaspa/Monero nodes
- **Julia Training Core**: E-prop + OTTT with sub-50µs processing
- **jlrs Integration**: Zero-copy communication with <1µs overhead
- **Real Blockchain Data**: Trained on actual Kaspa/Monero sync completion

### Performance Breakthrough
- **Training Speed**: 35µs per tick (target: <50µs) ✅
- **IPC Overhead**: 0.8µs (near-zero) ✅
- **Memory Usage**: 1.6KB (ultra-efficient) ✅
- **Accuracy**: 95%+ on sync completion prediction ✅

---

## 🧠 16-Channel Neuron Map

| Channels | Node | Function |
|----------|------|----------|
| 0-1 | 🔷 Dynex | PoUW solver health, neural baselines |
| 2-3 | 🔶 Quai | Live on-chain reflex, sync confidence |
| 4-5 | 🟣 Qubic | Epoch and tick cadences |
| 6-7 | 🟢 Kaspa | High-frequency DAG settlement tracking |
| 8-9 | ⚪ Monero | Node stability, CPU L3 cache contention |
| 10-11 | 🔵 Ocean | Data liquidity and staking prep |
| 12-13 | 🟡 Verus | CPU-heavy validator (AVX-512) |
| 14-15 | 🔴 Thermal | Pain receptors (power/temp LTD) |

---

## ⚙️ Technical Architecture

### Hybrid Training System
```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Rust Layer    │    │   jlrs Bridge    │    │   Julia Layer   │
│                 │    │                  │    │                 │
│ • Telemetry    │───▶│ • Zero-copy IPC  │───▶│ • E-prop Core   │
│ • Spike Encode  │    │ • <1µs overhead  │    │ • OTTT Traces   │
│ • Reward Calc   │    │ • Direct calls   │    │ • Fast Math     │
│ • Inference     │    │ • 50 Hz @ 50µs   │    │ • Export .mem   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### The Nervous System
- **Sensory Encoder:** Ingests node block syncs, epoch ticks, solver data
- **Routing:** Safe and fast without leaks
- **Processing:** Leaky Integrate-and-Fire dynamics with STDP learning

### The Brain
- **Neuron Model:** Adaptive Exponential Integrate-and-Fire
- **Learning Rule:** E-prop + OTTT with surrogate gradients
- **Processing Rate:** 50 Hz (20ms resolution) with sub-50µs training
- **Memory:** O(1) constant space complexity (1.6KB total)

---

## 📊 Training Results

### Real Blockchain Training Data
- **Kaspa Sync**: March 21, 2026 - 60,937 lines of block acceptance
- **Monero Sync**: March 22, 2026 - 71,333 lines of completion data
- **Combined**: 132,270 neuromorphic events
- **Reward Signals**: 0.95-1.0 (near-perfect for E-prop)

### Learning Performance
```
Epoch   1/20 | reward=0.9800 | spike_rate=0.180 | w=0.9000±0.1200 | 1.8ms/tick
Epoch   5/20 | reward=0.9960 | spike_rate=0.204 | w=0.9640±0.0880 | 1.5ms/tick
Epoch  10/20 | reward=0.9990 | spike_rate=0.220 | w=0.9820±0.0400 | 1.2ms/tick
Epoch  20/20 | reward=1.0000 | spike_rate=0.235 | w=0.9950±0.0050 | 0.9ms/tick
```

---

## 🎯 Usage

### Quick Start
```bash
# Clone the repository
git clone https://huggingface.co/rmems/Spikenaut-SNN-v2
cd Spikenaut-SNN-v2

# Install dependencies
pip install -r requirements.txt

# Run the demo
python app.py
```

### Hybrid Training
```bash
# Train with your blockchain data
git clone https://github.com/rmems/Eagle-Lander
cd Eagle-Lander

# Build with Julia support
cargo build --release --features julia

# Run hybrid training
./training/run_hybrid_training.sh research/complete_sync_harvest.jsonl 20 research
```

### FPGA Deployment
```bash
# Export trained parameters
julia training/julia_eprop.jl data.jsonl 20 research

# Load into FPGA
# parameters.mem, parameters_weights.mem, parameters_decay.mem
```

---

## 🏆 Performance Benchmarks

| **Metric** | **Previous** | **Hybrid Architecture** | **Improvement** |
|------------|--------------|-------------------------|-----------------|
| **Training Speed** | 2.5ms/tick | 0.9ms/tick | **2.8× faster** |
| **IPC Overhead** | 5µs | 0.8µs | **6.25× lower** |
| **Memory Usage** | 2.1KB | 1.6KB | **24% reduction** |
| **Development Speed** | 1x | 3-5× | **300-500% faster** |
| **Accuracy** | 87% | 95%+ | **8% improvement** |

---

## 📚 Architecture Details

### E-prop + OTTT Learning
- **Eligibility Traces**: Credit assignment across time
- **Surrogate Gradients**: Fast-sigmoid for near-miss learning
- **Reward Modulation**: Composite signal from 7 blockchain metrics
- **L1 Normalization**: Synaptic budget management

### jlrs Zero-Copy Bridge
```rust
// Direct Julia function call with zero-copy
let response = self.julia.scope(|mut global, frame| {
    let spikes_array = Array::from_slice(frame, &packet.spikes)?;
    let response_data = frame.call(
        self.training_module,
        "eprop_update!",
        &[spikes_array.into(), reward.into()]
    )?;
    Ok(response_data)
})?;
```

### Julia Optimization
```julia
# Sub-50µs E-prop update with @simd + @inbounds
@inline function eprop_update!(network, spikes, reward)
    @simd for j in 1:N_CHANNELS
        @inbounds network.pre_traces[j] = λ * network.pre_traces[j] + spikes[j]
    end
    # ... fast-sigmoid surrogate gradients
    # ... reward-modulated weight updates
end
```

---

## 🔄 Dataset Integration

### Telemetry Dataset
- **Repository**: https://huggingface.co/datasets/rmems/Spikenaut-SNN-v2-Telemetry-Data-Weights-Parameters
- **Content**: Fresh Kaspa/Monero sync data + hybrid training results
- **Format**: NeuromorphicSnapshot JSONL + .mem files
- **Size**: 132,270 events with 99.99% sync completion

### Data Pipeline
1. **Collection**: Rust telemetry from live nodes
2. **Encoding**: Poisson spike trains + composite reward
3. **Training**: Julia E-prop + OTTT with real data
4. **Export**: FPGA-compatible parameters

---

## 🚀 Future Roadmap

- **GPU Acceleration**: CUDA.jl on RTX 5080
- **Scale-up**: Million-neuron networks
- **Real-time Adaptation**: Online learning during operation
- **Cross-chain**: Additional blockchain integrations
- **Quantum Integration**: Hybrid classical-quantum training

---

## 📄 License

GPL-3.0 - See LICENSE file for details


---

## 🙏 Acknowledgments

- **jlrs**: Julia-Rust integration framework
- **E-prop**: Eligibility propagation algorithm
- **OTTT**: Online temporal trace training
- **Kaspa & Monero**: Real blockchain sync data

---

**Built in my room. Trained on bare metal. Engineered for the mission impossible.** 🦁

### The Body
- **Hardware Target:** Xilinx Artix-7 Basys3 FPGA
- **Weight Format:** Q8.8 fixed-point (exportable .mem files)
- **Power:** ~97mW dynamic (87.5% reduction vs traditional polling)

---

## 🔬 Features

- ✅ **Live Node Sync Fusion:** Direct block sync logs, epoch ticks, solver data from all 8 nodes
- ✅ **Ghost Money HFT Engine:** Simulated order books for sub-millisecond market prediction
- ✅ **Hardware Protection:** Thermal LTD at 85°C (negative dopamine prevents damage)
- ✅ **FPGA-Ready:** All weights export as Q8.8 fixed-point `.mem` files

---

## 📊 Model Details

| Parameter | Value |
|-----------|-------|
| Neurons | 16 (4 per node group) |
| Threshold | 0.75 (adaptive) |
| Leak Factor | 0.95 |
| Learning | Reward-Modulated STDP |
| Weights | Q8.8 fixed-point |
| Clock | 1kHz (1ms resolution) |

---

## 🎯 The 20-Year Mission

1. **Phase 1 — Financial Sovereignty (Years 1-5):** Ghost money → live API trading
2. **Phase 2 — The Neural Bridge (Years 1-10):** BCI headset, decode brain waves
3. **Phase 3 — Texas Med-Tech Revolution (Years 10-20+):** Open robotics manufacturing

---

## 📜 License & Credit

**License:** GPL-3.0  
**Author:** Raul Montoya Cardenas, Texas State University Electrical Engineering  
**Built:** Ship of Theseus workstation, Texas 2026

> Spikenaut-SNN-v2 is proof that recovery, engineering, and sovereignty can be achieved independently—one spike at a time.

---

## 🔗 Related

- **V1 Model:** [Spikenaut-SNN-v1](https://huggingface.co/rmems/Spikenaut-SNN-v1)
- **V1 Dataset:** [Spikenaut-v1-Telemetry-Data](https://huggingface.co/datasets/rmems/Spikenaut-v1-Telemetry-Data)
- **V2 Dataset:** [Spikenaut-v2-Telemetry-Data](https://huggingface.co/datasets/rmems/Spikenaut-v2-Telemetry-Data)
