# Spikenaut SNN v2 - Blockchain Telemetry Dataset

[![Hugging Face Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-blue)](https://huggingface.co/datasets/rmems/Spikenaut-SNN-v2-Telemetry-Data-Weights-Parameters)
[![License: GPL-3.0](https://img.shields.io/badge/License-GPL--3.0-blue.svg)](https://opensource.org/licenses/GPL-3.0)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://python.org)
[![Rust](https://img.shields.io/badge/Rust-1.70%2B-orange)](https://rust-lang.org)
[![Julia](https://img.shields.io/badge/Julia-1.8%2B-purple)](https://julialang.org)

## Dataset Overview

This dataset contains real-time blockchain telemetry data and hybrid Julia-Rust training results for Spikenaut v2, a 16-channel spiking neural network designed for blockchain monitoring and prediction.

### 🚀 Major Updates in v2.0

- ✅ **Hugging Face Compatible**: Proper DatasetDict format with train/validation/test splits
- ✅ **Enhanced Features**: 20+ derived columns including spike encodings and efficiency metrics  
- ✅ **FPGA Parameters**: Complete Q8.8 fixed-point weights for hardware deployment
- ✅ **Time Series Ready**: Temporal splits for forecasting benchmarks
- ✅ **Documentation**: Comprehensive usage examples and API reference

### Key Features

- **Real Blockchain Data**: Fresh telemetry from Kaspa and Monero mainnet nodes
- **Spike-Encoded Features**: Preprocessed neural representations for SNN training  
- **Time Series Ready**: Temporal splits for forecasting benchmarks
- **FPGA Parameters**: Q8.8 fixed-point weights for hardware deployment
- **Hybrid Training**: Julia-Rust integration with sub-50µs processing

## 📊 Dataset Contents

- **`hf_dataset/`**: Main dataset in Hugging Face format (train/validation/test splits)
- **`fresh_sync_data.jsonl`**: Original raw telemetry data
- **`hybrid_training_results.json`**: Julia-Rust training performance metrics
- **`parameters/`**: FPGA-compatible parameter files (Q8.8 format)
- **`dataset_card.json`**: Hugging Face dataset metadata

## 🗂️ Data Schema

### Core Fields (from fresh_sync_data.jsonl)

```json
{
  "timestamp": "2026-03-21 03:18:05.075",     // ISO timestamp
  "blockchain": "kaspa",                      // "kaspa" | "monero"
  "event": "block_acceptance",                // Event type
  "telemetry": {
    "hashrate_mh": 0.92,                      // Mining hashrate (MH/s)
    "power_w": 385.2,                         // Power consumption (Watts)
    "gpu_temp_c": 45.3,                      // GPU temperature (°C)
    "qubic_tick_trace": 1.0,                  // Qubic network trace
    "qubic_epoch_progress": 0.9991,           // Epoch completion
    "reward_hint": 0.9991                     // Reward signal strength
  },
  "blocks_accepted": 8,                       // Blocks in this batch
  "block_rate": 8.0                           // Blocks per second
}
```

### Enhanced Features (v2.0 additions)

```json
{
  "timestamp_unix": 1647836285.075,           // Unix timestamp
  "hour_of_day": 3,                           // Hour (0-23)
  "day_of_week": 0,                           // Day of week (0-6)
  "hashrate_normalized": 0.46,               // Normalized hashrate (0-1)
  "power_efficiency": 2.39,                   // MH/kW efficiency
  "thermal_efficiency": 0.020,                // MH/°C efficiency
  "spike_hashrate": 0,                        // Binary spike: hashrate > 0.9
  "spike_power": 0,                           // Binary spike: power > 390W
  "spike_temp": 0,                            // Binary spike: temp > 43°C
  "spike_qubic": 1,                           // Binary spike: qubic > 0.95
  "composite_reward": 0.819,                 // Composite reward signal
  "target_hashrate_change": 0.03,             // Next tick prediction target
  "target_power_change": 0.9                  // Next tick prediction target
}
```

## 🎯 Usage Examples

### Quick Start with Hugging Face

```python
from datasets import load_dataset

# Load the dataset
ds = load_dataset("rmems/Spikenaut-SNN-v2-Telemetry-Data-Weights-Parameters")

# Access splits
train_data = ds["train"]
val_data = ds["validation"] 
test_data = ds["test"]

print(f"Training samples: {len(train_data)}")
print(f"Features: {list(train_data.features.keys())}")

# View a sample
sample = train_data[0]
print(f"Blockchain: {sample['blockchain']}")
print(f"Hashrate: {sample['telemetry']['hashrate_mh']} MH/s")
print(f"Spike encoding: {sample['spike_hashrate']}")
```

### Time Series Forecasting

```python
import numpy as np
from datasets import load_dataset

ds = load_dataset("rmems/Spikenaut-SNN-v2-Telemetry-Data-Weights-Parameters")

# Prepare features for forecasting
def prepare_features(batch):
    features = [
        batch['hashrate_normalized'],
        batch['power_efficiency'], 
        batch['thermal_efficiency'],
        batch['spike_hashrate'],
        batch['spike_power'],
        batch['spike_qubic']
    ]
    return np.array(features).T

# Create training data
X_train = prepare_features(ds["train"][:])
y_train = ds["train"]["target_hashrate_change"]

print(f"Feature shape: {X_train.shape}")
print(f"Target shape: {y_train.shape}")
```

### Spiking Neural Network Training

```python
import torch
import torch.nn as nn
from datasets import load_dataset

class SpikingNeuron(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.linear = nn.Linear(input_size, hidden_size)
        self.threshold = 0.75
        
    def forward(self, x):
        membrane = self.linear(x)
        spikes = (membrane > self.threshold).float()
        return spikes, membrane

# Load spike-encoded data
ds = load_dataset("rmems/Spikenaut-SNN-v2-Telemetry-Data-Weights-Parameters")

# Prepare spike inputs
spike_cols = ['spike_hashrate', 'spike_power', 'spike_temp', 'spike_qubic']
X_spikes = torch.tensor(ds["train"][:][spike_cols], dtype=torch.float32)

# Train SNN
snn = SpikingNeuron(len(spike_cols), 16)
optimizer = torch.optim.Adam(snn.parameters(), lr=0.001)

for epoch in range(20):
    spikes, membrane = snn(X_spikes)
    loss = torch.mean((spikes - X_spikes)**2)  # Reconstruction loss
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if epoch % 5 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item():.4f}")
```

### FPGA Parameter Loading

```python
import numpy as np

def load_q8_8_parameters(filepath):
    """Load Q8.8 fixed-point parameters from .mem file"""
    with open(filepath, 'r') as f:
        hex_values = [line.strip() for line in f if line.strip()]
    
    # Convert hex to Q8.8 float values
    return np.array([int(hex_val, 16) / 256.0 for hex_val in hex_values], dtype=np.float32)

# Load FPGA parameters
thresholds = load_q8_8_parameters("parameters/parameters.mem")
weights = load_q8_8_parameters("parameters/parameters_weights.mem") 
decay = load_q8_8_parameters("parameters/parameters_decay.mem")

print(f"Loaded {len(thresholds)} thresholds")
print(f"Threshold range: [{thresholds.min():.3f}, {thresholds.max():.3f}]")
```

## 📈 Dataset Statistics

| **Split** | **Samples** | **Percentage** | **Time Range** |
|-----------|-------------|----------------|----------------|
| Train     | 5           | 62.5%          | Mar 21 03:18 - Mar 22 20:16 |
| Validation| 1           | 12.5%          | Mar 22 20:16                    |
| Test      | 2           | 25.0%          | Mar 22 20:16                    |

| **Feature Category** | **Count** | **Description** |
|----------------------|-----------|-----------------|
| Core telemetry       | 8         | Original blockchain data |
| Temporal features    | 3         | Time-based encodings |
| Efficiency metrics   | 3         | Performance ratios |
| Spike encodings      | 4         | Binary neural features |
| Target variables     | 2         | Forecasting targets |

## 🏗️ Data Pipeline Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Raw Telemetry │───▶│   Feature Eng    │───▶│   HF Dataset    │
│                 │    │                  │    │                 │
│ • Kaspa/Monero │    │ • Spike Encode  │    │ • Train/Val/Test│
│ • 8 samples    │    │ • Time Features │    │ • 20+ features  │
│ • JSONL format │    │ • Efficiency    │    │ • Metadata      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## 🔬 Technical Specifications

### Data Collection
- **Sources**: Kaspa mainnet, Monero mainnet
- **Frequency**: Event-driven (block acceptance, sync events)
- **Hardware**: RTX 5080, custom monitoring rig
- **Format**: JSONL → Apache Arrow (HF format)

### Feature Engineering
- **Spike Encoding**: Threshold-based binary features
- **Normalization**: Min-max scaling to [0,1] range
- **Temporal Features**: Hour/day cyclical encoding
- **Efficiency Metrics**: Hardware performance ratios

### Quality Assurance
- **Validation**: 100% valid JSON records
- **Completeness**: No missing values
- **Consistency**: Monitored timestamp ordering
- **Accuracy**: Cross-validated with node logs

## 🚀 Advanced Usage

### Custom Spike Encoding

```python
def custom_spike_encoder(telemetry, thresholds=None):
    """Create custom spike encodings from telemetry data"""
    if thresholds is None:
        thresholds = {
            'hashrate': 0.9,
            'power': 390,
            'temp': 43,
            'qubic': 0.95
        }
    
    spikes = {}
    spikes['hashrate'] = 1 if telemetry['hashrate_mh'] > thresholds['hashrate'] else 0
    spikes['power'] = 1 if telemetry['power_w'] > thresholds['power'] else 0
    spikes['temp'] = 1 if telemetry['gpu_temp_c'] > thresholds['temp'] else 0
    spikes['qubic'] = 1 if telemetry['qubic_tick_trace'] > thresholds['qubic'] else 0
    
    return spikes
```

### Real-time Inference

```python
import time
from datasets import load_dataset

# Load trained model parameters
parameters = load_q8_8_parameters("parameters/parameters_weights.mem")

def real_time_inference(telemetry_data):
    """Run real-time SNN inference on new telemetry"""
    # Encode spikes
    spikes = custom_spike_encoder(telemetry_data)
    
    # Simple matrix multiplication (simulating SNN)
    spike_vector = np.array([spikes['hashrate'], spikes['power'], 
                            spikes['temp'], spikes['qubic']])
    
    # Apply trained weights
    output = np.dot(spike_vector, parameters[:4])  # Simple example
    
    # Decode prediction
    prediction = {
        'next_hashrate_trend': 'up' if output > 0 else 'down',
        'confidence': abs(output),
        'recommendation': 'continue' if output > 0.1 else 'monitor'
    }
    
    return prediction

# Example usage
new_telemetry = {
    'hashrate_mh': 1.2,
    'power_w': 395.0,
    'gpu_temp_c': 44.5,
    'qubic_tick_trace': 0.98
}

result = real_time_inference(new_telemetry)
print(f"Prediction: {result}")
```

## 📚 Related Resources

- **Main Repository**: [Spikenaut SNN v2](https://github.com/rmems/Eagle-Lander)
- **FPGA Implementation**: [Basys3 Deployment Guide](https://github.com/rmems/Eagle-Lander/tree/main/HARDWARE)
- **Training Pipeline**: [Hybrid Julia-Rust Guide](https://github.com/rmems/Eagle-Lander/tree/main/CORE)
- **V1 Dataset**: [Spikenaut v1 Telemetry](https://huggingface.co/datasets/rmems/Spikenaut-v1-Telemetry-Data)

## 📄 License

GPL-3.0 - Same as main Spikenaut project. See LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Data Contributions Welcome!

- Additional blockchain telemetry
- New spike encoding methods
- Performance benchmarking results
- FPGA deployment examples

## 📞 Contact

**Author**: Raul Montoya Cardenas  
**Affiliation**: Texas State University Electrical Engineering  
**Email**: rmems@texasstate.edu  
**Built**: Ship of Theseus workstation, Texas 2026

---

> 🦁 **Spikenaut-SNN-v2** is proof that recovery, engineering, and sovereignty can be achieved independently—one spike at a time.
