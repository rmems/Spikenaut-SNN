# Spikenaut SNN v2 - Fresh Telemetry Data & Hybrid Training Results

## Dataset Overview

This dataset contains fresh blockchain telemetry data and hybrid Julia-Rust training results for Spikenaut v2.

### Contents

- `fresh_sync_data.jsonl`: Real-time blockchain sync data from Kaspa and Monero
- `hybrid_training_results.json`: Julia-Rust hybrid training performance metrics
- `parameters/`: FPGA-compatible parameter files (Q8.8 format)

### Data Sources

#### Kaspa Mainnet (March 21, 2026)
- **Event**: Real-time block acceptance
- **Pattern**: "Accepted X blocks ... via relay"
- **Performance**: 8-13 blocks/second
- **Status**: Fully synced and operational

#### Monero Mainnet (March 22, 2026)
- **Event**: Sync completion from 99.99% to 100%
- **Pattern**: "Synced 3635984/3635984"
- **Performance**: 9.268 blocks/second
- **Status**: Fully synced

### Hybrid Training Architecture

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

### Performance Metrics

| **Metric** | **Value** | **Status** |
|------------|-----------|------------|
| Training Speed | 35µs/tick | ✅ Target met |
| IPC Overhead | 0.8µs | ✅ Near-zero |
| Memory Usage | 1.6KB | ✅ Ultra-efficient |
| Accuracy | 95.2% | ✅ High accuracy |
| Data Quality | 99.99% sync | ✅ Premium data |

### Usage

```python
# Load fresh sync data
import json

with open("fresh_sync_data.jsonl", "r") as f:
    for line in f:
        sample = json.loads(line)
        print(f"Blockchain: {sample['blockchain']}")
        print(f"Reward: {sample['telemetry']['reward_hint']}")

# Load training results
with open("hybrid_training_results.json", "r") as f:
    results = json.load(f)
    print(f"Architecture: {results['architecture']}")
    print(f"Performance: {results['performance_metrics']}")
```

### License

GPL-3.0 - Same as main Spikenaut project
