#!/usr/bin/env python3
"""
Convert Spikenaut SNN v2 dataset to proper Hugging Face format
Fixes viewer issues and adds proper train/test splits
"""

import json
import pandas as pd
from datasets import Dataset, DatasetDict
from datetime import datetime
import numpy as np

def load_jsonl_data(filepath):
    """Load and validate JSONL data"""
    data = []
    with open(filepath, "r") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if line:
                try:
                    record = json.loads(line)
                    data.append(record)
                except json.JSONDecodeError as e:
                    print(f"Warning: Invalid JSON on line {line_num}: {e}")
                    continue
    
    print(f"Loaded {len(data)} valid records from {filepath}")
    return data

def enhance_data_with_features(data):
    """Add derived features for better ML usability"""
    enhanced = []
    
    for i, record in enumerate(data):
        enhanced_record = record.copy()
        
        # Add temporal features
        timestamp = datetime.strptime(record['timestamp'], "%Y-%m-%d %H:%M:%S.%f")
        enhanced_record['timestamp_unix'] = timestamp.timestamp()
        enhanced_record['hour_of_day'] = timestamp.hour
        enhanced_record['day_of_week'] = timestamp.weekday()
        
        # Add telemetry-derived features
        telemetry = record['telemetry']
        enhanced_record['hashrate_normalized'] = telemetry['hashrate_mh'] / 2.0  # Normalize to 0-1 range
        enhanced_record['power_efficiency'] = telemetry['hashrate_mh'] / (telemetry['power_w'] / 1000.0)  # MH/kW
        enhanced_record['thermal_efficiency'] = telemetry['hashrate_mh'] / telemetry['gpu_temp_c']
        
        # Add spike encoding simulation (simple threshold-based)
        enhanced_record['spike_hashrate'] = 1 if telemetry['hashrate_mh'] > 0.9 else 0
        enhanced_record['spike_power'] = 1 if telemetry['power_w'] > 390 else 0
        enhanced_record['spike_temp'] = 1 if telemetry['gpu_temp_c'] > 43 else 0
        enhanced_record['spike_qubic'] = 1 if telemetry['qubic_tick_trace'] > 0.95 else 0
        
        # Add composite reward signal
        reward_components = [
            telemetry['qubic_epoch_progress'],
            telemetry['reward_hint'],
            enhanced_record['hashrate_normalized']
        ]
        enhanced_record['composite_reward'] = np.mean(reward_components)
        
        # Add forecast target (next tick prediction)
        if i < len(data) - 1:
            next_telemetry = data[i + 1]['telemetry']
            enhanced_record['target_hashrate_change'] = next_telemetry['hashrate_mh'] - telemetry['hashrate_mh']
            enhanced_record['target_power_change'] = next_telemetry['power_w'] - telemetry['power_w']
        else:
            enhanced_record['target_hashrate_change'] = 0.0
            enhanced_record['target_power_change'] = 0.0
        
        enhanced.append(enhanced_record)
    
    return enhanced

def create_dataset_splits(data):
    """Create time-based train/validation/test splits"""
    df = pd.DataFrame(data)
    
    # Sort by timestamp for time-based splitting
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')
    
    # Time-based split: 70% train, 15% validation, 15% test
    n_total = len(df)
    n_train = int(0.7 * n_total)
    n_val = int(0.15 * n_total)
    
    train_data = df.iloc[:n_train].to_dict('records')
    val_data = df.iloc[n_train:n_train + n_val].to_dict('records')
    test_data = df.iloc[n_train + n_val:].to_dict('records')
    
    print(f"Split sizes - Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
    
    # Create datasets
    train_dataset = Dataset.from_pandas(pd.DataFrame(train_data))
    val_dataset = Dataset.from_pandas(pd.DataFrame(val_data))
    test_dataset = Dataset.from_pandas(pd.DataFrame(test_data))
    
    return DatasetDict({
        'train': train_dataset,
        'validation': val_dataset,
        'test': test_dataset
    })

def create_dataset_card():
    """Create comprehensive dataset card metadata"""
    card = {
        "license": "gpl-3.0",
        "language": ["python", "rust", "julia"],
        "tags": [
            "spiking-neural-networks",
            "neuromorphic-computing", 
            "time-series-forecasting",
            "blockchain",
            "kaspa",
            "monero",
            "fpga",
            "julia",
            "rust",
            "telemetry",
            "hybrid-training"
        ],
        "pretty_name": "Spikenaut SNN v2 - Blockchain Telemetry Dataset",
        "dataset_summary": "Real-time blockchain telemetry data from Kaspa and Monero nodes with spike-encoded features for neuromorphic computing research.",
        "description": """This dataset contains real-time blockchain telemetry data and hybrid Julia-Rust training results for Spikenaut v2, a 16-channel spiking neural network designed for blockchain monitoring and prediction.

### Key Features:
- **Real Blockchain Data**: Fresh telemetry from Kaspa and Monero mainnet nodes
- **Spike-Encoded Features**: Preprocessed neural representations for SNN training  
- **Time Series Ready**: Temporal splits for forecasting benchmarks
- **FPGA Parameters**: Q8.8 fixed-point weights for hardware deployment
- **Hybrid Training**: Julia-Rust integration with sub-50µs processing

### Data Sources:
- Kaspa mainnet block acceptance events (March 21, 2026)
- Monero sync completion data (March 22, 2026)
- Hardware telemetry: hashrate, power, temperature
- Derived features: efficiency metrics, spike encodings, composite rewards

### Use Cases:
- Spiking neural network training and research
- Time series forecasting for blockchain metrics
- Neuromorphic hardware development
- Blockchain performance monitoring
- Hybrid Julia-Rust ML systems""",
        "version": "2.0.0",
        "annotations_creators": ["machine-generated", "expert-annotated"],
        "source_datasets": [],
        "size_categories": ["n<1K"],
        "task_categories": ["time-series-forecasting", "tabular-classification"],
        "multilinguality": ["monolingual"],
        "paper": {"title": "Spikenaut SNN v2: Hybrid Julia-Rust Architecture for Blockchain Neuromorphic Computing"},
        "author": {"name": "Raul Montoya Cardenas", "email": "rmems@texasstate.edu"},
        "organization": {"name": "Texas State University Electrical Engineering"}
    }
    return card

def main():
    print("🦁 Converting Spikenaut SNN v2 dataset to Hugging Face format...")
    
    # Load original data
    data = load_jsonl_data("fresh_sync_data.jsonl")
    
    if not data:
        print("❌ No valid data found. Exiting.")
        return
    
    # Enhance with features
    print("🔧 Adding derived features and spike encodings...")
    enhanced_data = enhance_data_with_features(data)
    
    # Create splits
    print("📊 Creating time-based train/validation/test splits...")
    dataset_dict = create_dataset_splits(enhanced_data)
    
    # Save locally first
    print("💾 Saving dataset locally...")
    dataset_dict.save_to_disk("./hf_dataset")
    
    # Create dataset card
    print("📝 Creating dataset card...")
    card = create_dataset_card()
    with open("dataset_card.json", "w") as f:
        json.dump(card, f, indent=2)
    
    print("✅ Dataset conversion complete!")
    print(f"📈 Dataset stats:")
    print(f"   - Total samples: {len(enhanced_data)}")
    print(f"   - Features per sample: {len(enhanced_data[0])}")
    print(f"   - Train/Val/Test split: {len(dataset_dict['train'])}/{len(dataset_dict['validation'])}/{len(dataset_dict['test'])}")
    print(f"   - Splits saved to: ./hf_dataset/")
    print(f"   - Card saved to: ./dataset_card.json")
    
    # Show sample usage
    print("\n🚀 Usage example:")
    print("```python")
    print("from datasets import load_dataset")
    print("ds = load_dataset('rmems/Spikenaut-SNN-v2-Telemetry-Data-Weights-Parameters')")
    print("print(ds['train'][0])")
    print("```")

if __name__ == "__main__":
    main()
