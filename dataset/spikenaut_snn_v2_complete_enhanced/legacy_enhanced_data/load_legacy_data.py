
# Load and analyze YOUR massive legacy Spikenaut dataset
import json
import pandas as pd
import numpy as np
from pathlib import Path

def load_legacy_data(chunk_dir="legacy_enhanced_data"):
    """Load your enhanced legacy dataset"""
    all_data = []
    
    chunk_dir = Path(chunk_dir)
    chunk_files = sorted(chunk_dir.glob("legacy_chunk_*.jsonl"))
    
    print(f"🦁 Loading {len(chunk_files)} legacy data chunks...")
    
    for chunk_file in chunk_files:
        with open(chunk_file, 'r') as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    all_data.append(record)
    
    df = pd.DataFrame(all_data)
    print(f"✅ Loaded {len(df):,} records from legacy dataset")
    
    return df

# Load your legacy data
legacy_df = load_legacy_data()

print("\n📊 Legacy Dataset Overview:")
print(f"  Records: {len(legacy_df):,}")
print(f"  Columns: {list(legacy_df.columns)}")
print(f"  Date range: {legacy_df['timestamp'].min()} to {legacy_df['timestamp'].max()}")

# Analyze trading patterns
print("\n💰 Trading Analysis:")
action_counts = legacy_df['action'].value_counts()
for action, count in action_counts.items():
    print(f"  {action}: {count:,} ({count/len(legacy_df)*100:.1f}%)")

# Portfolio performance over time
if 'portfolio_value' in legacy_df.columns:
    portfolio_stats = legacy_df['portfolio_value'].describe()
    print(f"\n📈 Portfolio Performance:")
    print(f"  Initial: ${portfolio_stats['min']:.2f}")
    print(f"  Final: ${portfolio_stats['max']:.2f}")
    print(f"  Mean: ${portfolio_stats['mean']:.2f}")
    print(f"  Return: {(portfolio_stats['max']/500 - 1)*100:.2f}%")

# Blockchain health analysis
if 'blockchain_health_score' in legacy_df.columns:
    health_stats = legacy_df['blockchain_health_score'].describe()
    print(f"\n⛓️ Blockchain Health:")
    print(f"  Mean score: {health_stats['mean']:.3f}")
    print(f"  Health trend: {'Improving' if health_stats['mean'] > 0.6 else 'Stable' if health_stats['mean'] > 0.4 else 'Declining'}")

print("\n🎉 Your legacy dataset shows rich trading and blockchain telemetry!")
