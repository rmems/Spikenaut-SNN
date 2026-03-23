#!/usr/bin/env python3
"""
Integrate additional Spikenaut data sources:
- Training data (snn_training_*.jsonl)
- Supervisor telemetry (supervisor_telemetry.jsonl)
- Mining logs (miner.log)
- Mind telemetry (mind_telemetry.jsonl)
- Neuromorphic data (neuromorphic_data.jsonl)
"""

import json
import pandas as pd
from pathlib import Path
from datetime import datetime
import gzip

def analyze_training_data():
    """Analyze SNN training datasets"""
    
    print("🧠 Analyzing SNN Training Data")
    print("=" * 40)
    
    training_files = {
        'all_training': '/home/user/Eagle-Lander/DATA/research/snn_training_all.jsonl',
        'market_training': '/home/user/Eagle-Lander/DATA/research/snn_training_market.jsonl',
        'mind_training': '/home/user/Eagle-Lander/DATA/research/snn_training_mind.jsonl'
    }
    
    training_stats = {}
    
    for name, filepath in training_files.items():
        if not Path(filepath).exists():
            continue
            
        print(f"\n📊 {name.replace('_', ' ').title()}:")
        
        records = []
        with open(filepath, 'r') as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
        
        print(f"  Records: {len(records):,}")
        
        if records:
            # Analyze structure
            first_record = records[0]
            print(f"  Fields: {list(first_record.keys())}")
            
            # Time range
            if 'timestamp' in first_record:
                timestamps = [datetime.fromisoformat(r['timestamp'].replace('Z', '+00:00')) for r in records if 'timestamp' in r]
                if timestamps:
                    print(f"  Time range: {min(timestamps)} to {max(timestamps)}")
                    print(f"  Duration: {max(timestamps) - min(timestamps)}")
            
            # Spike analysis
            if 'expected_spikes' in first_record:
                spike_arrays = [r['expected_spikes'] for r in records if 'expected_spikes' in r]
                if spike_arrays:
                    avg_spikes = sum(sum(spikes) for spikes in spike_arrays) / len(spike_arrays)
                    print(f"  Average spikes per record: {avg_spikes:.2f}")
                    print(f"  Neuron count: {len(spike_arrays[0])}")
            
            # Reward signals
            if 'metadata' in first_record and 'reward_signal' in first_record['metadata']:
                rewards = [r['metadata']['reward_signal'] for r in records if 'metadata' in r and 'reward_signal' in r['metadata']]
                if rewards:
                    print(f"  Reward range: [{min(rewards):.3f}, {max(rewards):.3f}]")
                    print(f"  Average reward: {sum(rewards)/len(rewards):.3f}")
        
        training_stats[name] = {
            'records': len(records),
            'filepath': filepath,
            'size_mb': Path(filepath).stat().st_size / (1024*1024)
        }
    
    return training_stats

def analyze_supervisor_data():
    """Analyze supervisor telemetry"""
    
    print("\n👨‍💼 Analyzing Supervisor Telemetry")
    print("=" * 40)
    
    supervisor_file = '/home/user/Eagle-Lander/DATA/research/supervisor_telemetry.jsonl'
    
    if not Path(supervisor_file).exists():
        print("❌ Supervisor telemetry file not found")
        return {}
    
    records = []
    with open(supervisor_file, 'r') as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    
    print(f"📊 Supervisor Records: {len(records)}")
    
    if records:
        # Process events
        events = {}
        for record in records:
            status = record.get('status', 'unknown')
            if status not in events:
                events[status] = 0
            events[status] += 1
        
        print(f"📈 Event Types:")
        for status, count in events.items():
            print(f"  {status}: {count}")
        
        # Time analysis
        timestamps = [datetime.fromisoformat(r['timestamp'].replace('Z', '+00:00')) for r in records if 'timestamp' in r]
        if timestamps:
            print(f"⏰ Time range: {min(timestamps)} to {max(timestamps)}")
            print(f"📅 Duration: {max(timestamps) - min(timestamps)}")
    
    return {
        'records': len(records),
        'events': events if records else {},
        'filepath': supervisor_file,
        'size_mb': Path(supervisor_file).stat().st_size / (1024*1024)
    }

def analyze_mining_data():
    """Analyze mining operation logs"""
    
    print("\n⛏️ Analyzing Mining Data")
    print("=" * 40)
    
    miner_log = '/home/user/Eagle-Lander/DATA/research/miner.log'
    
    if not Path(miner_log).exists():
        print("❌ Mining log file not found")
        return {}
    
    # Get file info
    file_size_mb = Path(miner_log).stat().st_size / (1024*1024)
    print(f"📁 Mining Log: {file_size_mb:.1f} MB")
    
    # Sample analysis (first 1000 lines)
    sample_lines = []
    with open(miner_log, 'r') as f:
        for i, line in enumerate(f):
            if i < 1000:
                sample_lines.append(line.strip())
            else:
                break
    
    print(f"📊 Sample Analysis (first 1000 lines):")
    
    # Look for key patterns
    hashrate_lines = [line for line in sample_lines if 'MH/s' in line or 'hashrate' in line.lower()]
    temp_lines = [line for line in sample_lines if 'temp' in line.lower() or '°C' in line]
    error_lines = [line for line in sample_lines if 'error' in line.lower() or 'failed' in line.lower()]
    
    print(f"  Hashrate mentions: {len(hashrate_lines)}")
    print(f"  Temperature mentions: {len(temp_lines)}")
    print(f"  Error mentions: {len(error_lines)}")
    
    # Show sample hashrate data
    if hashrate_lines:
        print(f"\n💰 Sample Hashrate Data:")
        for line in hashrate_lines[:3]:
            print(f"  {line}")
    
    return {
        'file_size_mb': file_size_mb,
        'sample_lines': len(sample_lines),
        'hashrate_mentions': len(hashrate_lines),
        'temp_mentions': len(temp_lines),
        'error_mentions': len(error_lines),
        'filepath': miner_log
    }

def analyze_neuromorphic_data():
    """Analyze massive neuromorphic dataset"""
    
    print("\n🧬 Analyzing Neuromorphic Data")
    print("=" * 40)
    
    neuro_file = '/home/user/Eagle-Lander/DATA/research/neuromorphic_data.jsonl'
    
    if not Path(neuro_file).exists():
        print("❌ Neuromorphic data file not found")
        return {}
    
    # Get file info
    file_size_mb = Path(neuro_file).stat().st_size / (1024*1024)
    print(f"📁 Neuromorphic Dataset: {file_size_mb:.1f} MB")
    
    # Sample analysis (first 1000 records)
    records = []
    with open(neuro_file, 'r') as f:
        for i, line in enumerate(f):
            if i < 1000 and line.strip():
                try:
                    records.append(json.loads(line))
                except:
                    continue
    
    print(f"📊 Sample Analysis (first 1000 valid records):")
    print(f"  Valid records: {len(records)}")
    
    if records:
        first_record = records[0]
        print(f"  Sample fields: {list(first_record.keys())[:10]}...")  # Show first 10 fields
        
        # Check for spike data
        spike_fields = [k for k in first_record.keys() if 'spike' in k.lower()]
        if spike_fields:
            print(f"  Spike-related fields: {spike_fields}")
        
        # Check for timestamps
        if 'timestamp' in first_record:
            timestamps = [datetime.fromisoformat(r['timestamp'].replace('Z', '+00:00')) for r in records if 'timestamp' in r]
            if timestamps:
                print(f"  Time range: {min(timestamps)} to {max(timestamps)}")
    
    return {
        'file_size_mb': file_size_mb,
        'sample_records': len(records),
        'filepath': neuro_file
    }

def create_additional_data_summary():
    """Create comprehensive summary of all additional data"""
    
    print("\n🦁 Spikenaut Additional Data Analysis")
    print("=" * 60)
    
    # Analyze all data sources
    training_stats = analyze_training_data()
    supervisor_stats = analyze_supervisor_data()
    mining_stats = analyze_mining_data()
    neuro_stats = analyze_neuromorphic_data()
    
    # Create summary
    summary = {
        'additional_data_sources': {
            'training_data': training_stats,
            'supervisor_telemetry': supervisor_stats,
            'mining_logs': mining_stats,
            'neuromorphic_data': neuro_stats
        },
        'total_additional_size_mb': sum([
            sum(s.get('size_mb', 0) for s in training_stats.values()),
            supervisor_stats.get('size_mb', 0),
            mining_stats.get('file_size_mb', 0),
            neuro_stats.get('file_size_mb', 0)
        ]),
        'analysis_date': datetime.now().isoformat()
    }
    
    print(f"\n📊 Additional Data Summary:")
    print(f"  Total additional data: {summary['total_additional_size_mb']:.1f} MB")
    print(f"  Training datasets: {len(training_stats)} files")
    print(f"  Supervisor events: {supervisor_stats.get('records', 0)}")
    print(f"  Mining log size: {mining_stats.get('file_size_mb', 0):.1f} MB")
    print(f"  Neuromorphic dataset: {neuro_stats.get('file_size_mb', 0):.1f} MB")
    
    return summary

def create_integration_recommendations(summary):
    """Create recommendations for integrating additional data"""
    
    print("\n🚀 Integration Recommendations")
    print("=" * 40)
    
    recommendations = []
    
    # Training data
    training_stats = summary['additional_data_sources']['training_data']
    if training_stats:
        recommendations.append({
            'source': 'SNN Training Data',
            'value': f"{len(training_stats)} datasets with spike training records",
            'integration': 'Add as training/ folder with spike training examples',
            'priority': 'High'
        })
    
    # Supervisor data
    supervisor_stats = summary['additional_data_sources']['supervisor_telemetry']
    if supervisor_stats.get('records', 0) > 0:
        recommendations.append({
            'source': 'Supervisor Telemetry',
            'value': f"{supervisor_stats['records']} supervisor events",
            'integration': 'Add as operations/ folder for system monitoring',
            'priority': 'Medium'
        })
    
    # Mining data
    mining_stats = summary['additional_data_sources']['mining_logs']
    if mining_stats.get('file_size_mb', 0) > 0:
        recommendations.append({
            'source': 'Mining Logs',
            'value': f"{mining_stats['file_size_mb']:.1f} MB of mining operation data",
            'integration': 'Add as mining/ folder with hashrate/temperature metrics',
            'priority': 'High'
        })
    
    # Neuromorphic data
    neuro_stats = summary['additional_data_sources']['neuromorphic_data']
    if neuro_stats.get('file_size_mb', 0) > 0:
        recommendations.append({
            'source': 'Neuromorphic Dataset',
            'value': f"{neuro_stats['file_size_mb']:.1f} MB of neuromorphic records",
            'integration': 'Add as neuromorphic/ folder for advanced research',
            'priority': 'Medium'
        })
    
    print("📋 Integration Plan:")
    for i, rec in enumerate(recommendations, 1):
        print(f"  {i}. {rec['source']} ({rec['priority']} priority)")
        print(f"     Value: {rec['value']}")
        print(f"     Integration: {rec['integration']}")
        print()
    
    return recommendations

def main():
    """Main analysis pipeline"""
    
    # Analyze all additional data
    summary = create_additional_data_summary()
    
    # Create integration recommendations
    recommendations = create_integration_recommendations(summary)
    
    # Save summary
    output_file = Path("additional_data_analysis.json")
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✅ Analysis complete!")
    print(f"📁 Summary saved to: {output_file}")
    print(f"\n🎯 Key Findings:")
    print(f"  • You have extensive additional training data")
    print(f"  • Mining logs contain real operation metrics")
    print(f"  • Supervisor telemetry shows system events")
    print(f"  • Neuromorphic dataset is massive for research")
    
    print(f"\n🚀 Next Steps:")
    print(f"  1. Review integration recommendations")
    print(f"  2. Select high-priority datasets to include")
    print(f"  3. Create additional folder structure")
    print(f"  4. Update dataset documentation")

if __name__ == "__main__":
    main()
