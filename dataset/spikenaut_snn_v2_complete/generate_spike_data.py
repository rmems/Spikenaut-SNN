#!/usr/bin/env python3
"""
Generate spike-encoded versions of telemetry data for Spikenaut SNN v2
Creates neural representations and temporal covariance matrices
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from scipy import signal
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt

class SpikeEncoder:
    """Convert telemetry data to spike trains and neural representations"""
    
    def __init__(self, window_size=10, n_channels=16):
        self.window_size = window_size
        self.n_channels = n_channels
        self.spike_history = []
        
        # Channel mapping for 16-neuron architecture
        self.channel_map = {
            0: 'kaspa_hashrate',
            1: 'kaspa_power', 
            2: 'kaspa_temp',
            3: 'kaspa_qubic',
            4: 'monero_hashrate',
            5: 'monero_power',
            6: 'monero_temp', 
            7: 'monero_qubic',
            8: 'qubic_hashrate',
            9: 'qubic_power',
            10: 'qubic_temp',
            11: 'qubic_qubic',
            12: 'thermal_stress',
            13: 'power_efficiency',
            14: 'network_health',
            15: 'composite_reward'
        }
        
        # Spike encoding parameters
        self.thresholds = {
            'hashrate': {'low': 0.5, 'high': 1.5},
            'power': {'low': 370, 'high': 410},
            'temp': {'low': 40, 'high': 46},
            'qubic': {'low': 0.8, 'high': 0.98}
        }
    
    def load_telemetry_data(self, filepath):
        """Load telemetry JSONL data"""
        data = []
        with open(filepath, 'r') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        return data
    
    def temporal_encoding(self, value, channel_type, timestamp):
        """Temporal spike encoding with adaptive thresholds"""
        thresh_range = self.thresholds.get(channel_type, {'low': 0, 'high': 1})
        
        # Normalize to [0, 1]
        if channel_type == 'hashrate':
            normalized = np.clip((value - thresh_range['low']) / (thresh_range['high'] - thresh_range['low']), 0, 1)
        elif channel_type == 'power':
            normalized = np.clip((value - thresh_range['low']) / (thresh_range['high'] - thresh_range['low']), 0, 1)
        elif channel_type == 'temp':
            normalized = np.clip((value - thresh_range['low']) / (thresh_range['high'] - thresh_range['low']), 0, 1)
        else:  # qubic and others
            normalized = np.clip(value, 0, 1)
        
        # Poisson spike generation with rate modulation
        spike_rate = normalized * 100  # Max 100 Hz
        spike_prob = spike_rate / 1000  # Convert to probability per ms
        
        # Generate spike
        spike = 1 if np.random.random() < spike_prob else 0
        
        return {
            'value': value,
            'normalized': normalized,
            'spike': spike,
            'rate': spike_rate,
            'timestamp': timestamp
        }
    
    def encode_single_event(self, event):
        """Encode a single telemetry event into 16-channel spikes"""
        timestamp = datetime.strptime(event['timestamp'], "%Y-%m-%d %H:%M:%S.%f")
        telemetry = event['telemetry']
        blockchain = event['blockchain']
        
        spikes = {}
        
        # Basic telemetry channels (0-11)
        if blockchain == 'kaspa':
            spikes[0] = self.temporal_encoding(telemetry['hashrate_mh'], 'hashrate', timestamp)
            spikes[1] = self.temporal_encoding(telemetry['power_w'], 'power', timestamp)
            spikes[2] = self.temporal_encoding(telemetry['gpu_temp_c'], 'temp', timestamp)
            spikes[3] = self.temporal_encoding(telemetry['qubic_tick_trace'], 'qubic', timestamp)
        elif blockchain == 'monero':
            spikes[4] = self.temporal_encoding(telemetry['hashrate_mh'], 'hashrate', timestamp)
            spikes[5] = self.temporal_encoding(telemetry['power_w'], 'power', timestamp)
            spikes[6] = self.temporal_encoding(telemetry['gpu_temp_c'], 'temp', timestamp)
            spikes[7] = self.temporal_encoding(telemetry['qubic_tick_trace'], 'qubic', timestamp)
        elif blockchain == 'qubic':
            spikes[8] = self.temporal_encoding(telemetry['hashrate_mh'], 'hashrate', timestamp)
            spikes[9] = self.temporal_encoding(telemetry['power_w'], 'power', timestamp)
            spikes[10] = self.temporal_encoding(telemetry['gpu_temp_c'], 'temp', timestamp)
            spikes[11] = self.temporal_encoding(telemetry['qubic_tick_trace'], 'qubic', timestamp)
        
        # Derived channels (12-15)
        # Thermal stress (combined temperature indicator)
        temp_stress = (telemetry['gpu_temp_c'] - 40) / 6  # Normalize 40-46°C range
        spikes[12] = self.temporal_encoding(temp_stress, 'temp', timestamp)
        
        # Power efficiency (MH/kW)
        power_eff = telemetry['hashrate_mh'] / (telemetry['power_w'] / 1000)
        spikes[13] = self.temporal_encoding(power_eff / 5, 'hashrate', timestamp)  # Normalize to ~0-1
        
        # Network health (composite of qubic metrics)
        network_health = (telemetry['qubic_tick_trace'] + telemetry['qubic_epoch_progress']) / 2
        spikes[14] = self.temporal_encoding(network_health, 'qubic', timestamp)
        
        # Composite reward
        composite_reward = telemetry['reward_hint']
        spikes[15] = self.temporal_encoding(composite_reward, 'qubic', timestamp)
        
        return spikes
    
    def create_spike_train(self, data):
        """Convert full dataset to spike trains"""
        spike_trains = []
        
        for i, event in enumerate(data):
            spikes = self.encode_single_event(event)
            
            # Create spike vector
            spike_vector = np.zeros(self.n_channels)
            spike_rates = np.zeros(self.n_channels)
            normalized_values = np.zeros(self.n_channels)
            
            for channel_idx, spike_data in spikes.items():
                spike_vector[channel_idx] = spike_data['spike']
                spike_rates[channel_idx] = spike_data['rate']
                normalized_values[channel_idx] = spike_data['normalized']
            
            spike_trains.append({
                'timestamp': event['timestamp'],
                'blockchain': event['blockchain'],
                'event_type': event['event'],
                'spike_vector': spike_vector.tolist(),
                'spike_rates': spike_rates.tolist(),
                'normalized_values': normalized_values.tolist(),
                'raw_spikes': {str(k): v for k, v in spikes.items()}
            })
        
        return spike_trains
    
    def compute_temporal_covariance(self, spike_trains):
        """Compute temporal covariance matrices for neuromorphic training"""
        if len(spike_trains) < self.window_size:
            return None
        
        # Create spike matrix (time x channels)
        spike_matrix = np.array([train['spike_vector'] for train in spike_trains])
        
        # Compute rolling window covariances
        covariances = []
        for i in range(len(spike_matrix) - self.window_size + 1):
            window = spike_matrix[i:i + self.window_size]
            
            # Compute covariance matrix
            cov_matrix = np.cov(window.T)
            
            # Add temporal information
            covariances.append({
                'window_start': spike_trains[i]['timestamp'],
                'window_end': spike_trains[i + self.window_size - 1]['timestamp'],
                'covariance_matrix': cov_matrix.tolist(),
                'eigenvalues': np.linalg.eigvals(cov_matrix).tolist(),
                'spike_rate_mean': window.mean(axis=0).tolist(),
                'spike_correlation': np.corrcoef(window.T).tolist() if window.shape[0] > 1 else np.eye(self.n_channels).tolist()
            })
        
        return covariances
    
    def generate_forecast_targets(self, data, horizon=1):
        """Generate forecasting targets for time series prediction"""
        targets = []
        
        for i in range(len(data) - horizon):
            current = data[i]
            future = data[i + horizon]
            
            # Compute changes
            current_telemetry = current['telemetry']
            future_telemetry = future['telemetry']
            
            target = {
                'timestamp': current['timestamp'],
                'blockchain': current['blockchain'],
                'horizon_ticks': horizon,
                'target_hashrate_change': future_telemetry['hashrate_mh'] - current_telemetry['hashrate_mh'],
                'target_power_change': future_telemetry['power_w'] - current_telemetry['power_w'],
                'target_temp_change': future_telemetry['gpu_temp_c'] - current_telemetry['gpu_temp_c'],
                'target_qubic_change': future_telemetry['qubic_tick_trace'] - current_telemetry['qubic_tick_trace'],
                'target_reward_change': future_telemetry['reward_hint'] - current_telemetry['reward_hint'],
                'target_block_rate_change': future.get('block_rate', 0) - current.get('block_rate', 0)
            }
            
            targets.append(target)
        
        return targets
    
    def create_derived_features(self, data):
        """Create additional derived features for ML"""
        derived = []
        
        for i, event in enumerate(data):
            telemetry = event['telemetry']
            
            # Efficiency metrics
            power_efficiency = telemetry['hashrate_mh'] / (telemetry['power_w'] / 1000)  # MH/kW
            thermal_efficiency = telemetry['hashrate_mh'] / telemetry['gpu_temp_c']  # MH/°C
            qubic_efficiency = telemetry['qubic_tick_trace'] * telemetry['qubic_epoch_progress']
            
            # Stress indicators
            power_stress = max(0, (telemetry['power_w'] - 400) / 20)  # Stress above 400W
            thermal_stress = max(0, (telemetry['gpu_temp_c'] - 44) / 4)  # Stress above 44°C
            
            # Network health score
            network_health = (telemetry['qubic_tick_trace'] + telemetry['qubic_epoch_progress'] + telemetry['reward_hint']) / 3
            
            # Composite performance score
            performance_score = (telemetry['hashrate_mh'] / 2.0 + power_efficiency / 5.0 + network_health) / 3
            
            derived_features = {
                'timestamp': event['timestamp'],
                'blockchain': event['blockchain'],
                'power_efficiency_mh_per_kw': power_efficiency,
                'thermal_efficiency_mh_per_c': thermal_efficiency,
                'qubic_efficiency_score': qubic_efficiency,
                'power_stress_level': power_stress,
                'thermal_stress_level': thermal_stress,
                'network_health_score': network_health,
                'composite_performance_score': performance_score,
                'is_stressed': 1 if (power_stress > 0.5 or thermal_stress > 0.5) else 0,
                'performance_tier': self.get_performance_tier(performance_score)
            }
            
            derived.append(derived_features)
        
        return derived
    
    def get_performance_tier(self, score):
        """Classify performance into tiers"""
        if score >= 0.8:
            return 'excellent'
        elif score >= 0.6:
            return 'good'
        elif score >= 0.4:
            return 'moderate'
        else:
            return 'poor'
    
    def save_spike_data(self, spike_trains, covariances, targets, derived, output_dir):
        """Save all generated spike data"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Save spike trains
        with open(output_path / "spike_trains.jsonl", 'w') as f:
            for train in spike_trains:
                f.write(json.dumps(train) + '\n')
        
        # Save covariances
        if covariances:
            with open(output_path / "temporal_covariances.jsonl", 'w') as f:
                for cov in covariances:
                    f.write(json.dumps(cov) + '\n')
        
        # Save forecast targets
        with open(output_path / "forecast_targets.jsonl", 'w') as f:
            for target in targets:
                f.write(json.dumps(target) + '\n')
        
        # Save derived features
        with open(output_path / "derived_features.jsonl", 'w') as f:
            for feature in derived:
                f.write(json.dumps(feature) + '\n')
        
        # Save summary statistics
        stats = {
            'total_spike_trains': len(spike_trains),
            'total_covariances': len(covariances) if covariances else 0,
            'total_targets': len(targets),
            'total_derived': len(derived),
            'n_channels': self.n_channels,
            'window_size': self.window_size,
            'generation_timestamp': datetime.now().isoformat()
        }
        
        with open(output_path / "spike_generation_stats.json", 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"✅ Spike data saved to {output_path}")
        print(f"   - Spike trains: {len(spike_trains)}")
        print(f"   - Temporal covariances: {len(covariances) if covariances else 0}")
        print(f"   - Forecast targets: {len(targets)}")
        print(f"   - Derived features: {len(derived)}")

def main():
    """Main spike generation pipeline"""
    print("🦁 Spikenaut SNN v2 - Spike Data Generation")
    print("=" * 50)
    
    # Configuration
    input_file = "fresh_sync_data.jsonl"
    output_dir = "spike_encoded_data"
    
    print(f"Input file: {input_file}")
    print(f"Output directory: {output_dir}")
    print()
    
    # Initialize encoder
    encoder = SpikeEncoder(window_size=5, n_channels=16)
    
    # Load telemetry data
    print("📂 Loading telemetry data...")
    data = encoder.load_telemetry_data(input_file)
    print(f"   Loaded {len(data)} telemetry events")
    
    # Generate spike trains
    print("🔸 Generating spike trains...")
    spike_trains = encoder.create_spike_train(data)
    print(f"   Generated {len(spike_trains)} spike trains")
    
    # Compute temporal covariances
    print("🔗 Computing temporal covariances...")
    covariances = encoder.compute_temporal_covariance(spike_trains)
    print(f"   Generated {len(covariances) if covariances else 0} covariance windows")
    
    # Generate forecast targets
    print("🎯 Generating forecast targets...")
    targets = encoder.generate_forecast_targets(data, horizon=1)
    print(f"   Generated {len(targets)} forecast targets")
    
    # Create derived features
    print("📊 Creating derived features...")
    derived = encoder.create_derived_features(data)
    print(f"   Created {len(derived)} derived feature records")
    
    # Save all data
    print("💾 Saving spike data...")
    encoder.save_spike_data(spike_trains, covariances, targets, derived, output_dir)
    
    print("\n✅ Spike data generation completed!")
    print(f"📁 Check {output_dir}/ for all generated files")

if __name__ == "__main__":
    main()
