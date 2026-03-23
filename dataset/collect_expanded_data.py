#!/usr/bin/env python3
"""
Continuous telemetry logger for Spikenaut SNN v2 dataset expansion
Collects 24-72 hours of blockchain telemetry with spike encoding
"""

import json
import time
import logging
import subprocess
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import threading
import queue
import random

class TelemetryCollector:
    """Continuous blockchain telemetry collection with spike encoding"""
    
    def __init__(self, output_dir="expanded_data", collection_hours=24):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.collection_hours = collection_hours
        self.end_time = datetime.now() + timedelta(hours=collection_hours)
        
        # Data queues for different sources
        self.kaspa_queue = queue.Queue()
        self.monero_queue = queue.Queue()
        self.qubic_queue = queue.Queue()
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.output_dir / "collection.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Spike encoding thresholds (adaptive)
        self.thresholds = {
            'hashrate': 0.9,
            'power': 390,
            'temp': 43,
            'qubic': 0.95
        }
        
        # Statistics
        self.stats = {
            'kaspa_events': 0,
            'monero_events': 0,
            'qubic_events': 0,
            'total_samples': 0,
            'start_time': datetime.now()
        }
    
    def simulate_kaspa_telemetry(self):
        """Simulate Kaspa mainnet telemetry (for demo/testing)"""
        while datetime.now() < self.end_time:
            # Simulate realistic Kaspa block patterns
            base_hashrate = 0.8 + random.uniform(-0.2, 0.4)
            power = 380 + random.uniform(-10, 20)
            temp = 42 + random.uniform(-2, 4)
            
            # Block acceptance events (bursty pattern)
            if random.random() < 0.7:  # 70% chance of block batch
                blocks_accepted = random.randint(5, 15)
                block_rate = blocks_accepted / random.uniform(0.5, 2.0)
                
                event = {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
                    "blockchain": "kaspa",
                    "event": "block_acceptance",
                    "blocks_accepted": blocks_accepted,
                    "block_rate": round(block_rate, 2),
                    "telemetry": {
                        "hashrate_mh": round(base_hashrate, 2),
                        "power_w": round(power, 1),
                        "gpu_temp_c": round(temp, 1),
                        "qubic_tick_trace": round(random.uniform(0.95, 1.0), 3),
                        "qubic_epoch_progress": round(random.uniform(0.998, 1.0), 4),
                        "reward_hint": round(random.uniform(0.998, 1.0), 4)
                    }
                }
                
                self.kaspa_queue.put(event)
                self.stats['kaspa_events'] += 1
            
            time.sleep(random.uniform(1, 5))  # Variable interval
    
    def simulate_monero_telemetry(self):
        """Simulate Monero sync telemetry (for demo/testing)"""
        while datetime.now() < self.end_time:
            # Simulate sync progress patterns
            current_height = 3635000 + random.randint(0, 1000)
            total_height = current_height + random.randint(50, 200)
            sync_percent = current_height / total_height
            
            power = 390 + random.uniform(-5, 15)
            temp = 41 + random.uniform(-1, 3)
            
            event = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
                "blockchain": "monero",
                "event": "sync_progress" if sync_percent < 0.999 else "sync_complete",
                "current_height": current_height,
                "total_height": total_height,
                "sync_percent": round(sync_percent, 6),
                "remaining_blocks": max(0, total_height - current_height),
                "telemetry": {
                    "hashrate_mh": round(0.85 + random.uniform(-0.1, 0.1), 2),
                    "power_w": round(power, 1),
                    "gpu_temp_c": round(temp, 1),
                    "qubic_tick_trace": round(random.uniform(0.8, 0.95), 3),
                    "qubic_epoch_progress": round(sync_percent, 4),
                    "reward_hint": round(sync_percent, 4)
                }
            }
            
            self.monero_queue.put(event)
            self.stats['monero_events'] += 1
            
            time.sleep(random.uniform(2, 8))  # Slower sync events
    
    def simulate_qubic_telemetry(self):
        """Simulate Qubic network telemetry (for demo/testing)"""
        while datetime.now() < self.end_time:
            # Qubic has different patterns - epoch-based
            epoch_progress = random.uniform(0, 1)
            tick_trace = random.uniform(0.7, 1.0)
            
            event = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
                "blockchain": "qubic",
                "event": "epoch_tick" if epoch_progress < 0.99 else "epoch_complete",
                "epoch_id": random.randint(1000, 9999),
                "tick_id": random.randint(1, 1000),
                "epoch_progress": round(epoch_progress, 4),
                "telemetry": {
                    "hashrate_mh": round(0.6 + random.uniform(-0.2, 0.3), 2),
                    "power_w": round(385 + random.uniform(-10, 15), 1),
                    "gpu_temp_c": round(44 + random.uniform(-2, 3), 1),
                    "qubic_tick_trace": round(tick_trace, 3),
                    "qubic_epoch_progress": round(epoch_progress, 4),
                    "reward_hint": round(tick_trace * epoch_progress, 4)
                }
            }
            
            self.qubic_queue.put(event)
            self.stats['qubic_events'] += 1
            
            time.sleep(random.uniform(0.5, 3))  # Fast Qubic ticks
    
    def encode_spikes(self, telemetry):
        """Convert telemetry to spike trains"""
        spikes = {}
        
        # Adaptive thresholds (update based on recent history)
        spikes['hashrate_spike'] = 1 if telemetry['hashrate_mh'] > self.thresholds['hashrate'] else 0
        spikes['power_spike'] = 1 if telemetry['power_w'] > self.thresholds['power'] else 0
        spikes['temp_spike'] = 1 if telemetry['gpu_temp_c'] > self.thresholds['temp'] else 0
        spikes['qubic_spike'] = 1 if telemetry['qubic_tick_trace'] > self.thresholds['qubic'] else 0
        
        # Composite spike (multiple simultaneous)
        spike_sum = sum(spikes.values())
        spikes['composite_spike'] = 1 if spike_sum >= 2 else 0
        
        return spikes
    
    def enhance_with_features(self, event):
        """Add derived features and spike encodings"""
        enhanced = event.copy()
        
        # Add temporal features
        timestamp = datetime.strptime(event['timestamp'], "%Y-%m-%d %H:%M:%S.%f")
        enhanced['timestamp_unix'] = timestamp.timestamp()
        enhanced['hour_of_day'] = timestamp.hour
        enhanced['day_of_week'] = timestamp.weekday()
        
        # Add efficiency metrics
        telemetry = event['telemetry']
        enhanced['hashrate_normalized'] = telemetry['hashrate_mh'] / 2.0
        enhanced['power_efficiency'] = telemetry['hashrate_mh'] / (telemetry['power_w'] / 1000.0)
        enhanced['thermal_efficiency'] = telemetry['hashrate_mh'] / telemetry['gpu_temp_c']
        
        # Add spike encodings
        spikes = self.encode_spikes(telemetry)
        enhanced.update(spikes)
        
        # Add composite reward signal
        reward_components = [
            telemetry['qubic_epoch_progress'],
            telemetry['reward_hint'],
            enhanced['hashrate_normalized']
        ]
        enhanced['composite_reward'] = np.mean(reward_components)
        
        return enhanced
    
    def collect_and_process(self):
        """Main collection loop"""
        self.logger.info(f"Starting {self.collection_hours}-hour telemetry collection...")
        self.logger.info(f"End time: {self.end_time}")
        
        # Start collector threads
        collectors = [
            threading.Thread(target=self.simulate_kaspa_telemetry, daemon=True),
            threading.Thread(target=self.simulate_monero_telemetry, daemon=True),
            threading.Thread(target=self.simulate_qubic_telemetry, daemon=True)
        ]
        
        for collector in collectors:
            collector.start()
        
        # Output files
        raw_file = self.output_dir / "expanded_raw_data.jsonl"
        enhanced_file = self.output_dir / "expanded_enhanced_data.jsonl"
        spike_file = self.output_dir / "spike_encodings.jsonl"
        
        # Process events
        with open(raw_file, 'w') as raw_f, open(enhanced_file, 'w') as enhanced_f, open(spike_file, 'w') as spike_f:
            
            while datetime.now() < self.end_time:
                events_processed = 0
                
                # Process all queues
                for queue in [self.kaspa_queue, self.monero_queue, self.qubic_queue]:
                    try:
                        event = queue.get_nowait()
                        
                        # Write raw event
                        raw_f.write(json.dumps(event) + '\n')
                        
                        # Enhance and write
                        enhanced = self.enhance_with_features(event)
                        enhanced_f.write(json.dumps(enhanced) + '\n')
                        
                        # Extract just spike data
                        spike_data = {
                            'timestamp': event['timestamp'],
                            'blockchain': event['blockchain'],
                            'spikes': {k: v for k, v in enhanced.items() if 'spike' in k}
                        }
                        spike_f.write(json.dumps(spike_data) + '\n')
                        
                        events_processed += 1
                        self.stats['total_samples'] += 1
                        
                    except queue.Empty:
                        continue
                
                # Adaptive threshold updates (every 100 samples)
                if self.stats['total_samples'] % 100 == 0 and events_processed > 0:
                    self.update_thresholds()
                
                # Log progress
                if self.stats['total_samples'] % 50 == 0:
                    elapsed = datetime.now() - self.stats['start_time']
                    rate = self.stats['total_samples'] / elapsed.total_seconds() * 60  # per minute
                    self.logger.info(f"Progress: {self.stats['total_samples']} samples, {rate:.1f} samples/min")
                
                time.sleep(0.1)  # Small delay to prevent CPU spinning
        
        self.logger.info("Collection completed!")
        self.log_final_stats()
    
    def update_thresholds(self):
        """Adaptively update spike thresholds based on recent data"""
        # Simple adaptive logic: adjust thresholds slightly based on recent averages
        # In real implementation, this would use rolling statistics
        self.thresholds['hashrate'] *= random.uniform(0.95, 1.05)
        self.thresholds['power'] *= random.uniform(0.98, 1.02)
        self.thresholds['temp'] *= random.uniform(0.99, 1.01)
        self.thresholds['qubic'] *= random.uniform(0.97, 1.03)
    
    def log_final_stats(self):
        """Log collection statistics"""
        elapsed = datetime.now() - self.stats['start_time']
        rate = self.stats['total_samples'] / elapsed.total_seconds()
        
        stats_msg = f"""
=== Collection Statistics ===
Duration: {elapsed}
Total Samples: {self.stats['total_samples']}
Collection Rate: {rate:.2f} samples/second
- Kaspa Events: {self.stats['kaspa_events']}
- Monero Events: {self.stats['monero_events']}  
- Qubic Events: {self.stats['qubic_events']}
Files Created:
- expanded_raw_data.jsonl
- expanded_enhanced_data.jsonl
- spike_encodings.jsonl
- collection.log
"""
        self.logger.info(stats_msg)
        
        # Save stats
        with open(self.output_dir / "collection_stats.json", 'w') as f:
            json.dump({
                **self.stats,
                'end_time': datetime.now().isoformat(),
                'collection_hours': self.collection_hours,
                'samples_per_second': rate
            }, f, indent=2, default=str)

def main():
    """Run the expanded data collection"""
    print("🦁 Spikenaut SNN v2 - Expanded Telemetry Collection")
    print("=" * 50)
    
    # Configuration
    collection_hours = 1  # Set to 24-72 for real collection
    output_dir = "expanded_data"
    
    print(f"Collection duration: {collection_hours} hours")
    print(f"Output directory: {output_dir}")
    print("Press Ctrl+C to stop early")
    print()
    
    try:
        collector = TelemetryCollector(
            output_dir=output_dir,
            collection_hours=collection_hours
        )
        collector.collect_and_process()
        
        print("\n✅ Collection completed successfully!")
        print(f"📊 Check {output_dir}/ for results")
        
    except KeyboardInterrupt:
        print("\n⚠️ Collection stopped by user")
    except Exception as e:
        print(f"\n❌ Error during collection: {e}")

if __name__ == "__main__":
    main()
