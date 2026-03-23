
# Compare YOUR legacy data with v2 telemetry data
import matplotlib.pyplot as plt
import seaborn as sns

def compare_legacy_vs_v2():
    """Compare legacy trading data with v2 telemetry"""
    
    # Load legacy data
    legacy_df = load_legacy_data()
    
    # Load v2 data (current dataset)
    from datasets import load_dataset
    try:
        v2_ds = load_dataset("rmems/Spikenaut-SNN-v2-Telemetry-Data-Weights-Parameters")
        v2_df = v2_ds['train'].to_pandas()
        print("✅ V2 dataset loaded")
    except:
        print("⚠️ V2 dataset not available, using sample")
        v2_df = None
    
    print("\n🔍 Dataset Comparison:")
    print(f"Legacy: {len(legacy_df):,} records (trading focus)")
    if v2_df is not None:
        print(f"V2: {len(v2_df)} records (telemetry focus)")
    
    # Compare time ranges
    if 'timestamp' in legacy_df.columns:
        legacy_df['timestamp'] = pd.to_datetime(legacy_df['timestamp'])
        print(f"\n⏰ Time Coverage:")
        print(f"Legacy: {legacy_df['timestamp'].min()} to {legacy_df['timestamp'].max()}")
        print(f"Duration: {legacy_df['timestamp'].max() - legacy_df['timestamp'].min()}")
    
    # Compare data types
    print(f"\n📋 Data Types:")
    print(f"Legacy focus: Trading actions, portfolio management, blockchain metrics")
    if v2_df is not None:
        print(f"V2 focus: Blockchain telemetry, spike encodings, SNN features")
    
    # Visualize portfolio evolution (legacy)
    if 'portfolio_value' in legacy_df.columns:
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        # Sample every 1000th point for performance
        sample_legacy = legacy_df.iloc[::1000]
        plt.plot(sample_legacy.index, sample_legacy['portfolio_value'], alpha=0.7)
        plt.title('🦁 Legacy Portfolio Evolution')
        plt.xlabel('Record Index')
        plt.ylabel('Portfolio Value ($)')
        plt.grid(True, alpha=0.3)
        
        # Action distribution
        plt.subplot(1, 2, 2)
        action_counts = legacy_df['action'].value_counts()
        plt.pie(action_counts.values, labels=action_counts.index, autopct='%1.1f%%')
        plt.title('Legacy Action Distribution')
        
        plt.tight_layout()
        plt.show()
    
    print("\n🎯 Key Insights:")
    print("• Legacy: Rich trading history with 200K+ records")
    print("• V2: Focused telemetry with spike encodings")
    print("• Combined: Complete picture of Spikenaut evolution")

# Run comparison
compare_legacy_vs_v2()
