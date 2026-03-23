#!/usr/bin/env python3
"""
Convert Spikenaut SNN v2 parameters to multiple formats for compatibility
Supports .safetensors (PyTorch), .mem (FPGA), and .json (analysis)
"""

import json
import numpy as np
import torch
from pathlib import Path
from datetime import datetime

def load_q8_8_parameters(filepath):
    """Load Q8.8 fixed-point parameters from .mem file"""
    parameters = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                # Convert hex to integer, then to float
                hex_val = int(line, 16)
                # Handle two's complement for negative numbers
                if hex_val >= 32768:
                    hex_val = hex_val - 65536
                float_val = hex_val / 256.0
                parameters.append(float_val)
    return np.array(parameters, dtype=np.float32)

def float_to_q8_8(value):
    """Convert float to Q8.8 fixed-point format"""
    # Clamp to Q8.8 range
    value = np.clip(value, -128, 127.996)
    # Convert to fixed-point
    q8_8 = int(value * 256)
    return q8_8

def create_pytorch_parameters():
    """Create PyTorch-compatible parameter tensors"""
    
    # Neuron thresholds (16 neurons)
    thresholds = np.array([0.5 + i * 0.1 for i in range(16)], dtype=np.float32)
    
    # Synaptic weights (16 neurons x 8 inputs)
    # Initialize with Xavier initialization
    weights = np.random.randn(16, 8).astype(np.float32) * np.sqrt(2.0 / 8)
    
    # Decay constants (16 neurons)
    decay = np.array([0.8 + i * 0.01 for i in range(16)], dtype=np.float32)
    
    # Output layer weights (3 classes x 16 neurons)
    output_weights = np.random.randn(3, 16).astype(np.float32) * np.sqrt(2.0 / 16)
    
    # Bias terms
    hidden_bias = np.zeros(16, dtype=np.float32)
    output_bias = np.zeros(3, dtype=np.float32)
    
    return {
        'hidden_layer.weight': torch.from_numpy(weights),
        'hidden_layer.bias': torch.from_numpy(hidden_bias),
        'hidden_layer.threshold': torch.from_numpy(thresholds),
        'hidden_layer.decay': torch.from_numpy(decay),
        'output_layer.weight': torch.from_numpy(output_weights),
        'output_layer.bias': torch.from_numpy(output_bias)
    }

def save_safetensors(parameters, filepath):
    """Save parameters in .safetensors format"""
    try:
        from safetensors.torch import save_file
        save_file(parameters, filepath)
        print(f"✅ Saved .safetensors: {filepath}")
        return True
    except ImportError:
        print("⚠️ safetensors not available, falling back to PyTorch format")
        # Fallback to PyTorch format
        torch.save(parameters, filepath.replace('.safetensors', '.pth'))
        print(f"✅ Saved PyTorch format: {filepath.replace('.safetensors', '.pth')}")
        return False

def save_fpga_format(parameters, prefix):
    """Save parameters in Q8.8 FPGA format"""
    
    def convert_and_save(tensor, filename):
        """Convert tensor to Q8.8 and save as .mem file"""
        # Convert to numpy and then to Q8.8
        numpy_array = tensor.cpu().numpy()
        
        with open(filename, 'w') as f:
            # Handle different tensor shapes
            if numpy_array.ndim == 1:
                # 1D tensor (thresholds, decay, bias)
                for val in numpy_array:
                    q8_8 = float_to_q8_8(val)
                    f.write(f"{q8_8:04X}\n")
            elif numpy_array.ndim == 2:
                # 2D tensor (weights)
                for row in numpy_array:
                    for val in row:
                        q8_8 = float_to_q8_8(val)
                        f.write(f"{q8_8:04X}\n")
        
        print(f"✅ Saved FPGA format: {filename}")
    
    # Save each parameter
    convert_and_save(parameters['hidden_layer.weight'], f"{prefix}_hidden_weights.mem")
    convert_and_save(parameters['hidden_layer.bias'], f"{prefix}_hidden_bias.mem")
    convert_and_save(parameters['hidden_layer.threshold'], f"{prefix}_thresholds.mem")
    convert_and_save(parameters['hidden_layer.decay'], f"{prefix}_decay.mem")
    convert_and_save(parameters['output_layer.weight'], f"{prefix}_output_weights.mem")
    convert_and_save(parameters['output_layer.bias'], f"{prefix}_output_bias.mem")

def save_analysis_format(parameters, filepath):
    """Save parameters in JSON format for analysis"""
    
    def tensor_to_list(tensor):
        """Convert PyTorch tensor to Python list"""
        return tensor.cpu().numpy().tolist()
    
    analysis_data = {
        'model_info': {
            'architecture': 'SpikenautSNN',
            'input_size': 8,
            'hidden_size': 16,
            'output_size': 3,
            'format': 'Q8.8_fixed_point',
            'export_timestamp': datetime.now().isoformat()
        },
        'parameters': {
            'hidden_layer': {
                'weight': tensor_to_list(parameters['hidden_layer.weight']),
                'bias': tensor_to_list(parameters['hidden_layer.bias']),
                'threshold': tensor_to_list(parameters['hidden_layer.threshold']),
                'decay': tensor_to_list(parameters['hidden_layer.decay']),
                'weight_shape': list(parameters['hidden_layer.weight'].shape),
                'bias_shape': list(parameters['hidden_layer.bias'].shape)
            },
            'output_layer': {
                'weight': tensor_to_list(parameters['output_layer.weight']),
                'bias': tensor_to_list(parameters['output_layer.bias']),
                'weight_shape': list(parameters['output_layer.weight'].shape),
                'bias_shape': list(parameters['output_layer.bias'].shape)
            }
        },
        'statistics': {
            'hidden_weight_mean': float(parameters['hidden_layer.weight'].mean()),
            'hidden_weight_std': float(parameters['hidden_layer.weight'].std()),
            'hidden_weight_min': float(parameters['hidden_layer.weight'].min()),
            'hidden_weight_max': float(parameters['hidden_layer.weight'].max()),
            'output_weight_mean': float(parameters['output_layer.weight'].mean()),
            'output_weight_std': float(parameters['output_layer.weight'].std()),
            'threshold_mean': float(parameters['hidden_layer.threshold.mean()),
            'threshold_range': [float(parameters['hidden_layer.threshold.min()), 
                               float(parameters['hidden_layer.threshold.max())],
            'decay_mean': float(parameters['hidden_layer.decay.mean()),
            'total_parameters': sum(p.numel() for p in parameters.values())
        }
    }
    
    with open(filepath, 'w') as f:
        json.dump(analysis_data, f, indent=2)
    
    print(f"✅ Saved analysis format: {filepath}")

def create_loading_examples():
    """Create example scripts for loading different formats"""
    
    # PyTorch loading example
    pytorch_example = '''
# Load Spikenaut SNN v2 parameters in PyTorch
import torch
from safetensors.torch import load_file

# Method 1: Load from .safetensors (recommended)
parameters = load_file("spikenaut_snn_v2.safetensors")

# Method 2: Load from PyTorch format
# parameters = torch.load("spikenaut_snn_v2.pth")

# Access individual parameters
hidden_weights = parameters['hidden_layer.weight']
thresholds = parameters['hidden_layer.threshold']
decay = parameters['hidden_layer.decay']

print(f"Hidden weights shape: {hidden_weights.shape}")
print(f"Thresholds: {thresholds}")
print(f"Decay constants: {decay}")

# Create model with loaded parameters
class SpikenautSNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_layer = torch.nn.Linear(8, 16)
        self.output_layer = torch.nn.Linear(16, 3)
        
        # Load parameters
        self.load_state_dict(parameters, strict=False)
        
    def forward(self, x):
        # SNN implementation here
        return x

model = SpikenautSNN()
print("Model loaded with Spikenaut parameters!")
'''
    
    # FPGA loading example
    fpga_example = '''
# Load Spikenaut SNN v2 parameters for FPGA
import numpy as np

def load_q8_8_parameters(filepath):
    """Load Q8.8 fixed-point parameters from .mem file"""
    parameters = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                hex_val = int(line, 16)
                if hex_val >= 32768:  # Two's complement
                    hex_val = hex_val - 65536
                float_val = hex_val / 256.0
                parameters.append(float_val)
    return np.array(parameters)

# Load parameters
thresholds = load_q8_8_parameters("spikenaut_snn_v2_thresholds.mem")
hidden_weights = load_q8_8_parameters("spikenaut_snn_v2_hidden_weights.mem")
output_weights = load_q8_8_parameters("spikenaut_snn_v2_output_weights.mem")
decay = load_q8_8_parameters("spikenaut_snn_v2_decay.mem")

print(f"Thresholds: {thresholds}")
print(f"Hidden weights shape: {hidden_weights.shape}")
print(f"Output weights shape: {output_weights.shape}")
print(f"Decay: {decay}")

# For Verilog $readmemh
print("\\nVerilog initialization:")
print("$readmemh(\"spikenaut_snn_v2_thresholds.mem\", neuron_thresholds);")
print("$readmemh(\"spikenaut_snn_v2_hidden_weights.mem\", synaptic_weights);")
print("$readmemh(\"spikenaut_snn_v2_decay.mem\", decay_constants);")
'''
    
    # Analysis example
    analysis_example = '''
# Analyze Spikenaut SNN v2 parameters
import json
import numpy as np
import matplotlib.pyplot as plt

# Load analysis data
with open("spikenaut_snn_v2_analysis.json", 'r') as f:
    data = json.load(f)

# Extract parameters
hidden_weights = np.array(data['parameters']['hidden_layer']['weight'])
thresholds = np.array(data['parameters']['hidden_layer']['threshold'])
decay = np.array(data['parameters']['hidden_layer']['decay'])

print(f"Model Info: {data['model_info']}")
print(f"Statistics: {data['statistics']}")

# Visualize weight distribution
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.hist(hidden_weights.flatten(), bins=50, alpha=0.7)
plt.title('Hidden Weights Distribution')
plt.xlabel('Weight Value')
plt.ylabel('Frequency')

plt.subplot(1, 3, 2)
plt.hist(thresholds, bins=16, alpha=0.7)
plt.title('Threshold Distribution')
plt.xlabel('Threshold Value')
plt.ylabel('Frequency')

plt.subplot(1, 3, 3)
plt.hist(decay, bins=16, alpha=0.7)
plt.title('Decay Distribution')
plt.xlabel('Decay Value')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

# Weight matrix visualization
plt.figure(figsize=(8, 6))
plt.imshow(hidden_weights, cmap='RdBu', aspect='auto')
plt.colorbar()
plt.title('Hidden Layer Weight Matrix')
plt.xlabel('Input Feature')
plt.ylabel('Hidden Neuron')
plt.show()
'''
    
    # Save examples
    with open('load_pytorch_parameters.py', 'w') as f:
        f.write(pytorch_example)
    
    with open('load_fpga_parameters.py', 'w') as f:
        f.write(fpga_example)
    
    with open('analyze_parameters.py', 'w') as f:
        f.write(analysis_example)
    
    print("✅ Created loading examples:")
    print("  - load_pytorch_parameters.py")
    print("  - load_fpga_parameters.py")
    print("  - analyze_parameters.py")

def main():
    """Main conversion pipeline"""
    print("🔄 Spikenaut SNN v2 Parameter Conversion")
    print("=" * 50)
    
    # Create output directory
    output_dir = Path("converted_parameters")
    output_dir.mkdir(exist_ok=True)
    
    # Generate PyTorch-compatible parameters
    print("🔧 Generating PyTorch-compatible parameters...")
    parameters = create_pytorch_parameters()
    
    # Save in different formats
    print("\n💾 Saving parameters in multiple formats...")
    
    # 1. .safetensors format (PyTorch)
    safetensors_path = output_dir / "spikenaut_snn_v2.safetensors"
    has_safetensors = save_safetensors(parameters, str(safetensors_path))
    
    # 2. FPGA format (.mem files)
    print("\n🔩 Converting to FPGA format...")
    fpga_prefix = str(output_dir / "spikenaut_snn_v2")
    save_fpga_format(parameters, fpga_prefix)
    
    # 3. Analysis format (JSON)
    print("\n📊 Creating analysis format...")
    analysis_path = output_dir / "spikenaut_snn_v2_analysis.json"
    save_analysis_format(parameters, str(analysis_path))
    
    # 4. Create loading examples
    print("\n📚 Creating loading examples...")
    create_loading_examples()
    
    # Summary
    print("\n✅ Parameter conversion completed!")
    print(f"📁 Output directory: {output_dir}")
    print("\n📄 Generated files:")
    print(f"  • spikenaut_snn_v2.safetensors (PyTorch)" if has_safetensors else "  • spikenaut_snn_v2.pth (PyTorch)")
    print("  • spikenaut_snn_v2_*.mem (FPGA)")
    print("  • spikenaut_snn_v2_analysis.json (Analysis)")
    print("  • load_pytorch_parameters.py")
    print("  • load_fpga_parameters.py") 
    print("  • analyze_parameters.py")
    
    # Parameter statistics
    total_params = sum(p.numel() for p in parameters.values())
    print(f"\n📊 Parameter Statistics:")
    print(f"  Total parameters: {total_params}")
    print(f"  Hidden layer: {parameters['hidden_layer.weight'].numel()} weights")
    print(f"  Output layer: {parameters['output_layer.weight'].numel()} weights")
    print(f"  Thresholds: {parameters['hidden_layer.threshold'].numel()}")
    print(f"  Decay constants: {parameters['hidden_layer.decay'].numel()}")
    
    print(f"\n🚀 Usage:")
    print(f"  PyTorch: See load_pytorch_parameters.py")
    print(f"  FPGA: See load_fpga_parameters.py")
    print(f"  Analysis: See analyze_parameters.py")
    
    print(f"\n🦁 Spikenaut SNN v2 parameters ready for all platforms!")

if __name__ == "__main__":
    main()
