#!/usr/bin/env python3
"""
Simple parameter conversion for Spikenaut SNN v2
"""

import numpy as np
import torch
import json
from pathlib import Path
from datetime import datetime

def create_parameters():
    """Create sample parameters"""
    
    # Hidden layer weights (16x8)
    hidden_weights = torch.randn(16, 8) * 0.1
    
    # Thresholds (16)
    thresholds = torch.linspace(0.5, 2.0, 16)
    
    # Decay (16)
    decay = torch.linspace(0.8, 0.95, 16)
    
    # Output weights (3x16)
    output_weights = torch.randn(3, 16) * 0.1
    
    return {
        'hidden_layer.weight': hidden_weights,
        'hidden_layer.threshold': thresholds,
        'hidden_layer.decay': decay,
        'output_layer.weight': output_weights
    }

def save_pytorch_format(parameters, filepath):
    """Save in PyTorch format"""
    torch.save(parameters, filepath)
    print(f"✅ Saved PyTorch format: {filepath}")

def save_fpga_format(parameters, prefix):
    """Save in FPGA Q8.8 format"""
    
    def float_to_q8_8(value):
        value = np.clip(value, -128, 127.996)
        return int(value * 256)
    
    def save_tensor_as_mem(tensor, filename):
        numpy_array = tensor.cpu().numpy()
        with open(filename, 'w') as f:
            if numpy_array.ndim == 1:
                for val in numpy_array:
                    q8_8 = float_to_q8_8(val)
                    f.write(f"{q8_8:04X}\n")
            elif numpy_array.ndim == 2:
                for row in numpy_array:
                    for val in row:
                        q8_8 = float_to_q8_8(val)
                        f.write(f"{q8_8:04X}\n")
        print(f"✅ Saved FPGA format: {filename}")
    
    save_tensor_as_mem(parameters['hidden_layer.weight'], f"{prefix}_hidden_weights.mem")
    save_tensor_as_mem(parameters['hidden_layer.threshold'], f"{prefix}_thresholds.mem")
    save_tensor_as_mem(parameters['hidden_layer.decay'], f"{prefix}_decay.mem")
    save_tensor_as_mem(parameters['output_layer.weight'], f"{prefix}_output_weights.mem")

def main():
    print("🔄 Simple Spikenaut SNN v2 Parameter Conversion")
    
    # Create parameters
    parameters = create_parameters()
    
    # Create output directory
    output_dir = Path("converted_parameters")
    output_dir.mkdir(exist_ok=True)
    
    # Save PyTorch format
    save_pytorch_format(parameters, output_dir / "spikenaut_snn_v2.pth")
    
    # Save FPGA format
    save_fpga_format(parameters, str(output_dir / "spikenaut_snn_v2"))
    
    print(f"\n✅ Conversion completed!")
    print(f"📁 Output directory: {output_dir}")

if __name__ == "__main__":
    main()
