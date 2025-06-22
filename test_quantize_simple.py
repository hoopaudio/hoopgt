"""
🏀 Simple HoopGT Quantization Test

Test M4 quantization functionality.
"""

import torch
import torch.nn as nn
import torch.ao.quantization as quant
import platform
import sys
import os

# Import from hoopgt package
from hoopgt import HoopQuantizer


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(5, 1)
    
    def forward(self, x):
        return self.linear(x)


def test_m4_quantization():
    """Test quantization on M4"""
    
    print("🏀 Testing HoopGT Quantization on M4")
    print(f"🔍 System: {platform.system()} {platform.machine()}")
    
    # Initialize quantizer and define target
    quantizer = HoopQuantizer()
    target_hardware = "apple-silicon"
    print(f"🎯 Explicitly setting target to: {target_hardware}")
    
    # Create test model
    model = SimpleModel()
    model.eval()
    
    print(f"📊 Original model parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Test dynamic quantization
    print("\n🚀 Testing dynamic quantization...")
    quantized_model, target = quantizer.quantize_dynamic(model, target_hardware)
    
    print(f"✅ Quantized for: {target}")
    
    # Test inference
    test_input = torch.randn(1, 5)
    with torch.no_grad():
        original_output = model(test_input)
        quantized_output = quantized_model(test_input)
        
        print(f"📈 Original output: {original_output.item():.4f}")
        print(f"📉 Quantized output: {quantized_output.item():.4f}")
        print(f"🔄 Difference: {abs(original_output.item() - quantized_output.item()):.4f}")
    
    # Check size reduction
    stats = quantizer.get_model_size_reduction(model, quantized_model)
    print(f"\n📊 Model Stats:")
    print(f"   Original: {stats['original_size_mb']:.3f} MB")
    print(f"   Quantized: {stats['quantized_size_mb']:.3f} MB") 
    print(f"   Reduction: {stats['reduction_ratio']:.1f}x")
    print(f"   Savings: {stats['size_savings_percent']:.1f}%")
    
    return True


if __name__ == "__main__":
    try:
        success = test_m4_quantization()
        if success:
            print("\n🎉 M4 quantization test completed successfully!")
        else:
            print("\n❌ Test failed")
            sys.exit(1)
    except Exception as e:
        print(f"\n💥 Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1) 