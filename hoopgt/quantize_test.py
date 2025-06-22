"""
🏀 HoopGT Quantization Tests

Test quantization on Apple Silicon M4.
"""

import torch
import torch.nn as nn
import tempfile
import pytest
from pathlib import Path
from hoopgt.quantize import HoopQuantizer


class TestModel(nn.Module):
    """Simple test model for quantization testing"""
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(10, 20)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(20, 1)
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x


def test_quantizer_initialization():
    """Test HoopQuantizer initialization"""
    quantizer = HoopQuantizer()
    assert "apple-silicon" in quantizer.target_configs
    assert "x86-server" in quantizer.target_configs
    assert "arm-mobile" in quantizer.target_configs


def test_quantization_config():
    """Test quantization configuration"""
    quantizer = HoopQuantizer()
    
    # Test dynamic config for Apple Silicon
    config = quantizer.get_quantization_config("apple-silicon", "dynamic")
    assert config["backend"] == "qnnpack"
    assert config["dtype"] == torch.qint8
    
    # Test static config
    config = quantizer.get_quantization_config("apple-silicon", "static")
    assert config["backend"] == "qnnpack"
    assert config["qconfig"] is not None


def test_dynamic_quantization():
    """Test dynamic quantization on Apple Silicon"""
    quantizer = HoopQuantizer()
    model = TestModel()
    model.eval()
    
    # Test dynamic quantization
    quantized_model, target = quantizer.quantize_dynamic(model, "apple-silicon")
    
    assert target == "apple-silicon"
    assert quantized_model is not None
    
    # Test that model can still make predictions
    test_input = torch.randn(1, 10)
    with torch.no_grad():
        output = quantized_model(test_input)
        assert output.shape == (1, 1)
    
    print("✅ Dynamic quantization test passed!")


def test_model_size_reduction():
    """Test model size calculation"""
    quantizer = HoopQuantizer()
    
    original_model = TestModel()
    quantized_model, _ = quantizer.quantize_dynamic(original_model, "apple-silicon")
    
    stats = quantizer.get_model_size_reduction(original_model, quantized_model)
    
    assert "original_size_mb" in stats
    assert "quantized_size_mb" in stats
    assert "reduction_ratio" in stats
    assert "size_savings_percent" in stats
    
    # Should have some size reduction
    assert stats["reduction_ratio"] > 1.0
    assert stats["size_savings_percent"] > 0
    
    print(f"📊 Size reduction: {stats['reduction_ratio']:.2f}x")


def test_backend_configuration():
    """Test that backends are correctly configured"""
    quantizer = HoopQuantizer()
    
    # Test Apple Silicon uses qnnpack
    config = quantizer.get_quantization_config("apple-silicon", "dynamic")
    assert config["backend"] == "qnnpack"
    
    # Test x86 uses fbgemm
    config = quantizer.get_quantization_config("x86-server", "dynamic")
    assert config["backend"] == "fbgemm"
    
    print("✅ Backend configuration test passed!")


if __name__ == "__main__":
    print("🧪 Running HoopGT quantization tests...")
    test_quantizer_initialization()
    test_quantization_config()
    test_dynamic_quantization()
    test_model_size_reduction()
    test_backend_configuration()
    print("🎉 All quantization tests passed!") 