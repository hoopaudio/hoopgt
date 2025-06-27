"""
🏀 Test HoopGT New Quantization Architecture

Demonstrates the integration of MVP quantizer with plugin architecture.
"""

import torch
import torch.nn as nn
from hoopgt import QuantizationEngine, OptimizationConfig, TargetHardware


class TestModel(nn.Module):
    """Simple test model for quantization testing"""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3)
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.linear1 = nn.Linear(32, 64)
        self.lstm = nn.LSTM(64, 32, batch_first=True)
        self.linear2 = nn.Linear(32, 10)
    
    def forward(self, x):
        # Conv layers
        x = self.conv1(x)
        x = self.conv2(x)
        
        # Global average pooling to handle variable input sizes
        x = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = x.flatten(1)
        
        # Linear + LSTM
        x = self.linear1(x)
        x = x.unsqueeze(1)  # Add sequence dimension
        x, _ = self.lstm(x)
        x = x.squeeze(1)    # Remove sequence dimension
        x = self.linear2(x)
        return x


def test_new_architecture():
    """Test the new quantization architecture"""
    
    print("🏀 Testing HoopGT New Quantization Architecture")
    print("=" * 50)
    
    # Create test model
    model = TestModel()
    model.eval()
    target = TargetHardware.APPLE_SILICON
    
    print(f"Model: {model.__class__.__name__}")
    print(f"Target: {target.value}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Initialize the new quantization engine
    engine = QuantizationEngine()
    
    print(f"\n📋 Available algorithms: {engine.get_available_algorithms()}")
    
    # Test 1: Auto-quantization (should use plugin algorithm)
    print(f"\n🎯 Test 1: Auto-quantization")
    try:
        quantized_model_auto, stats_auto = engine.auto_quantize(model, target)
        print(f"   ✅ Success with {stats_auto['algorithm']}")
        print(f"   📊 Size reduction: {stats_auto['reduction_ratio']:.1f}x")
        print(f"   📈 Estimated speedup: {stats_auto.get('estimated_speedup', 'N/A'):.1f}x")
    except Exception as e:
        print(f"   ❌ Failed: {e}")
    
    # Test 2: Specific plugin algorithm
    print(f"\n🔧 Test 2: Specific plugin algorithm (torch_dynamic)")
    try:
        quantized_model_plugin, stats_plugin = engine.quantize_with_algorithm(
            model, "torch_dynamic", target
        )
        print(f"   ✅ Success with torch_dynamic")
        print(f"   📊 Size reduction: {stats_plugin['reduction_ratio']:.1f}x")
        print(f"   📈 Estimated speedup: {stats_plugin['estimated_speedup']:.1f}x")
    except Exception as e:
        print(f"   ❌ Failed: {e}")
    
    # Test 3: MVP fallback
    print(f"\n🔄 Test 3: MVP fallback (dynamic)")
    try:
        quantized_model_mvp, stats_mvp = engine.quantize_with_algorithm(
            model, "dynamic", target
        )
        print(f"   ✅ Success with MVP dynamic")
        print(f"   📊 Size reduction: {stats_mvp['reduction_ratio']:.1f}x")
    except Exception as e:
        print(f"   ❌ Failed: {e}")
    
    # Test 4: Configuration-based optimization
    print(f"\n⚙️  Test 4: Configuration-based optimization")
    try:
        config = OptimizationConfig.for_apple_silicon()
        config.set_algorithm_config("torch_dynamic", {"dtype": torch.qint8})
        
        quantized_model_config, stats_config = engine.auto_quantize(model, target, config)
        print(f"   ✅ Success with configuration")
        print(f"   📊 Algorithm: {stats_config['algorithm']}")
        print(f"   📊 Size reduction: {stats_config['reduction_ratio']:.1f}x")
    except Exception as e:
        print(f"   ❌ Failed: {e}")
    
    # Test 5: Benchmark multiple algorithms
    print(f"\n🏁 Test 5: Algorithm benchmark")
    try:
        results = engine.benchmark_algorithms(model, target, ["torch_dynamic", "dynamic"])
        print(f"   📊 Benchmark Results:")
        for algo, stats in results.items():
            if "error" in stats:
                print(f"      {algo}: ❌ {stats['error']}")
            else:
                print(f"      {algo}: ✅ {stats['reduction_ratio']:.1f}x reduction")
    except Exception as e:
        print(f"   ❌ Benchmark failed: {e}")
    
    print(f"\n🎉 Architecture test completed!")


def test_inference():
    """Test that quantized models can still run inference"""
    
    print(f"\n🔬 Testing quantized model inference")
    
    model = TestModel()
    model.eval()
    target = TargetHardware.APPLE_SILICON
    
    # Create test input
    test_input = torch.randn(1, 3, 8, 8)  # Batch, channels, height, width
    
    # Get original output
    with torch.no_grad():
        original_output = model(test_input)
    
    # Quantize and test
    engine = QuantizationEngine()
    quantized_model, stats = engine.auto_quantize(model, target)
    
    with torch.no_grad():
        quantized_output = quantized_model(test_input)
    
    # Compare outputs
    mse = torch.nn.functional.mse_loss(original_output, quantized_output).item()
    print(f"   📊 MSE between original and quantized: {mse:.6f}")
    print(f"   📊 Max absolute difference: {torch.max(torch.abs(original_output - quantized_output)).item():.6f}")
    
    if mse < 0.1:  # Reasonable threshold
        print(f"   ✅ Inference test passed!")
    else:
        print(f"   ⚠️  High MSE - check quantization quality")


if __name__ == "__main__":
    test_new_architecture()
    test_inference() 