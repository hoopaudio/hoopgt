"""
🏀 HoopGT Clean API Example

Simple example showing how to use HoopGT's clean quantization algorithms.
"""

import torch
import torch.nn as nn
from hoopgt import TargetHardware, DynamicQuantizer, StaticQuantizer


def create_sample_models():
    """Create sample models for demonstration."""
    
    # Transformer-like model (good for dynamic quantization)
    class SimpleTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(1000, 256)
            self.attention = nn.MultiheadAttention(256, 8, batch_first=True)
            self.ffn = nn.Sequential(
                nn.Linear(256, 512),
                nn.ReLU(),
                nn.Linear(512, 256)
            )
            self.output = nn.Linear(256, 10)
            
        def forward(self, x):
            # Simple transformer block
            x = self.embedding(x)
            attn_out, _ = self.attention(x, x, x)
            x = x + attn_out  # Residual connection
            ffn_out = self.ffn(x)
            x = x + ffn_out   # Residual connection
            x = torch.mean(x, dim=1)  # Global average pooling
            return self.output(x)
    
    # CNN model (good for static quantization)
    class SimpleCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 64, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 128, 3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((4, 4))
            )
            self.classifier = nn.Sequential(
                nn.Linear(128 * 16, 256),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(256, 10)
            )
            
        def forward(self, x):
            x = self.features(x)
            x = x.view(x.size(0), -1)
            return self.classifier(x)
    
    return {
        "transformer": SimpleTransformer(),
        "cnn": SimpleCNN()
    }


def example_dynamic_quantization():
    """Example of using dynamic quantization for LLM/VLM models."""
    print("🏀 Dynamic Quantization Example")
    print("=" * 40)
    
    # Create a transformer model
    models = create_sample_models()
    transformer = models["transformer"]
    target = TargetHardware.APPLE_SILICON
    
    print(f"Original model parameters: {sum(p.numel() for p in transformer.parameters()):,}")
    
    # Initialize dynamic quantizer
    quantizer = DynamicQuantizer()
    
    # Check if model can be optimized
    if quantizer.can_optimize(transformer, target):
        print(f"✅ Model can be optimized for {target.value}")
        
        # Get performance estimate
        estimate = quantizer.get_performance_estimate(transformer, target)
        print(f"📊 Estimated speedup: {estimate['speed_up']:.2f}x")
        print(f"💾 Estimated memory reduction: {estimate['memory_reduction']:.2f}x")
        print(f"🎯 Estimated accuracy impact: {estimate['accuracy_impact']:.1%}")
        
        # Apply quantization
        print("\n🔧 Applying dynamic quantization...")
        quantized_model = quantizer.apply(transformer, target)
        
        # Test inference
        print("\n🧪 Testing inference...")
        sample_input = torch.randint(0, 1000, (2, 10))  # batch_size=2, seq_len=10
        
        with torch.no_grad():
            original_output = transformer(sample_input)
            quantized_output = quantized_model(sample_input)
        
        print(f"Original output shape: {original_output.shape}")
        print(f"Quantized output shape: {quantized_output.shape}")
        print(f"Output difference (max): {torch.max(torch.abs(original_output - quantized_output)).item():.6f}")
        
        print("\n✅ Dynamic quantization completed successfully!")
    else:
        print(f"❌ Model cannot be optimized for {target.value}")


def example_static_quantization():
    """Example of using static quantization for CNN models."""
    print("\n\n🏀 Static Quantization Example") 
    print("=" * 40)
    
    # Create a CNN model
    models = create_sample_models()
    cnn = models["cnn"]
    target = TargetHardware.APPLE_SILICON
    
    print(f"Original model parameters: {sum(p.numel() for p in cnn.parameters()):,}")
    
    # Initialize static quantizer
    quantizer = StaticQuantizer()
    
    # Check if model can be optimized
    if quantizer.can_optimize(cnn, target):
        print(f"✅ Model can be optimized for {target.value}")
        
        # Get performance estimate
        estimate = quantizer.get_performance_estimate(cnn, target)
        print(f"📊 Estimated speedup: {estimate['speed_up']:.2f}x")
        print(f"💾 Estimated memory reduction: {estimate['memory_reduction']:.2f}x")
        print(f"🎯 Estimated accuracy impact: {estimate['accuracy_impact']:.1%}")
        
        # Create calibration data (important for static quantization)
        print("\n📋 Creating calibration data...")
        calibration_data = torch.randn(16, 3, 32, 32)  # 16 samples of 32x32 RGB images
        config = {"calibration_data": calibration_data}
        
        # Apply quantization with calibration
        print("\n🔧 Applying static quantization with calibration...")
        quantized_model = quantizer.apply(cnn, target, config)
        
        # Test inference
        print("\n🧪 Testing inference...")
        sample_input = torch.randn(2, 3, 32, 32)  # batch_size=2
        
        with torch.no_grad():
            original_output = cnn(sample_input)
            quantized_output = quantized_model(sample_input)
        
        print(f"Original output shape: {original_output.shape}")
        print(f"Quantized output shape: {quantized_output.shape}")
        print(f"Output difference (max): {torch.max(torch.abs(original_output - quantized_output)).item():.6f}")
        
        print("\n✅ Static quantization completed successfully!")
    else:
        print(f"❌ Model cannot be optimized for {target.value}")


def example_benchmarking():
    """Example of benchmarking quantization performance."""
    print("\n\n🏀 Benchmarking Example")
    print("=" * 40)
    
    models = create_sample_models()
    transformer = models["transformer"]
    target = TargetHardware.APPLE_SILICON
    
    quantizer = DynamicQuantizer()
    
    print("🏃 Running performance benchmark...")
    input_shape = (4, 10)  # batch_size=4, seq_len=10
    
    benchmark_results = quantizer.benchmark(transformer, target, input_shape, runs=10)
    
    print(f"⏱️  Original inference time: {benchmark_results['original_time_ms']:.2f}ms")
    print(f"⏱️  Quantized inference time: {benchmark_results['quantized_time_ms']:.2f}ms")
    print(f"🚀 Actual speedup: {benchmark_results['speedup']:.2f}x")
    print(f"🔄 Benchmark runs: {benchmark_results['runs']}")


def example_multi_target():
    """Example of optimizing for different hardware targets."""
    print("\n\n🏀 Multi-Target Example")
    print("=" * 40)
    
    models = create_sample_models()
    transformer = models["transformer"]
    quantizer = DynamicQuantizer()
    
    targets = [
        TargetHardware.APPLE_SILICON,
        TargetHardware.X86_SERVER,
        TargetHardware.ARM_MOBILE,
        TargetHardware.NVIDIA_JETSON,
    ]
    
    for target in targets:
        print(f"\n🎯 Target: {target.value}")
        
        if quantizer.can_optimize(transformer, target):
            config = quantizer.get_optimization_config(transformer, target)
            print(f"   Backend: {config['backend']}")
            print(f"   Description: {config['description']}")
            
            estimate = quantizer.get_performance_estimate(transformer, target)
            print(f"   Estimated speedup: {estimate['speed_up']:.2f}x")
            print(f"   ✅ Optimization available")
        else:
            print(f"   ❌ Optimization not available")


def main():
    """Run all examples."""
    print("🏀 HoopGT Clean API Examples")
    print("Demonstrating production-ready edge AI optimization")
    print("=" * 60)
    
    try:
        example_dynamic_quantization()
        example_static_quantization()
        example_benchmarking()
        example_multi_target()
        
        print("\n\n🎉 All examples completed successfully!")
        print("\n🚀 Ready for:")
        print("   • LLM/VLM optimization for edge devices")
        print("   • Private AI assistant deployment")  
        print("   • B2B quantization services")
        
    except Exception as e:
        print(f"\n❌ Example failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 