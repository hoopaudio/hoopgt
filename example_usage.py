"""
🏀 HoopGT Usage Examples

Demonstrates both MVP quantization and new plugin architecture.
"""

import torch
import torch.nn as nn
from hoopgt import (
    QuantizationEngine, 
    HoopQuantizer,  # MVP
    OptimizationConfig, 
    TargetHardware
)


# Simple model for testing
class ExampleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3)
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.linear = nn.Linear(32, 10)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = x.flatten(1)
        return self.linear(x)


def example_mvp_usage():
    """Example using your original MVP quantizer"""
    print("🎯 MVP Quantization Example")
    print("-" * 30)
    
    model = ExampleModel()
    model.eval()
    
    # Original MVP approach
    quantizer = HoopQuantizer()
    target = "apple-silicon"
    
    # Auto-select best method
    method = quantizer.get_recommended_method(model, target)
    print(f"Recommended method: {method}")
    
    # Apply quantization
    quantized_model, _ = quantizer.quantize_dynamic(model, target)
    
    # Check results
    stats = quantizer.get_model_size_reduction(model, quantized_model)
    print(f"Size reduction: {stats['reduction_ratio']:.1f}x")
    print(f"Savings: {stats['size_savings_percent']:.1f}%")


def example_new_engine():
    """Example using the new plugin architecture"""
    print("\n🚀 New Plugin Architecture Example")
    print("-" * 40)
    
    model = ExampleModel()
    model.eval()
    target = TargetHardware.APPLE_SILICON
    
    # Initialize new engine
    engine = QuantizationEngine()
    
    print(f"Available algorithms: {engine.get_available_algorithms()}")
    
    # Method 1: Auto-quantization (recommended)
    print("\n1. Auto-quantization:")
    quantized_model, stats = engine.auto_quantize(model, target)
    print(f"   Used: {stats['algorithm']}")
    print(f"   Size reduction: {stats['reduction_ratio']:.1f}x")
    print(f"   Estimated speedup: {stats['estimated_speedup']:.1f}x")
    
    # Method 2: Specific algorithm
    print("\n2. Specific algorithm:")
    quantized_model2, stats2 = engine.quantize_with_algorithm(
        model, "torch_dynamic", target
    )
    print(f"   Size reduction: {stats2['reduction_ratio']:.1f}x")
    
    # Method 3: With configuration
    print("\n3. With configuration:")
    config = OptimizationConfig.for_apple_silicon()
    config.set_algorithm_config("torch_dynamic", {"dtype": torch.qint8})
    
    quantized_model3, stats3 = engine.auto_quantize(model, target, config)
    print(f"   Used: {stats3['algorithm']}")
    print(f"   Size reduction: {stats3['reduction_ratio']:.1f}x")


def example_benchmark():
    """Example of benchmarking multiple algorithms"""
    print("\n📊 Algorithm Benchmark Example")
    print("-" * 35)
    
    model = ExampleModel()
    model.eval()
    target = TargetHardware.APPLE_SILICON
    
    engine = QuantizationEngine()
    
    # Benchmark all compatible algorithms
    results = engine.benchmark_algorithms(model, target)
    
    print("Benchmark Results:")
    for algo, stats in results.items():
        if "error" in stats:
            print(f"  {algo}: ❌ {stats['error']}")
        else:
            print(f"  {algo}: ✅ {stats['reduction_ratio']:.1f}x reduction")


def example_adding_new_algorithm():
    """Example showing how easy it is to add new quantization algorithms"""
    print("\n🔧 Adding New Algorithm Example")
    print("-" * 35)
    
    # This is how you would add a new quantization method
    # (You don't need to run this, it's just for demonstration)
    
    code_example = '''
    # In hoopgt/algorithms/quantization/my_new_quantizer.py:
    
    class MyCustomQuantizer(HoopGTQuantizerBase):
        algorithm_name = "my_custom"
        description = "My custom quantization method"
        
        supported_targets = [TargetHardware.APPLE_SILICON]
        
        def can_optimize(self, model, target):
            return True  # Your logic here
            
        def get_optimization_config(self, model, target):
            return {"custom_param": "value"}
            
        def apply(self, model, target, config=None):
            # Your quantization logic here
            return model
    
    # In hoopgt/algorithms/quantization/__init__.py:
    
    QUANTIZATION_ALGORITHMS = {
        "torch_dynamic": TorchDynamicQuantizer,
        "my_custom": MyCustomQuantizer,  # <- Add this line
    }
    
    # That's it! Now it's automatically available:
    
    engine = QuantizationEngine()
    engine.quantize_with_algorithm(model, "my_custom", target)
    '''
    
    print("Adding a new algorithm is as simple as:")
    print(code_example)


if __name__ == "__main__":
    print("🏀 HoopGT Quantization Examples")
    print("=" * 50)
    
    example_mvp_usage()
    example_new_engine()
    example_benchmark()
    example_adding_new_algorithm()
    
    print("\n🎉 Examples completed! Your torch implementation is ready and the")
    print("   architecture welcomes new quantization algorithms!") 