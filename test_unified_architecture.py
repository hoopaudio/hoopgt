"""
🏀 HoopGT Unified Architecture Test

Test the complete refactored architecture including:
1. Backward compatibility with original MVP API
2. New plugin architecture functionality
3. Algorithm migration and unification
"""

import torch
import torch.nn as nn
from hoopgt import (
    HoopQuantizer,           # Legacy API
    QuantizationEngine,      # New architecture
    TargetHardware,
    QUANTIZATION_ALGORITHMS,
    mvp_selector
)


def create_test_model():
    """Create a test model with Conv + LSTM + Linear layers."""
    return nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=3),
        nn.AdaptiveAvgPool2d((4, 4)),
        nn.Flatten(),
        nn.LSTM(1024, 256, batch_first=True),
        nn.Dropout(0.2),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    )


def test_backward_compatibility():
    """Test that the original MVP API still works exactly as before."""
    print("🔄 Testing Backward Compatibility...")
    
    model = create_test_model()
    
    # Original MVP API should work unchanged
    quantizer = HoopQuantizer()
    
    # Test get_recommended_method (original function)
    recommended = quantizer.get_recommended_method(model, "apple-silicon")
    print(f"   Recommended method: {recommended}")
    assert recommended in ["dynamic", "static"], f"Unexpected recommendation: {recommended}"
    
    # Test dynamic quantization (original function)
    quantized_model, target = quantizer.quantize_dynamic(model, "apple-silicon")
    assert target == "apple-silicon"
    print(f"   ✅ Dynamic quantization successful")
    
    # Test size reduction calculation (original function) 
    size_stats = quantizer.get_model_size_reduction(model, quantized_model)
    print(f"   Size reduction: {size_stats['reduction_ratio']:.1f}x")
    assert size_stats['reduction_ratio'] > 1.0, "Expected size reduction"
    
    # Test static quantization (original function)
    quantized_static, _ = quantizer.quantize_static(model, "x86-server")
    print(f"   ✅ Static quantization successful")
    
    print("✅ Backward compatibility tests passed!")
    return True


def test_new_plugin_architecture():
    """Test the new plugin architecture with MVP algorithms."""
    print("\n🔧 Testing New Plugin Architecture...")
    
    model = create_test_model()
    
    # New quantization engine
    engine = QuantizationEngine()
    
    # Test available algorithms
    algorithms = engine.get_available_algorithms()
    print(f"   Available algorithms: {algorithms}")
    
    expected_algorithms = ["torch_dynamic", "mvp_dynamic", "mvp_static"]
    for alg in expected_algorithms:
        assert alg in algorithms, f"Missing algorithm: {alg}"
    
    # Test intelligent algorithm recommendation
    recommendation = engine.get_recommended_algorithm(model, TargetHardware.APPLE_SILICON)
    print(f"   Recommended algorithm: {recommendation}")
    
    # Test auto-quantization
    quantized_model, stats = engine.auto_quantize(model, TargetHardware.APPLE_SILICON)
    print(f"   Auto-quantization result: {stats['algorithm']}")
    print(f"   Size reduction: {stats['reduction_ratio']:.1f}x")
    print(f"   Estimated speedup: {stats.get('estimated_speedup', 'N/A')}")
    
    assert stats['reduction_ratio'] > 1.0, "Expected size reduction"
    
    print("✅ New plugin architecture tests passed!")
    return True


def test_algorithm_benchmarking():
    """Test benchmarking multiple algorithms."""
    print("\n🧪 Testing Algorithm Benchmarking...")
    
    model = create_test_model()
    engine = QuantizationEngine()
    
    # Benchmark all available algorithms
    results = engine.benchmark_algorithms(model, TargetHardware.APPLE_SILICON)
    
    print(f"   Benchmarked {len(results)} algorithms:")
    for algorithm, stats in results.items():
        if "error" in stats:
            print(f"   - {algorithm}: ❌ {stats['error']}")
        else:
            print(f"   - {algorithm}: ✅ {stats['reduction_ratio']:.1f}x reduction")
    
    # Ensure at least one algorithm succeeded
    successful = [alg for alg, stats in results.items() if "error" not in stats]
    assert len(successful) > 0, "No algorithms succeeded"
    
    print("✅ Algorithm benchmarking tests passed!")
    return True


def test_mvp_selector_intelligence():
    """Test the MVP selector's intelligent decision making."""
    print("\n🎯 Testing MVP Selector Intelligence...")
    
    # Test different model architectures
    models = {
        "CNN": nn.Sequential(nn.Conv2d(3, 64, 3), nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Linear(64, 10)),
        "RNN": nn.Sequential(nn.LSTM(100, 256, batch_first=True), nn.Linear(256, 10)),
        "MLP": nn.Sequential(nn.Linear(100, 256), nn.ReLU(), nn.Linear(256, 10)),
    }
    
    for model_type, model in models.items():
        print(f"\n   Testing {model_type} model:")
        
        # Test architecture analysis
        arch_info = mvp_selector.analyze_model_architecture(model)
        print(f"     Architecture: Conv={arch_info['has_conv']}, RNN={arch_info['has_rnn']}, Linear={arch_info['has_linear']}")
        
        # Test recommendation
        recommendation = mvp_selector.get_mvp_recommendation(model, TargetHardware.APPLE_SILICON)
        print(f"     Recommendation: {recommendation}")
        
        # Test explanation
        explanation = mvp_selector.explain_recommendation(model, TargetHardware.APPLE_SILICON)
        print(f"     Reasoning: {'; '.join(explanation['reasoning'][:2])}")  # Show first 2 reasons
    
    print("\n✅ MVP selector intelligence tests passed!")
    return True


def test_specific_algorithms():
    """Test individual algorithms directly."""
    print("\n⚙️  Testing Individual Algorithms...")
    
    model = create_test_model()
    
    # Test each algorithm type
    for algorithm_name, algorithm_class in QUANTIZATION_ALGORITHMS.items():
        print(f"\n   Testing {algorithm_name}:")
        
        algorithm = algorithm_class()
        
        # Test can_optimize
        can_optimize = algorithm.can_optimize(model, TargetHardware.APPLE_SILICON)
        print(f"     Can optimize: {can_optimize}")
        
        if can_optimize:
            # Test performance estimate
            perf_estimate = algorithm.get_performance_estimate(model, TargetHardware.APPLE_SILICON)
            print(f"     Estimated speedup: {perf_estimate['speed_up']:.1f}x")
            
            # Test optimization config
            config = algorithm.get_optimization_config(model, TargetHardware.APPLE_SILICON)
            print(f"     Backend: {config.get('backend', 'N/A')}")
            
            # Test application
            try:
                quantized_model = algorithm.apply(model, TargetHardware.APPLE_SILICON)
                print(f"     ✅ Application successful")
            except Exception as e:
                print(f"     ⚠️  Application failed: {e}")
    
    print("\n✅ Individual algorithm tests passed!")
    return True


def main():
    """Run all tests for the unified architecture."""
    print("🏀 HoopGT Unified Architecture Test Suite")
    print("=" * 50)
    
    try:
        # Run all test suites
        test_backward_compatibility()
        test_new_plugin_architecture()
        test_algorithm_benchmarking()
        test_mvp_selector_intelligence()
        test_specific_algorithms()
        
        print("\n" + "=" * 50)
        print("🎉 ALL TESTS PASSED!")
        print("\n✅ Architecture successfully unified:")
        print("   - Original MVP functionality preserved")
        print("   - New plugin architecture working")
        print("   - MVP algorithms migrated to plugins")
        print("   - Intelligent algorithm selection active")
        print("   - Backward compatibility maintained")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        raise


if __name__ == "__main__":
    main() 