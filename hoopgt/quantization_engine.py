"""
🏀 HoopGT Quantization Engine v2

Combines the MVP quantization engine with the new plugin architecture.
"""

import torch
from typing import Tuple, Dict, Any, Optional, List
from .types import TargetHardware, QuantizationMethod
from .config.optimization_config import OptimizationConfig
from .algorithms.quantization import QUANTIZATION_ALGORITHMS, mvp_selector
from .algorithms.hoopgt_base import HoopGTQuantizerBase
from .legacy_quantize import HoopQuantizer  # Backward-compatible MVP interface


class QuantizationEngine:
    """
    Advanced quantization engine that combines MVP functionality with plugin architecture.
    
    This engine:
    1. Maintains backward compatibility with your existing HoopQuantizer
    2. Adds support for pluggable quantization algorithms
    3. Provides intelligent algorithm selection
    4. Enables easy addition of new quantization methods
    """
    
    def __init__(self):
        """Initialize the quantization engine."""
        self.mvp_quantizer = HoopQuantizer()  # Your original quantizer
        self.plugin_algorithms = self._load_algorithms()
        
    def _load_algorithms(self) -> Dict[str, HoopGTQuantizerBase]:
        """Load and instantiate all available quantization algorithms."""
        algorithms = {}
        for name, algorithm_class in QUANTIZATION_ALGORITHMS.items():
            try:
                algorithms[name] = algorithm_class()
                print(f"✅ Loaded quantization algorithm: {name}")
            except Exception as e:
                print(f"⚠️  Failed to load algorithm {name}: {e}")
        return algorithms
    
    def get_available_algorithms(self) -> List[str]:
        """Get list of all available quantization algorithms."""
        mvp_methods = ["dynamic", "static", "qat"]  # From your MVP
        plugin_methods = list(self.plugin_algorithms.keys())
        return mvp_methods + plugin_methods
    
    def get_recommended_algorithm(
        self, 
        model: torch.nn.Module, 
        target: TargetHardware,
        prefer_plugin: bool = False  # Changed default to prefer MVP (proven algorithms)
    ) -> str:
        """
        Get the recommended quantization algorithm for this model and target.
        
        Uses the original MVP's intelligent selection logic, now enhanced with plugin support.
        
        Args:
            model: PyTorch model to analyze
            target: Target hardware platform
            prefer_plugin: If True, prefer new plugin algorithms over proven MVP methods
            
        Returns:
            Name of recommended algorithm
        """
        
        # Get all available algorithm names
        available_algorithms = list(self.plugin_algorithms.keys())
        
        # Use MVP selector for intelligent recommendation
        if prefer_plugin:
            # Try plugin algorithms first, fall back to MVP
            recommendation = mvp_selector.get_expanded_recommendation(model, target, available_algorithms)
        else:
            # Prefer proven MVP algorithms
            mvp_recommendation = mvp_selector.get_mvp_recommendation(model, target)
            if mvp_recommendation in available_algorithms:
                recommendation = mvp_recommendation
            else:
                # Fall back to expanded recommendation if MVP algorithm not available
                recommendation = mvp_selector.get_expanded_recommendation(model, target, available_algorithms)
        
        print(f"🎯 Recommended algorithm: {recommendation}")
        return recommendation
    
    def quantize_with_algorithm(
        self,
        model: torch.nn.Module,
        algorithm_name: str,
        target: TargetHardware,
        config: Optional[Dict[str, Any]] = None
    ) -> Tuple[torch.nn.Module, Dict[str, Any]]:
        """
        Apply specific quantization algorithm to model.
        
        Args:
            model: PyTorch model to quantize
            algorithm_name: Name of algorithm to use
            target: Target hardware platform
            config: Optional algorithm configuration
            
        Returns:
            Tuple of (quantized_model, stats)
        """
        
        # Check if it's a plugin algorithm
        if algorithm_name in self.plugin_algorithms:
            return self._quantize_with_plugin(model, algorithm_name, target, config)
        
        # Handle MVP algorithms (both new and legacy names)
        elif algorithm_name in ["dynamic", "static", "qat"]:
            return self._quantize_with_mvp(model, algorithm_name, target, config)
        elif algorithm_name in ["mvp_dynamic", "mvp_static"]:
            # New MVP algorithm names - map to legacy methods
            legacy_method = algorithm_name.replace("mvp_", "")
            return self._quantize_with_mvp(model, legacy_method, target, config)
        
        else:
            raise ValueError(f"Unknown quantization algorithm: {algorithm_name}")
    
    def _quantize_with_plugin(
        self,
        model: torch.nn.Module,
        algorithm_name: str,
        target: TargetHardware,
        config: Optional[Dict[str, Any]] = None
    ) -> Tuple[torch.nn.Module, Dict[str, Any]]:
        """Apply plugin quantization algorithm."""
        
        algorithm = self.plugin_algorithms[algorithm_name]
        
        # Check compatibility
        if not algorithm.can_optimize(model, target):
            raise ValueError(f"Algorithm {algorithm_name} cannot optimize this model for {target.value}")
        
        # Get performance estimate before optimization
        perf_estimate = algorithm.get_performance_estimate(model, target)
        
        # Apply quantization
        original_size = self._get_model_size(model)
        quantized_model = algorithm.apply(model, target, config)
        quantized_size = self._get_model_size(quantized_model)
        
        # Calculate actual stats
        stats = {
            "algorithm": algorithm_name,
            "target": target.value,
            "original_size_mb": original_size / (1024 * 1024),
            "quantized_size_mb": quantized_size / (1024 * 1024),
            "reduction_ratio": original_size / quantized_size if quantized_size > 0 else 1.0,
            "size_savings_percent": (1 - quantized_size / original_size) * 100,
            "estimated_speedup": perf_estimate["speed_up"],
            "estimated_accuracy_impact": perf_estimate["accuracy_impact"],
        }
        
        return quantized_model, stats
    
    def _quantize_with_mvp(
        self,
        model: torch.nn.Module,
        method: str,
        target: TargetHardware,
        config: Optional[Dict[str, Any]] = None
    ) -> Tuple[torch.nn.Module, Dict[str, Any]]:
        """Apply MVP quantization method."""
        
        print(f"🔧 Using MVP quantization method: {method}")
        
        if method == "dynamic":
            quantized_model, _ = self.mvp_quantizer.quantize_dynamic(model, target.value)
        elif method == "static":
            calibration_data = config.get("calibration_data") if config else None
            quantized_model, _ = self.mvp_quantizer.quantize_static(
                model, target.value, calibration_data
            )
        else:
            raise ValueError(f"MVP method {method} not yet implemented")
        
        # Calculate stats using MVP method
        size_stats = self.mvp_quantizer.get_model_size_reduction(model, quantized_model)
        
        # Create properly typed stats dictionary
        stats = {
            "algorithm": f"mvp_{method}",
            "target": target.value,
            "original_size_mb": size_stats["original_size_mb"],
            "quantized_size_mb": size_stats["quantized_size_mb"],
            "reduction_ratio": size_stats["reduction_ratio"],
            "size_savings_percent": size_stats["size_savings_percent"],
        }
        
        return quantized_model, stats
    
    def auto_quantize(
        self,
        model: torch.nn.Module,
        target: TargetHardware,
        optimization_config: Optional[OptimizationConfig] = None
    ) -> Tuple[torch.nn.Module, Dict[str, Any]]:
        """
        Automatically select and apply the best quantization for this model and target.
        
        Args:
            model: PyTorch model to quantize
            target: Target hardware platform
            optimization_config: Optional optimization configuration
            
        Returns:
            Tuple of (quantized_model, stats)
        """
        
        print(f"🎯 Auto-quantizing model for {target.value}")
        
        # Auto-select algorithm
        algorithm_name = self.get_recommended_algorithm(model, target)
        
        # Extract algorithm-specific config if provided
        config = None
        if optimization_config:
            config = optimization_config.get_algorithm_config(algorithm_name)
        
        # Apply quantization
        return self.quantize_with_algorithm(model, algorithm_name, target, config)
    
    def benchmark_algorithms(
        self,
        model: torch.nn.Module,
        target: TargetHardware,
        algorithms: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Benchmark multiple quantization algorithms on the same model.
        
        Args:
            model: PyTorch model to test
            target: Target hardware platform
            algorithms: List of algorithms to test (None = test all compatible)
            
        Returns:
            Dictionary mapping algorithm names to their stats
        """
        
        if algorithms is None:
            # Test all compatible algorithms
            algorithms = []
            for name, algorithm in self.plugin_algorithms.items():
                if algorithm.can_optimize(model, target):
                    algorithms.append(name)
            
            # Add MVP methods
            algorithms.extend(["dynamic", "static"])
        
        results = {}
        for algorithm_name in algorithms:
            try:
                print(f"\n🧪 Testing {algorithm_name}...")
                
                # Create a proper deep copy of the model
                import copy
                model_copy = copy.deepcopy(model)
                
                _, stats = self.quantize_with_algorithm(model_copy, algorithm_name, target)
                results[algorithm_name] = stats
                print(f"   ✅ Success: {stats['reduction_ratio']:.1f}x size reduction")
            except Exception as e:
                print(f"   ❌ Failed: {e}")
                results[algorithm_name] = {"error": str(e)}
        
        return results
    
    def _get_model_size(self, model: torch.nn.Module) -> int:
        """Calculate model size in bytes."""
        param_size = sum(p.numel() * p.element_size() for p in model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
        return param_size + buffer_size 