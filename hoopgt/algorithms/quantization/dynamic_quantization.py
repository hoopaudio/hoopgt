"""
🏀 HoopGT Dynamic Quantization

Production-ready dynamic quantization for edge devices.
Optimized for LLMs/VLMs on Apple Silicon, Jetson, and ARM devices.
"""

import torch
import torch.ao.quantization as quant
from typing import Dict, Any, Optional, List, Union
from ..hoopgt_base import HoopGTQuantizerBase
from ...types import TargetHardware


class DynamicQuantizer(HoopGTQuantizerBase):
    """
    Production dynamic quantization for edge AI.
    
    Perfect for:
    - LLMs (Llama, GPT, etc.)
    - VLMs (LLaVA, CLIP, etc.) 
    - Transformer models
    - RNN/LSTM models
    """
    
    algorithm_name = "dynamic"
    description = "Dynamic quantization - best for LLMs/VLMs and transformers"
    
    supported_targets = [
        TargetHardware.APPLE_SILICON,
        TargetHardware.X86_SERVER,
        TargetHardware.ARM_MOBILE,
        TargetHardware.NVIDIA_JETSON,
    ]
    
    requires_calibration = False
    requires_dataset = False
    
    def __init__(self):
        super().__init__()
        self.backend_configs = {
            TargetHardware.APPLE_SILICON: {
                "backend": "qnnpack",  # ARM optimized
                "description": "Apple M1/M2/M3/M4 optimization",
                "dtype": torch.qint8,
                "performance_multiplier": 1.3,  # M-series handles quantization exceptionally well
            },
            TargetHardware.X86_SERVER: {
                "backend": "fbgemm",  # Intel/AMD optimized
                "description": "Intel/AMD x86 optimization", 
                "dtype": torch.qint8,
                "performance_multiplier": 1.1,
            },
            TargetHardware.ARM_MOBILE: {
                "backend": "qnnpack",
                "description": "ARM mobile/embedded optimization",
                "dtype": torch.qint8,
                "performance_multiplier": 0.9,  # Conservative for battery life
            },
            TargetHardware.NVIDIA_JETSON: {
                "backend": "qnnpack",  # CPU quantization on Jetson
                "description": "NVIDIA Jetson optimization",
                "dtype": torch.qint8,
                "performance_multiplier": 1.0,
            },
        }
    
    def can_optimize(self, model: torch.nn.Module, target: TargetHardware) -> bool:
        """Check if model can be dynamically quantized."""
        if target not in self.supported_targets:
            return False
        
        # Dynamic quantization works on models with these layer types
        quantizable_layers = {
            torch.nn.Linear,
            torch.nn.LSTM, 
            torch.nn.GRU,
            torch.nn.RNN,
        }
        
        has_quantizable = any(
            isinstance(module, tuple(quantizable_layers))
            for module in model.modules()
        )
        
        return has_quantizable
    
    def _analyze_model_architecture(self, model: torch.nn.Module) -> Dict[str, Any]:
        """Analyze model to understand quantization potential."""
        
        total_params = sum(p.numel() for p in model.parameters())
        
        # Count parameters by layer type
        linear_params = sum(
            sum(p.numel() for p in module.parameters())
            for module in model.modules()
            if isinstance(module, torch.nn.Linear)
        )
        
        lstm_params = sum(
            sum(p.numel() for p in module.parameters())
            for module in model.modules()
            if isinstance(module, (torch.nn.LSTM, torch.nn.GRU, torch.nn.RNN))
        )
        
        # Check for transformer/attention patterns
        has_attention = any(
            "attention" in module.__class__.__name__.lower() or
            "transformer" in module.__class__.__name__.lower()
            for module in model.modules()
        )
        
        quantizable_params = linear_params + lstm_params
        quantizable_ratio = quantizable_params / total_params if total_params > 0 else 0.0
        
        return {
            "total_params": total_params,
            "linear_params": linear_params,
            "lstm_params": lstm_params,
            "quantizable_params": quantizable_params,
            "quantizable_ratio": quantizable_ratio,
            "has_attention": has_attention,
            "linear_ratio": linear_params / total_params if total_params > 0 else 0.0,
        }
    
    def get_optimization_config(self, model: torch.nn.Module, target: TargetHardware) -> Dict[str, Any]:
        """Get hardware-optimized configuration."""
        
        if target not in self.backend_configs:
            raise ValueError(f"Unsupported target hardware: {target}")
        
        backend_config = self.backend_configs[target]
        arch_info = self._analyze_model_architecture(model)
        
        # Dynamic quantization targets these layer types
        target_layers = {torch.nn.Linear}
        
        # Add LSTM layers if present (good for RNN models)
        if arch_info["lstm_params"] > 0:
            target_layers.update({torch.nn.LSTM, torch.nn.GRU, torch.nn.RNN})
        
        return {
            "backend": backend_config["backend"],
            "dtype": backend_config["dtype"], 
            "target_layers": target_layers,
            "performance_multiplier": backend_config["performance_multiplier"],
            "description": backend_config["description"],
        }
    
    def apply(self, model: torch.nn.Module, target: TargetHardware, config: Optional[Dict[str, Any]] = None) -> torch.nn.Module:
        """Apply dynamic quantization."""
        
        opt_config = self.get_optimization_config(model, target)
        if config:
            opt_config.update(config)
        
        print(f"🏀 Applying dynamic quantization for {target.value}")
        print(f"   Backend: {opt_config['backend']} ({opt_config['description']})")
        print(f"   Target layers: {[cls.__name__ for cls in opt_config['target_layers']]}")
        
        # Set hardware-optimized backend
        torch.backends.quantized.engine = opt_config["backend"]
        
        try:
            # Apply dynamic quantization
            quantized_model = quant.quantize_dynamic(
                model,
                opt_config["target_layers"],
                dtype=opt_config["dtype"],
            )
            
            print(f"✅ Dynamic quantization completed successfully")
            return quantized_model
            
        except Exception as e:
            print(f"❌ Dynamic quantization failed: {e}")
            print("   Returning original model")
            return model
    
    def get_performance_estimate(self, model: torch.nn.Module, target: TargetHardware) -> Dict[str, float]:
        """Estimate performance gains from dynamic quantization."""
        
        arch_info = self._analyze_model_architecture(model)
        config = self.get_optimization_config(model, target)
        
        # Base performance estimates
        quantizable_ratio = arch_info["quantizable_ratio"]
        
        # Dynamic quantization typically provides:
        # - 1.3-2x inference speedup on quantizable layers
        # - ~2x memory reduction on quantizable layers  
        # - Minimal accuracy loss (usually <1%)
        
        base_speedup = 1.0 + (quantizable_ratio * 0.4)  # Conservative estimate
        base_memory_reduction = 1.0 + (quantizable_ratio * 0.7)  # Memory savings
        
        # Apply hardware-specific multiplier
        perf_multiplier = config["performance_multiplier"]
        hardware_speedup = base_speedup * perf_multiplier
        
        # Transformer models often see better gains
        if arch_info["has_attention"] and arch_info["linear_ratio"] > 0.6:
            hardware_speedup *= 1.2  # Transformers benefit significantly
        
        # Accuracy impact is typically minimal for dynamic quantization
        accuracy_impact = min(quantizable_ratio * 0.008, 0.02)  # Cap at 2%
        
        return {
            "speed_up": min(hardware_speedup, 2.5),  # Realistic cap
            "memory_reduction": min(base_memory_reduction, 2.0),
            "accuracy_impact": accuracy_impact,
            "quantizable_ratio": quantizable_ratio,
        }
    
    def benchmark(self, model: torch.nn.Module, target: TargetHardware, input_shape: tuple, runs: int = 10) -> Dict[str, float]:
        """Benchmark quantized vs original model performance."""
        
        # Create sample input
        sample_input = torch.randn(input_shape)
        
        # Quantize model
        quantized_model = self.apply(model, target)
        
        # Benchmark original model
        model.eval()
        with torch.no_grad():
            start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
            
            if start_time:
                start_time.record()
                for _ in range(runs):
                    _ = model(sample_input)
                end_time = torch.cuda.Event(enable_timing=True)
                end_time.record()
                torch.cuda.synchronize()
                original_time = start_time.elapsed_time(end_time) / runs
            else:
                import time
                start = time.time()
                for _ in range(runs):
                    _ = model(sample_input)
                original_time = (time.time() - start) / runs * 1000  # Convert to ms
        
        # Benchmark quantized model  
        quantized_model.eval()
        with torch.no_grad():
            if start_time:
                start_time = torch.cuda.Event(enable_timing=True)
                start_time.record()
                for _ in range(runs):
                    _ = quantized_model(sample_input)
                end_time = torch.cuda.Event(enable_timing=True)
                end_time.record()
                torch.cuda.synchronize()
                quantized_time = start_time.elapsed_time(end_time) / runs
            else:
                start = time.time()
                for _ in range(runs):
                    _ = quantized_model(sample_input)
                quantized_time = (time.time() - start) / runs * 1000
        
        speedup = original_time / quantized_time if quantized_time > 0 else 1.0
        
        return {
            "original_time_ms": original_time,
            "quantized_time_ms": quantized_time,
            "speedup": speedup,
            "runs": runs,
        } 