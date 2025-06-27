"""
🏀 HoopGT PyTorch Dynamic Quantization

Hardware-optimized dynamic quantization using PyTorch's built-in quantization.
"""

import torch
import torch.ao.quantization as quant
from typing import Dict, Any, Optional
from ..hoopgt_base import HoopGTQuantizerBase
from ...types import TargetHardware
from ...config.optimization_config import OptimizationConfig


class TorchDynamicQuantizer(HoopGTQuantizerBase):
    """
    PyTorch dynamic quantization optimized for different hardware targets.
    
    Dynamic quantization converts weights to lower precision (int8) at runtime,
    providing good speedup for RNN/Transformer models without calibration data.
    """
    
    algorithm_name = "torch_dynamic"
    description = "PyTorch built-in dynamic quantization"
    
    # Hardware support
    supported_targets = [
        TargetHardware.APPLE_SILICON,
        TargetHardware.X86_SERVER,
        TargetHardware.ARM_MOBILE,
        TargetHardware.NVIDIA_JETSON,
    ]
    
    # Performance characteristics
    requires_calibration = False
    requires_dataset = False
    
    def can_optimize(self, model: torch.nn.Module, target: TargetHardware) -> bool:
        """Check if dynamic quantization can be applied to this model."""
        if target not in self.supported_targets:
            return False
        
        # Check if model has quantizable layers
        quantizable_modules = self.get_quantizable_modules(model, target)
        has_quantizable = any(
            isinstance(module, tuple(quantizable_modules))
            for module in model.modules()
        )
        
        return has_quantizable
    
    def get_optimization_config(self, model: torch.nn.Module, target: TargetHardware) -> Dict[str, Any]:
        """Get optimal configuration for this model and target."""
        config = {
            "backend": self.get_backend_for_target(target),
            "dtype": torch.qint8,  # Default to int8
            "modules_to_quantize": list(self.get_quantizable_modules(model, target)),
        }
        
        # Target-specific optimizations
        if target == TargetHardware.APPLE_SILICON:
            # M-series chips prefer certain configurations
            config.update({
                "preserve_embeddings": True,  # Better accuracy on M-series
                "qconfig_spec": None,  # Use default qconfig
            })
        elif target == TargetHardware.X86_SERVER:
            # x86 servers can handle more aggressive quantization
            config.update({
                "preserve_embeddings": False,
                "include_lstm": True,  # x86 handles LSTM quantization well
            })
        elif target == TargetHardware.ARM_MOBILE:
            # Conservative settings for mobile
            config.update({
                "preserve_embeddings": True,
                "conservative_quantization": True,
            })
        
        return config
    
    def apply(self, model: torch.nn.Module, target: TargetHardware, config: Optional[Dict[str, Any]] = None) -> torch.nn.Module:
        """Apply dynamic quantization to the model."""
        
        # Get default config and merge with user config
        opt_config = self.get_optimization_config(model, target)
        if config:
            opt_config.update(config)
        
        print(f"🔧 Applying PyTorch dynamic quantization for {target.value}")
        print(f"   Backend: {opt_config['backend']}")
        print(f"   Data type: {opt_config['dtype']}")
        
        # Set quantization backend
        torch.backends.quantized.engine = opt_config["backend"]
        
        # Prepare modules to quantize
        modules_to_quantize = set()
        for module_type in opt_config["modules_to_quantize"]:
            if isinstance(module_type, str):
                # Handle string module names
                if hasattr(torch.nn, module_type):
                    modules_to_quantize.add(getattr(torch.nn, module_type))
            else:
                # Handle already instantiated types
                modules_to_quantize.add(module_type)
        
        # Handle device-specific constraints
        device = next(model.parameters()).device if model.parameters() else torch.device("cpu")
        if device.type == "cuda" and target != TargetHardware.NVIDIA_JETSON:
            # Remove Linear layers for CUDA (serialization issues)
            modules_to_quantize.discard(torch.nn.Linear)
            print("   ⚠️  Excluding Linear layers on CUDA (serialization constraint)")
        
        # Apply quantization
        try:
            quantized_model = quant.quantize_dynamic(
                model,
                qconfig_spec=modules_to_quantize,
                dtype=opt_config["dtype"],
                mapping=None,
                inplace=False  # Always create a copy
            )
            
            print(f"✅ Dynamic quantization completed successfully")
            return quantized_model
            
        except Exception as e:
            print(f"❌ Dynamic quantization failed: {e}")
            print("   Falling back to original model")
            return model
    
    def get_performance_estimate(self, model: torch.nn.Module, target: TargetHardware) -> Dict[str, float]:
        """Estimate performance improvements for dynamic quantization."""
        
        # Count quantizable parameters
        total_params = sum(p.numel() for p in model.parameters())
        quantizable_params = 0
        
        quantizable_modules = self.get_quantizable_modules(model, target)
        for module in model.modules():
            if isinstance(module, tuple(quantizable_modules)):
                quantizable_params += sum(p.numel() for p in module.parameters())
        
        if total_params == 0:
            quantizable_ratio = 0.0
        else:
            quantizable_ratio = quantizable_params / total_params
        
        # Estimate based on quantizable ratio and target hardware
        base_speedup = 1.0 + (quantizable_ratio * 0.5)  # 50% speedup for quantized layers
        base_size_reduction = 1.0 + (quantizable_ratio * 1.0)  # 2x reduction for quantized layers
        
        # Target-specific adjustments
        if target == TargetHardware.APPLE_SILICON:
            base_speedup *= 1.2  # M-series chips benefit more
        elif target == TargetHardware.X86_SERVER:
            base_speedup *= 1.1  # Good x86 optimization
        elif target == TargetHardware.ARM_MOBILE:
            base_speedup *= 0.9  # Conservative on mobile
        
        return {
            "speed_up": min(base_speedup, 3.0),  # Cap at 3x
            "size_reduction": min(base_size_reduction, 2.0),  # Cap at 2x
            "accuracy_impact": quantizable_ratio * 0.02,  # ~2% loss per fully quantized model
        }
    
    def get_quantizable_modules(self, model: torch.nn.Module, target: TargetHardware) -> set:
        """Get hardware-optimized set of modules to quantize."""
        # Use the base class method but add torch dynamic specific optimizations
        base_modules = super().get_quantizable_modules(model, target)
        
        # Dynamic quantization works particularly well with Linear and LSTM
        if target in [TargetHardware.APPLE_SILICON, TargetHardware.X86_SERVER]:
            base_modules.add(torch.nn.LSTM)
            base_modules.add(torch.nn.GRU)
        
        return base_modules
