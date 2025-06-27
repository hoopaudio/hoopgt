"""
🏀 HoopGT MVP Dynamic Quantization

The original MVP dynamic quantization method, now as a proper algorithm plugin.
"""

import torch
import torch.ao.quantization as quant
from typing import Dict, Any, Optional
from ..hoopgt_base import HoopGTQuantizerBase
from ...types import TargetHardware


class MVPDynamicQuantizer(HoopGTQuantizerBase):
    """
    Original MVP dynamic quantization implementation.
    
    This is your original hardware-optimized dynamic quantization with 
    intelligent backend selection and layer targeting.
    """
    
    algorithm_name = "mvp_dynamic"
    description = "HoopGT MVP dynamic quantization (hardware-optimized)"
    
    # Hardware support - original MVP supported all these
    supported_targets = [
        TargetHardware.APPLE_SILICON,
        TargetHardware.X86_SERVER,
        TargetHardware.ARM_MOBILE,
        TargetHardware.NVIDIA_JETSON,
    ]
    
    # Performance characteristics
    requires_calibration = False
    requires_dataset = False
    
    def __init__(self):
        super().__init__()
        # Original MVP target configurations
        self.target_configs = {
            TargetHardware.APPLE_SILICON.value: {
                "backend": "qnnpack",  # ARM optimized
                "description": "Apple M1/M2/M3/M4 chips",
                "supports_dynamic": True,
                "supports_static": True,
                "supports_qat": True,
            },
            TargetHardware.X86_SERVER.value: {
                "backend": "fbgemm",
                "description": "Intel/AMD x86 servers",
                "supports_dynamic": True,
                "supports_static": True,
                "supports_qat": True,
            },
            TargetHardware.ARM_MOBILE.value: {
                "backend": "qnnpack",
                "description": "ARM mobile/embedded devices",
                "supports_dynamic": True,
                "supports_static": True,
                "supports_qat": False,
            },
            TargetHardware.NVIDIA_JETSON.value: {
                "backend": "qnnpack",  # QNNPACK is often used on Jetson for CPU-based quantization
                "description": "NVIDIA Jetson devices",
                "supports_dynamic": True,
                "supports_static": True,
                "supports_qat": False,
            },
        }
    
    def can_optimize(self, model: torch.nn.Module, target: TargetHardware) -> bool:
        """Check if MVP dynamic quantization can be applied."""
        if target not in self.supported_targets:
            return False
        
        # MVP dynamic works on models with Linear or LSTM layers
        has_linear = any(isinstance(module, torch.nn.Linear) for module in model.modules())
        has_lstm = any(isinstance(module, torch.nn.LSTM) for module in model.modules())
        
        return has_linear or has_lstm
    
    def get_optimization_config(self, model: torch.nn.Module, target: TargetHardware) -> Dict[str, Any]:
        """Get configuration using original MVP logic."""
        target_str = target.value
        
        if target_str not in self.target_configs:
            raise ValueError(f"Unsupported target: {target_str}")

        config = self.target_configs[target_str]
        backend = config["backend"]

        return {
            "backend": backend,
            "dtype": torch.qint8,
            "target_layer_types": {torch.nn.Linear, torch.nn.LSTM},  # Original MVP targets
            "qconfig": None,  # Dynamic doesn't need qconfig
        }
    
    def apply(self, model: torch.nn.Module, target: TargetHardware, config: Optional[Dict[str, Any]] = None) -> torch.nn.Module:
        """Apply MVP dynamic quantization using original implementation."""
        
        # Get configuration (original MVP logic)
        opt_config = self.get_optimization_config(model, target)
        if config:
            opt_config.update(config)
        
        print(f"🏀 Applying HoopGT MVP dynamic quantization for {target.value}")
        print(f"   Backend: {opt_config['backend']} (hardware-optimized)")
        
        # Set backend (original MVP approach)
        torch.backends.quantized.engine = opt_config["backend"]
        
        # Apply dynamic quantization (exactly like original MVP)
        try:
            quantized_model = quant.quantize_dynamic(
                model,
                opt_config["target_layer_types"],  # Original MVP layer targeting
                dtype=opt_config["dtype"],
            )
            
            print(f"✅ MVP dynamic quantization completed using {opt_config['backend']} backend")
            return quantized_model
            
        except Exception as e:
            print(f"❌ MVP dynamic quantization failed: {e}")
            print("   Falling back to original model")
            return model
    
    def get_performance_estimate(self, model: torch.nn.Module, target: TargetHardware) -> Dict[str, float]:
        """Estimate performance using original MVP heuristics."""
        
        # Count Linear and LSTM parameters (original MVP focus)
        total_params = sum(p.numel() for p in model.parameters())
        quantizable_params = 0
        
        for module in model.modules():
            if isinstance(module, (torch.nn.Linear, torch.nn.LSTM)):
                quantizable_params += sum(p.numel() for p in module.parameters())
        
        if total_params == 0:
            quantizable_ratio = 0.0
        else:
            quantizable_ratio = quantizable_params / total_params
        
        # Original MVP performance estimates
        base_speedup = 1.0 + (quantizable_ratio * 0.3)  # Conservative estimate from MVP
        base_size_reduction = 1.0 + (quantizable_ratio * 0.75)  # MVP typically saw ~2x on Linear layers
        
        # Hardware-specific adjustments (original MVP insights)
        if target == TargetHardware.APPLE_SILICON:
            base_speedup *= 1.3  # MVP saw great performance on M-series
        elif target == TargetHardware.X86_SERVER:
            base_speedup *= 1.1  # Good but not as dramatic as M-series
        elif target == TargetHardware.ARM_MOBILE:
            base_speedup *= 0.8  # More conservative on mobile
        
        return {
            "speed_up": min(base_speedup, 2.5),  # MVP cap
            "size_reduction": min(base_size_reduction, 2.0),  # Realistic MVP expectations
            "accuracy_impact": quantizable_ratio * 0.01,  # MVP was quite accurate
        } 