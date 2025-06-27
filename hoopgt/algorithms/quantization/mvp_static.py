"""
🏀 HoopGT MVP Static Quantization

The original MVP static quantization method, now as a proper algorithm plugin.
"""

import torch
import torch.ao.quantization as quant
from typing import Dict, Any, Optional
from ..hoopgt_base import HoopGTQuantizerBase
from ...types import TargetHardware


class MVPStaticQuantizer(HoopGTQuantizerBase):
    """
    Original MVP static quantization implementation.
    
    This is your original hardware-optimized static quantization with 
    intelligent calibration and backend selection.
    """
    
    algorithm_name = "mvp_static"
    description = "HoopGT MVP static quantization (hardware-optimized, best for CNNs)"
    
    # Hardware support - original MVP supported all these
    supported_targets = [
        TargetHardware.APPLE_SILICON,
        TargetHardware.X86_SERVER,
        TargetHardware.ARM_MOBILE,
        TargetHardware.NVIDIA_JETSON,
    ]
    
    # Performance characteristics
    requires_calibration = True  # Static quantization benefits from calibration
    requires_dataset = False     # But can work without it
    
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
        """Check if MVP static quantization can be applied."""
        if target not in self.supported_targets:
            return False
        
        # Check if target supports static quantization
        target_config = self.target_configs.get(target.value, {})
        if not target_config.get("supports_static", False):
            return False
        
        # MVP static works best on models with Conv layers
        has_conv = any(
            isinstance(module, (torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d))
            for module in model.modules()
        )
        
        # But can also work on Linear layers
        has_linear = any(isinstance(module, torch.nn.Linear) for module in model.modules())
        
        return has_conv or has_linear
    
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
            "qconfig": quant.get_default_qconfig(backend),
            "calibration_samples": 64,  # Default from original MVP
        }
    
    def apply(self, model: torch.nn.Module, target: TargetHardware, config: Optional[Dict[str, Any]] = None) -> torch.nn.Module:
        """Apply MVP static quantization using original implementation."""
        
        # Get configuration (original MVP logic)
        opt_config = self.get_optimization_config(model, target)
        if config:
            opt_config.update(config)
        
        print(f"🏀 Applying HoopGT MVP static quantization for {target.value}")
        print(f"   Backend: {opt_config['backend']} (hardware-optimized)")
        print(f"   Best for: CNN models with Conv layers")
        
        # Set backend (original MVP approach)
        torch.backends.quantized.engine = opt_config["backend"]
        
        # Apply static quantization (exactly like original MVP)
        try:
            # Prepare model for quantization
            model.qconfig = opt_config["qconfig"]
            prepared = quant.prepare(model, inplace=False)
            
            # Calibration (if data provided)
            calibration_data = config.get("calibration_data") if config else None
            if calibration_data is not None:
                print("   Running calibration...")
                prepared.eval()
                with torch.no_grad():
                    if hasattr(calibration_data, "__iter__"):
                        for batch in calibration_data:
                            prepared(batch)
                    else:
                        prepared(calibration_data)
            else:
                print("   ⚠️  No calibration data provided - using model's current state")

            # Convert to quantized model
            quantized_model = quant.convert(prepared, inplace=False)
            
            print(f"✅ MVP static quantization completed using {opt_config['backend']} backend")
            return quantized_model
            
        except Exception as e:
            print(f"❌ MVP static quantization failed: {e}")
            print("   Falling back to original model")
            return model
    
    def get_performance_estimate(self, model: torch.nn.Module, target: TargetHardware) -> Dict[str, float]:
        """Estimate performance using original MVP heuristics."""
        
        # Count Conv and Linear parameters (MVP static focus)
        total_params = sum(p.numel() for p in model.parameters())
        quantizable_params = 0
        
        # MVP static works particularly well on Conv layers
        conv_params = 0
        linear_params = 0
        
        for module in model.modules():
            if isinstance(module, (torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d)):
                conv_params += sum(p.numel() for p in module.parameters())
                quantizable_params += sum(p.numel() for p in module.parameters())
            elif isinstance(module, torch.nn.Linear):
                linear_params += sum(p.numel() for p in module.parameters())
                quantizable_params += sum(p.numel() for p in module.parameters())
        
        if total_params == 0:
            quantizable_ratio = 0.0
            conv_ratio = 0.0
        else:
            quantizable_ratio = quantizable_params / total_params
            conv_ratio = conv_params / total_params
        
        # Original MVP performance estimates (static typically better than dynamic on CNNs)
        base_speedup = 1.0 + (quantizable_ratio * 0.4)  # Static can be more aggressive
        base_size_reduction = 1.0 + (quantizable_ratio * 0.8)  # Better compression with calibration
        
        # Conv layers benefit more from static quantization
        if conv_ratio > 0.3:  # CNN-heavy model
            base_speedup *= 1.2
            base_size_reduction *= 1.1
        
        # Hardware-specific adjustments (original MVP insights)
        if target == TargetHardware.APPLE_SILICON:
            base_speedup *= 1.2  # M-series handles static quantization well
        elif target == TargetHardware.X86_SERVER:
            base_speedup *= 1.15  # Good static quantization support
        elif target == TargetHardware.ARM_MOBILE:
            base_speedup *= 0.9  # More conservative on mobile
        
        # Accuracy impact is typically higher for static without proper calibration
        accuracy_impact = quantizable_ratio * 0.015  # Slightly higher than dynamic
        
        return {
            "speed_up": min(base_speedup, 3.0),  # Static can be faster than dynamic
            "size_reduction": min(base_size_reduction, 2.5),  # Better compression
            "accuracy_impact": accuracy_impact,  # Depends on calibration quality
        } 