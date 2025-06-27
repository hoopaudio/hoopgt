"""
🏀 HoopGT Legacy Quantization Interface

Backward-compatible wrapper around the original HoopQuantizer API.
This preserves the original MVP interface while using the new plugin architecture.
"""

import torch
from typing import Tuple, Dict, Any, Optional
from .types import TargetHardware
from .algorithms.quantization.mvp_dynamic import MVPDynamicQuantizer
from .algorithms.quantization.mvp_static import MVPStaticQuantizer
from .algorithms.quantization.mvp_selector import MVPAlgorithmSelector


class HoopQuantizer:
    """
    Legacy HoopQuantizer class - maintains exact API compatibility.
    
    This is your original MVP quantizer, now powered by the new plugin architecture.
    All existing code using HoopQuantizer will work unchanged.
    """

    def __init__(self):
        """Initialize with the original MVP algorithms as plugins."""
        self.mvp_dynamic = MVPDynamicQuantizer()
        self.mvp_static = MVPStaticQuantizer()
        self.mvp_selector = MVPAlgorithmSelector()
        
        # Original target configurations (for backward compatibility)
        self.target_configs = self.mvp_selector.target_configs

    def get_quantization_config(
        self, target: str, method: str = "dynamic"
    ) -> Dict[str, Any]:
        """
        Get quantization configuration for target hardware.
        
        Maintains exact compatibility with original MVP API.
        """
        # Convert string target to TargetHardware enum
        target_hw = self._string_to_target_hardware(target)
        
        if method == "dynamic":
            return self.mvp_dynamic.get_optimization_config(torch.nn.Linear(), target_hw)
        elif method == "static":
            return self.mvp_static.get_optimization_config(torch.nn.Linear(), target_hw)
        elif method == "qat":
            if target not in self.target_configs:
                raise ValueError(f"Unsupported target: {target}")
            config = self.target_configs[target]
            if not config.get("supports_qat", False):
                raise ValueError(f"QAT not supported for {target}")
            return {
                "backend": config["backend"],
                "qconfig": torch.ao.quantization.get_default_qat_qconfig(config["backend"]),
                "dtype": torch.qint8,
            }
        else:
            raise ValueError(f"Unknown method: {method}")

    def quantize_dynamic(
        self, model: torch.nn.Module, target: str
    ) -> Tuple[torch.nn.Module, str]:
        """
        Dynamic quantization - best for transformers/LSTM.
        
        Maintains exact compatibility with original MVP API.
        """
        target_hw = self._string_to_target_hardware(target)
        quantized_model = self.mvp_dynamic.apply(model, target_hw)
        return quantized_model, target

    def quantize_static(
        self, model: torch.nn.Module, target: str, calibration_data: Any = None
    ) -> Tuple[torch.nn.Module, str]:
        """
        Static quantization - best for CNNs.
        
        Maintains exact compatibility with original MVP API.
        """
        target_hw = self._string_to_target_hardware(target)
        
        # Pass calibration data through config if provided
        config = None
        if calibration_data is not None:
            config = {"calibration_data": calibration_data}
        
        quantized_model = self.mvp_static.apply(model, target_hw, config)
        return quantized_model, target

    def get_model_size_reduction(
        self, original_model: torch.nn.Module, quantized_model: torch.nn.Module
    ) -> Dict[str, float]:
        """
        Calculate model size reduction from quantization.
        
        Maintains exact compatibility with original MVP API.
        """
        def get_model_size(model):
            param_size = sum(p.numel() * p.element_size() for p in model.parameters())
            buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
            return param_size + buffer_size

        original_size = get_model_size(original_model)
        quantized_size = get_model_size(quantized_model)
        reduction_ratio = original_size / quantized_size if quantized_size > 0 else 1.0

        return {
            "original_size_mb": original_size / (1024 * 1024),
            "quantized_size_mb": quantized_size / (1024 * 1024),
            "reduction_ratio": reduction_ratio,
            "size_savings_percent": (1 - quantized_size / original_size) * 100,
        }

    def get_recommended_method(self, model: torch.nn.Module, target: str) -> str:
        """
        Automatically recommend the best quantization method based on model and target.
        
        Maintains exact compatibility with original MVP API.
        """
        target_hw = self._string_to_target_hardware(target)
        
        # Use the MVP selector's original logic
        mvp_recommendation = self.mvp_selector.get_mvp_recommendation(model, target_hw)
        
        # Convert back to legacy method names
        if mvp_recommendation == "mvp_dynamic":
            return "dynamic"
        elif mvp_recommendation == "mvp_static":
            return "static"
        else:
            return "dynamic"  # Default fallback

    def _string_to_target_hardware(self, target: str) -> TargetHardware:
        """Convert string target to TargetHardware enum."""
        target_mapping = {
            "apple-silicon": TargetHardware.APPLE_SILICON,
            "x86-server": TargetHardware.X86_SERVER,
            "arm-mobile": TargetHardware.ARM_MOBILE,
            "nvidia-jetson": TargetHardware.NVIDIA_JETSON,
        }
        
        if target not in target_mapping:
            # Default fallback
            return TargetHardware.X86_SERVER
        
        return target_mapping[target] 