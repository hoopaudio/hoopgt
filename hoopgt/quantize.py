"""
🏀 HoopGT Quantization Engine

Optimized quantization for Apple Silicon and other target hardware.
"""

import torch
import torch.ao.quantization as quant
from typing import Tuple, Dict, Any, Optional, Union
from pathlib import Path
import platform
from .types import TargetHardware


class HoopQuantizer:
    """Quantization engine for different target hardware"""

    def __init__(self):
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

    def get_quantization_config(
        self, target: str, method: str = "dynamic"
    ) -> Dict[str, Any]:
        """Get quantization configuration for target hardware"""
        if target not in self.target_configs:
            raise ValueError(f"Unsupported target: {target}")

        config = self.target_configs[target]
        backend = config["backend"]

        if method == "dynamic":
            return {
                "backend": backend,
                "dtype": torch.qint8,
                "qconfig": None,  # Dynamic doesn't need qconfig
            }
        elif method == "static":
            return {
                "backend": backend,
                "qconfig": quant.get_default_qconfig(backend),
                "dtype": torch.qint8,
            }
        elif method == "qat":
            if not config["supports_qat"]:
                raise ValueError(f"QAT not supported for {target}")
            return {
                "backend": backend,
                "qconfig": quant.get_default_qat_qconfig(backend),
                "dtype": torch.qint8,
            }
        else:
            raise ValueError(f"Unknown method: {method}")

    def quantize_dynamic(
        self, model: torch.nn.Module, target: str
    ) -> Tuple[torch.nn.Module, str]:
        """
        Dynamic quantization - best for transformers/LSTM
        """
        print(f"🚀 Dynamic quantization for {target}")
        config = self.get_quantization_config(target, "dynamic")

        # Set backend
        torch.backends.quantized.engine = config["backend"]

        # Dynamic quantization - works on Linear layers
        quantized_model = quant.quantize_dynamic(
            model,
            {torch.nn.Linear, torch.nn.LSTM},  # Target layer types
            dtype=config["dtype"],
        )

        print(f"✅ Dynamic quantization completed using {config['backend']} backend")
        return quantized_model, target

    def quantize_static(
        self, model: torch.nn.Module, target: str, calibration_data: Any = None
    ) -> Tuple[torch.nn.Module, str]:
        """
        Static quantization - best for CNNs
        """
        print(f"🚀 Static quantization for {target}")
        config = self.get_quantization_config(target, "static")

        # Set backend
        torch.backends.quantized.engine = config["backend"]

        # Prepare model for quantization
        model.qconfig = config["qconfig"]
        prepared = quant.prepare(model, inplace=False)

        # Calibration (if data provided)
        if calibration_data is not None:
            print("📊 Running calibration...")
            prepared.eval()
            with torch.no_grad():
                if hasattr(calibration_data, "__iter__"):
                    for batch in calibration_data:
                        prepared(batch)
                else:
                    prepared(calibration_data)

        # Convert to quantized model
        quantized_model = quant.convert(prepared, inplace=False)

        print(f"✅ Static quantization completed using {config['backend']} backend")
        return quantized_model, target

    def get_model_size_reduction(
        self, original_model: torch.nn.Module, quantized_model: torch.nn.Module
    ) -> Dict[str, float]:
        """Calculate model size reduction from quantization"""

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
        Automatically recommend the best quantization method based on model and target

        Args:
            model: PyTorch model to analyze
            target: Target hardware platform

        Returns:
            Recommended quantization method ("dynamic", "static", or "qat")
        """

        # Check if model has LSTM/RNN layers (prefer dynamic)
        has_rnn = any(
            isinstance(module, (torch.nn.LSTM, torch.nn.RNN, torch.nn.GRU))
            for module in model.modules()
        )

        # Check if model has Conv layers (prefer static)
        has_conv = any(
            isinstance(module, (torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d))
            for module in model.modules()
        )

        # Check if model has attention/transformer layers (prefer dynamic)
        has_attention = any(
            "attention" in module.__class__.__name__.lower()
            or "transformer" in module.__class__.__name__.lower()
            for module in model.modules()
        )

        # Get target config
        if target not in self.target_configs:
            target = "x86-server"  # Default fallback

        config = self.target_configs[target]

        # Decision logic based on model architecture
        if has_rnn or has_attention:
            # RNN/Transformer models work best with dynamic quantization
            return "dynamic"
        elif has_conv and config["supports_static"]:
            # CNN models benefit from static quantization
            return "static"
        else:
            # Default to dynamic (most compatible)
            return "dynamic"
