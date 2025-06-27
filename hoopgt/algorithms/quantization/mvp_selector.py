"""
🏀 HoopGT MVP Algorithm Selector

The original MVP's intelligent algorithm selection logic, now as a reusable component.
"""

import torch
from typing import Dict, Any, List
from ...types import TargetHardware


class MVPAlgorithmSelector:
    """
    Encapsulates the original MVP's intelligent algorithm selection logic.
    
    This is your brilliant `get_recommended_method` function, now as a 
    reusable component that can recommend any quantization algorithm.
    """
    
    def __init__(self):
        """Initialize with original MVP target configurations."""
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
    
    def analyze_model_architecture(self, model: torch.nn.Module) -> Dict[str, Any]:
        """
        Analyze model architecture to understand its characteristics.
        
        This is the core logic from your original MVP's get_recommended_method.
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
        
        # Check for Linear layers
        has_linear = any(
            isinstance(module, torch.nn.Linear)
            for module in model.modules()
        )
        
        # Count parameters by type for better decision making
        total_params = sum(p.numel() for p in model.parameters())
        conv_params = sum(
            sum(p.numel() for p in module.parameters())
            for module in model.modules()
            if isinstance(module, (torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d))
        )
        linear_params = sum(
            sum(p.numel() for p in module.parameters())
            for module in model.modules()
            if isinstance(module, torch.nn.Linear)
        )
        rnn_params = sum(
            sum(p.numel() for p in module.parameters())
            for module in model.modules()
            if isinstance(module, (torch.nn.LSTM, torch.nn.RNN, torch.nn.GRU))
        )
        
        return {
            "has_rnn": has_rnn,
            "has_conv": has_conv,
            "has_attention": has_attention,
            "has_linear": has_linear,
            "total_params": total_params,
            "conv_params": conv_params,
            "linear_params": linear_params,
            "rnn_params": rnn_params,
            "conv_ratio": conv_params / total_params if total_params > 0 else 0.0,
            "linear_ratio": linear_params / total_params if total_params > 0 else 0.0,
            "rnn_ratio": rnn_params / total_params if total_params > 0 else 0.0,
        }
    
    def get_mvp_recommendation(self, model: torch.nn.Module, target: TargetHardware) -> str:
        """
        Get the original MVP recommendation for quantization method.
        
        This is your exact original get_recommended_method logic.
        """
        
        # Analyze model architecture
        arch_info = self.analyze_model_architecture(model)
        
        # Get target config
        target_str = target.value if isinstance(target, TargetHardware) else target
        if target_str not in self.target_configs:
            target_str = "x86-server"  # Default fallback

        config = self.target_configs[target_str]

        # Decision logic based on model architecture (original MVP logic)
        if arch_info["has_rnn"] or arch_info["has_attention"]:
            # RNN/Transformer models work best with dynamic quantization
            return "mvp_dynamic"
        elif arch_info["has_conv"] and config["supports_static"]:
            # CNN models benefit from static quantization
            return "mvp_static"
        else:
            # Default to dynamic (most compatible)
            return "mvp_dynamic"
    
    def get_expanded_recommendation(
        self, 
        model: torch.nn.Module, 
        target: TargetHardware,
        available_algorithms: List[str]
    ) -> str:
        """
        Get recommendation considering both MVP algorithms and new plugin algorithms.
        
        This extends your original logic to work with the new plugin architecture.
        """
        
        # First get the MVP recommendation
        mvp_recommendation = self.get_mvp_recommendation(model, target)
        
        # If MVP algorithm is available, prefer it (proven and tested)
        if mvp_recommendation in available_algorithms:
            return mvp_recommendation
        
        # Otherwise, fall back to newer plugin algorithms
        arch_info = self.analyze_model_architecture(model)
        
        # Look for suitable plugin algorithms
        if "torch_dynamic" in available_algorithms:
            if arch_info["has_rnn"] or arch_info["has_attention"] or arch_info["has_linear"]:
                return "torch_dynamic"
        
        # If nothing else, try the first available algorithm
        if available_algorithms:
            return available_algorithms[0]
        
        # Fallback to MVP dynamic if somehow no algorithms are available
        return "mvp_dynamic"
    
    def explain_recommendation(self, model: torch.nn.Module, target: TargetHardware) -> Dict[str, Any]:
        """
        Explain why a particular algorithm was recommended.
        
        This provides transparency into the decision-making process.
        """
        
        arch_info = self.analyze_model_architecture(model)
        recommendation = self.get_mvp_recommendation(model, target)
        
        explanation = {
            "recommended_algorithm": recommendation,
            "target_hardware": target.value,
            "model_characteristics": arch_info,
            "reasoning": [],
        }
        
        # Build reasoning based on original MVP logic
        if arch_info["has_rnn"]:
            explanation["reasoning"].append("Model has RNN/LSTM layers - dynamic quantization works best")
        
        if arch_info["has_attention"]:
            explanation["reasoning"].append("Model has attention/transformer layers - dynamic quantization recommended")
        
        if arch_info["has_conv"] and arch_info["conv_ratio"] > 0.3:
            explanation["reasoning"].append("Model is CNN-heavy - static quantization can provide better compression")
        
        if arch_info["has_linear"] and arch_info["linear_ratio"] > 0.5:
            explanation["reasoning"].append("Model has many Linear layers - good candidate for quantization")
        
        # Hardware-specific insights
        target_config = self.target_configs.get(target.value, {})
        if target == TargetHardware.APPLE_SILICON:
            explanation["reasoning"].append("Apple M-series chips: excellent quantization support")
        elif target == TargetHardware.X86_SERVER:
            explanation["reasoning"].append("x86 servers: full quantization method support")
        elif target == TargetHardware.ARM_MOBILE:
            explanation["reasoning"].append("ARM mobile: conservative quantization for battery efficiency")
        
        return explanation 