"""
🏀 HoopGT Algorithm Base Classes

Simplified base classes for HoopGT optimization algorithms.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import torch
from ..types import TargetHardware


class HoopGTAlgorithmBase(ABC):
    """Base class for all HoopGT optimization algorithms."""
    
    # Subclasses must define these
    algorithm_name: str
    algorithm_group: str  # e.g., "quantizer", "pruner", "compiler"
    description: str
    
    # Hardware support
    supported_targets: List[TargetHardware] = []
    supported_devices: List[str] = ["cpu", "cuda", "mps"]
    
    # Requirements
    requires_calibration: bool = False
    requires_dataset: bool = False
    
    def __init__(self):
        """Initialize the algorithm."""
        self.validate_configuration()
    
    def validate_configuration(self) -> None:
        """Validate algorithm configuration."""
        if not hasattr(self, 'algorithm_name'):
            raise ValueError(f"{self.__class__.__name__} must define algorithm_name")
        if not hasattr(self, 'algorithm_group'):
            raise ValueError(f"{self.__class__.__name__} must define algorithm_group")
    
    @abstractmethod
    def can_optimize(self, model: torch.nn.Module, target: TargetHardware) -> bool:
        """
        Check if this algorithm can optimize the given model for the target hardware.
        
        Args:
            model: PyTorch model to check
            target: Target hardware platform
            
        Returns:
            True if optimization is supported
        """
        pass
    
    @abstractmethod
    def get_optimization_config(self, model: torch.nn.Module, target: TargetHardware) -> Dict[str, Any]:
        """
        Get the optimal configuration for this model and target.
        
        Args:
            model: PyTorch model
            target: Target hardware platform
            
        Returns:
            Configuration dictionary
        """
        pass
    
    @abstractmethod
    def apply(self, model: torch.nn.Module, target: TargetHardware, config: Optional[Dict[str, Any]] = None) -> torch.nn.Module:
        """
        Apply the optimization to the model.
        
        Args:
            model: PyTorch model to optimize
            target: Target hardware platform
            config: Optional configuration override
            
        Returns:
            Optimized model
        """
        pass
    
    def get_performance_estimate(self, model: torch.nn.Module, target: TargetHardware) -> Dict[str, float]:
        """
        Estimate performance improvements for this optimization.
        
        Args:
            model: PyTorch model
            target: Target hardware platform
            
        Returns:
            Dictionary with estimated improvements (speed_up, size_reduction, etc.)
        """
        return {
            "speed_up": 1.0,      # Estimated speedup multiplier
            "size_reduction": 1.0, # Estimated size reduction ratio
            "accuracy_impact": 0.0, # Estimated accuracy loss (0-1)
        }


class HoopGTQuantizerBase(HoopGTAlgorithmBase):
    """Base class specifically for quantization algorithms."""
    
    algorithm_group: str = "quantizer"
    
    # Common quantization properties
    supported_dtypes: List[torch.dtype] = [torch.qint8, torch.quint8]
    default_dtype: torch.dtype = torch.qint8
    
    def get_quantizable_modules(self, model: torch.nn.Module, target: TargetHardware) -> set:
        """
        Get the set of module types that should be quantized for the target.
        
        Args:
            model: PyTorch model
            target: Target hardware platform
            
        Returns:
            Set of module types to quantize
        """
        # Default implementation - can be overridden
        base_modules = {torch.nn.Linear}
        
        if target == TargetHardware.APPLE_SILICON:
            # M-series chips handle conv layers well
            base_modules.update({torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d})
        elif target == TargetHardware.X86_SERVER:
            # Full support on x86
            base_modules.update({
                torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d,
                torch.nn.LSTM, torch.nn.GRU
            })
        elif target == TargetHardware.ARM_MOBILE:
            # Conservative for mobile
            base_modules.update({torch.nn.Conv2d})
            
        return base_modules
    
    def get_backend_for_target(self, target: TargetHardware) -> str:
        """Get the optimal quantization backend for target hardware."""
        backend_map = {
            TargetHardware.APPLE_SILICON: "qnnpack",
            TargetHardware.X86_SERVER: "fbgemm", 
            TargetHardware.ARM_MOBILE: "qnnpack",
            TargetHardware.NVIDIA_JETSON: "qnnpack",
        }
        return backend_map.get(target, "fbgemm") 