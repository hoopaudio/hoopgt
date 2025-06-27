"""
🏀 HoopGT Optimization Configuration

Simple configuration management without heavy dependencies.
"""

from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, field
from ..types import TargetHardware, OptimizationLevel, QuantizationMethod


@dataclass
class OptimizationConfig:
    """Simple configuration for optimization algorithms."""
    
    # Target configuration
    target_hardware: TargetHardware
    optimization_level: OptimizationLevel = OptimizationLevel.BALANCED
    
    # Quantization settings
    quantization_enabled: bool = True
    quantization_method: Optional[QuantizationMethod] = None  # Auto-select if None
    quantization_dtype: str = "qint8"  # "qint8", "quint8", "qint4"
    
    # Model-specific settings
    calibration_samples: int = 64
    batch_size: int = 1
    
    # Algorithm-specific overrides
    algorithm_configs: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Device settings
    device: Optional[str] = None  # Auto-detect if None
    
    def __post_init__(self):
        """Validate and normalize configuration after initialization."""
        # Convert string to enum if needed
        if isinstance(self.target_hardware, str):
            self.target_hardware = TargetHardware.from_string(self.target_hardware)
        
        if isinstance(self.optimization_level, str):
            self.optimization_level = OptimizationLevel.from_string(self.optimization_level)
            
        if isinstance(self.quantization_method, str):
            self.quantization_method = QuantizationMethod.from_string(self.quantization_method)
    
    def get_algorithm_config(self, algorithm_name: str) -> Dict[str, Any]:
        """Get configuration for a specific algorithm."""
        return self.algorithm_configs.get(algorithm_name, {})
    
    def set_algorithm_config(self, algorithm_name: str, config: Dict[str, Any]) -> None:
        """Set configuration for a specific algorithm."""
        self.algorithm_configs[algorithm_name] = config
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "target_hardware": self.target_hardware.value,
            "optimization_level": self.optimization_level.value,
            "quantization_enabled": self.quantization_enabled,
            "quantization_method": self.quantization_method.value if self.quantization_method else None,
            "quantization_dtype": self.quantization_dtype,
            "calibration_samples": self.calibration_samples,
            "batch_size": self.batch_size,
            "algorithm_configs": self.algorithm_configs,
            "device": self.device,
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "OptimizationConfig":
        """Create from dictionary representation."""
        # Handle enum conversion
        if "target_hardware" in config_dict:
            config_dict["target_hardware"] = TargetHardware.from_string(config_dict["target_hardware"])
        if "optimization_level" in config_dict:
            config_dict["optimization_level"] = OptimizationLevel.from_string(config_dict["optimization_level"])
        if config_dict.get("quantization_method"):
            config_dict["quantization_method"] = QuantizationMethod.from_string(config_dict["quantization_method"])
            
        return cls(**config_dict)
    
    @classmethod
    def for_apple_silicon(cls, optimization_level: OptimizationLevel = OptimizationLevel.BALANCED) -> "OptimizationConfig":
        """Create optimized config for Apple Silicon (M1/M2/M3/M4)."""
        return cls(
            target_hardware=TargetHardware.APPLE_SILICON,
            optimization_level=optimization_level,
            quantization_enabled=True,
            device="mps"
        )
    
    @classmethod
    def for_x86_server(cls, optimization_level: OptimizationLevel = OptimizationLevel.BALANCED) -> "OptimizationConfig":
        """Create optimized config for x86 servers."""
        return cls(
            target_hardware=TargetHardware.X86_SERVER,
            optimization_level=optimization_level,
            quantization_enabled=True,
            device="cpu"
        )
    
    @classmethod 
    def for_nvidia_jetson(cls, optimization_level: OptimizationLevel = OptimizationLevel.BALANCED) -> "OptimizationConfig":
        """Create optimized config for NVIDIA Jetson devices."""
        return cls(
            target_hardware=TargetHardware.NVIDIA_JETSON,
            optimization_level=optimization_level,
            quantization_enabled=True,
            device="cuda"
        ) 