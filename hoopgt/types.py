"""
🏀 HoopGT Types

Type definitions for HoopGT SDK.
"""

from enum import Enum


class TargetHardware(Enum):
    """Supported target hardware platforms"""
    APPLE_SILICON = "apple-silicon"
    X86_SERVER = "x86-server"
    ARM_MOBILE = "arm-mobile"
    NVIDIA_JETSON = "nvidia-jetson"
    
    def __str__(self) -> str:
        return self.value
    
    @classmethod
    def from_string(cls, value: str) -> "TargetHardware":
        """Convert string to TargetHardware enum"""
        for target in cls:
            if target.value == value:
                return target
        raise ValueError(f"Invalid target hardware: {value}. Supported: {[t.value for t in cls]}")


class OptimizationLevel(Enum):
    """Optimization intensity levels"""
    LIGHT = "light"
    BALANCED = "balanced"
    AGGRESSIVE = "aggressive"
    
    def __str__(self) -> str:
        return self.value
    
    @classmethod
    def from_string(cls, value: str) -> "OptimizationLevel":
        """Convert string to OptimizationLevel enum"""
        for level in cls:
            if level.value == value:
                return level
        raise ValueError(f"Invalid optimization level: {value}. Supported: {[l.value for l in cls]}")


class QuantizationMethod(Enum):
    """Quantization methods"""
    DYNAMIC = "dynamic"
    STATIC = "static"
    QAT = "qat"  # Quantization Aware Training
    
    def __str__(self) -> str:
        return self.value
    
    @classmethod
    def from_string(cls, value: str) -> "QuantizationMethod":
        """Convert string to QuantizationMethod enum"""
        for method in cls:
            if method.value == value:
                return method
        raise ValueError(f"Invalid quantization method: {value}. Supported: {[m.value for m in cls]}")


# Type aliases for convenience
Target = TargetHardware
Level = OptimizationLevel
QuantMethod = QuantizationMethod 