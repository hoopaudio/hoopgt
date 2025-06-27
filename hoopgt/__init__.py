"""
HoopGT SDK - Model Optimization Platform

Hardware-optimized AI model optimization with pluggable algorithms.
"""

__version__ = "0.1.0"

# Core engine and optimization
from .engine import OptimizationEngine
from .quantization_engine import QuantizationEngine

# Legacy MVP quantizer (for backward compatibility) - now powered by plugin architecture
from .legacy_quantize import HoopQuantizer

# Configuration
from .config import OptimizationConfig

# Types and enums
from .types import TargetHardware, OptimizationLevel, QuantizationMethod

# Utilities
from .utils.loading import load_model_for_target

# Algorithm base classes (for extending)
from .algorithms.hoopgt_base import HoopGTAlgorithmBase, HoopGTQuantizerBase

__all__ = [
    "__version__",
    
    # Core engines
    "OptimizationEngine",
    "QuantizationEngine",
    
    # Legacy (backward compatibility)
    "HoopQuantizer",
    
    # Configuration
    "OptimizationConfig",
    
    # Types
    "TargetHardware",
    "OptimizationLevel", 
    "QuantizationMethod",
    
    # Utilities
    "load_model_for_target",
    
    # Base classes for extending
    "HoopGTAlgorithmBase",
    "HoopGTQuantizerBase",
] 