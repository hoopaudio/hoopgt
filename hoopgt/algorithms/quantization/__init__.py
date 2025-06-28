"""
🏀 HoopGT Quantization Algorithms

Clean, production-ready quantization implementations for edge AI.
"""

from __future__ import annotations

from ..hoopgt_base import HoopGTQuantizerBase
from .dynamic_quantization import DynamicQuantizer
from .static_quantization import StaticQuantizer

# Registry of available quantization algorithms
QUANTIZATION_ALGORITHMS = {
    "dynamic": DynamicQuantizer,
    "static": StaticQuantizer,
}

__all__ = [
    "HoopGTQuantizerBase", 
    "DynamicQuantizer",
    "StaticQuantizer",
    "QUANTIZATION_ALGORITHMS",
]
