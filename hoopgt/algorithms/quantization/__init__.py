"""
🏀 HoopGT Quantization Algorithms

Hardware-optimized quantization implementations including the original MVP algorithms.
"""

from __future__ import annotations

from ..hoopgt_base import HoopGTQuantizerBase
from .torch_dynamic import TorchDynamicQuantizer
from .mvp_dynamic import MVPDynamicQuantizer
from .mvp_static import MVPStaticQuantizer
from .mvp_selector import MVPAlgorithmSelector

# Registry of available quantization algorithms
QUANTIZATION_ALGORITHMS = {
    "torch_dynamic": TorchDynamicQuantizer,
    "mvp_dynamic": MVPDynamicQuantizer,
    "mvp_static": MVPStaticQuantizer,
}

# Create the MVP selector instance for intelligent algorithm selection
mvp_selector = MVPAlgorithmSelector()

__all__ = [
    "HoopGTQuantizerBase", 
    "TorchDynamicQuantizer",
    "MVPDynamicQuantizer",
    "MVPStaticQuantizer", 
    "MVPAlgorithmSelector",
    "QUANTIZATION_ALGORITHMS",
    "mvp_selector"
]
