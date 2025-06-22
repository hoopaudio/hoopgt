"""
🏀 HoopGT Core Optimization Engine

This module contains the primary business logic for the optimization pipeline.
"""

import torch
from typing import Dict, Any, Optional, Tuple

from .types import TargetHardware, OptimizationLevel
from .quantize import HoopQuantizer
from .utils.loading import load_model_for_target

class OptimizationEngine:
    """Handles the core model optimization workflow."""

    def quantize(
        self, model: torch.nn.Module, target: TargetHardware
    ) -> Tuple[torch.nn.Module, str, Dict[str, Any]]:
        """
        Applies auto-selected quantization to the model for a given target.

        Args:
            model: The PyTorch model to quantize.
            target: The target hardware for quantization.

        Returns:
            A tuple containing:
            - The quantized model.
            - The name of the quantization method used ('dynamic' or 'static').
            - A dictionary of quantization statistics.
        """
        print("Applying quantization...")
        quantizer = HoopQuantizer()
        # Quantization is a CPU-bound operation
        model_cpu = model.cpu()

        best_method = quantizer.get_recommended_method(model_cpu, target.value)
        print(f"Auto-selected quantization method: {best_method}")

        if best_method == "dynamic":
            quantized_model, _ = quantizer.quantize_dynamic(model_cpu, target.value)
        elif best_method == "static":
            print(
                "Static quantization calibration data not yet implemented. Using model's current state."
            )
            quantized_model, _ = quantizer.quantize_static(
                model_cpu, target=target.value
            )
        else:
            raise ValueError(f"Unsupported quantization method: {best_method}")

        stats = quantizer.get_model_size_reduction(model_cpu, quantized_model)
        return quantized_model, best_method, stats

    def run(
        self,
        model_path: str,
        target: TargetHardware,
        level: OptimizationLevel,
        quantize: bool,
        output_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Runs the full optimization pipeline.

        This method encapsulates the entire process, from model loading to
        quantization and saving, returning a structured dictionary of the results.

        Returns:
            A dictionary containing the results and statistics of the optimization.
        """
        print(f"Initializing optimization engine for target: {target.value}")
        
        model, device = load_model_for_target(model_path, target)

        results = {
            "model_path": model_path,
            "target": target.value,
            "level": level.value,
            "quantization_enabled": quantize,
            "quantization_method": None,
            "quantization_stats": None,
            "output_path": None,
            "final_model": model,
        }

        if quantize:
            quantized_model, best_method, stats = self.quantize(model, target)
            results["quantization_method"] = best_method
            results["quantization_stats"] = stats
            results["final_model"] = quantized_model

        # 3. TODO: Apply other optimizations based on `level` (e.g., pruning, compilation)
        final_model = results["final_model"]

        # 4. Save the final model if an output path is provided
        if output_path:
            print(f"Saving optimized model to: {output_path}")
            torch.save(final_model.state_dict(), output_path)
            results["output_path"] = output_path
        
        print("Engine run completed.")
        return results 