"""
HoopGT Model Loading Utilities
"""

import torch
from typing import Tuple, Union
from ..types import TargetHardware

def load_model_for_target(
    model_path: str, target_hardware: Union[str, TargetHardware]
) -> Tuple[torch.nn.Module, str]:
    """
    Loads a model from a path and prepares it for a specific hardware target.

    This function determines the correct device (e.g., 'mps' for Apple Silicon)
    and handles the safe loading of the model.

    Args:
        model_path: Path to the model file.
        target_hardware: The hardware target (enum or string).

    Returns:
        A tuple containing:
        - The loaded PyTorch model.
        - The device string it was loaded onto (e.g., 'mps', 'cpu').
    """
    if isinstance(target_hardware, str):
        target_hardware = TargetHardware.from_string(target_hardware)

    if (
        target_hardware == TargetHardware.APPLE_SILICON
        and torch.backends.mps.is_available()
    ):
        device = "mps"
    else:
        device = "cpu"

    print(f"Preparing to load model for target: {target_hardware.value} on device: {device}")

    try:
        model = torch.load(model_path, map_location=device, weights_only=True)
        print("Model loaded safely (weights_only=True).")
    except Exception:
        model = torch.load(model_path, map_location=device, weights_only=False)
        print("Could not load weights only. Loaded full model (less safe).")

    model.eval()
    print(f"Model '{model_path}' loaded and in evaluation mode.")

    return model, device 