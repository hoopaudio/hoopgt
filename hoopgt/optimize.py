import torch

from enum import Enum


class TargetHardware(Enum):
    APPLE_SILICON = "apple-silicon"
    NVIDIA_JETSON_ORIN_NANO = "nvidia-jetson-orin-nano"


def optimize_model(
    model_path: str, target_hardware: TargetHardware = TargetHardware.MPS
):
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"🚀 Using {device}")

    try:
        model = torch.load(model_path, map_location=device, weights_only=True)
        print("✅ Loaded with weights_only=True (safe)")
    except Exception:
        model = torch.load(model_path, map_location=device, weights_only=False)
        print("⚠️  Loaded with weights_only=False (full model)")

    model.eval()
    print(f"✅ Model loaded on: {device}")

    return model, device
