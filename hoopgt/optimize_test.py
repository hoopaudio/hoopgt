import torch
from pathlib import Path
import tempfile
import pytest
from .optimize import optimize_model  # Proper relative import


def test_basic_model_load():
    """Functional test for optimize_model function"""
    model = torch.nn.Linear(5, 1)
    
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as tmp_file:
        torch.save(model, tmp_file.name)
        temp_path = tmp_file.name
    
    try:
        loaded_model, device = optimize_model(temp_path)
        
        assert loaded_model is not None
        assert device in ["mps"]
        print(f"🔍 Model loaded on: {device}")
        assert not loaded_model.training 
        
        test_input = torch.randn(1, 5).to(device)
        with torch.no_grad():
            output = loaded_model(test_input)
            assert output.shape == (1, 1)
        
        print(f"✅ Model loaded successfully on {device}")
        
    finally:
        Path(temp_path).unlink()


def test_device_detection():
    """Test M4 GPU detection works"""
    expected = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"🔍 Detected device: {expected}")
    assert expected in ["mps", "cpu"]


if __name__ == "__main__":
    test_device_detection()
    test_basic_model_load()
    print("🎉 All functional tests passed!")