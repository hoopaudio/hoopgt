"""
🏀 HoopGT Functional Tests

Tests for complete workflows and CLI functionality.
"""

import pytest
import subprocess
from pathlib import Path
from typer.testing import CliRunner
import torch
import torch.nn as nn

from hoopgt.cli import app


class SimpleModel(nn.Module):
    """Simple test model for functional testing"""
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 1)
    
    def forward(self, x):
        return self.linear(x)


@pytest.fixture
def test_model_path(tmp_path):
    """Create a temporary test model"""
    model = SimpleModel()
    model_path = tmp_path / "test_model.pt"
    torch.save(model, model_path)
    return str(model_path)


class TestCLIFunctional:
    """Functional tests for HoopGT CLI"""
    
    def test_cli_help(self):
        """Test CLI help command"""
        runner = CliRunner()
        result = runner.invoke(app, ["--help"])
        
        assert result.exit_code == 0
        assert "HoopGT SDK" in result.stdout
    
    def test_optimize_command_with_valid_model(self, test_model_path):
        """Test optimize command with a real model"""
        runner = CliRunner()
        result = runner.invoke(app, [
            "optimize",
            test_model_path,
            "--target", "apple-silicon",
            "--level", "balanced"
        ])
        
        assert result.exit_code == 0
        assert "Initializing optimization engine" in result.stdout
        assert "Engine run completed" in result.stdout
    
    def test_list_command(self):
        """Test list command"""
        runner = CliRunner()
        result = runner.invoke(app, ["list"])
        
        assert result.exit_code == 0
        assert "Available Models" in result.stdout
    
    def test_info_command(self):
        """Test info command"""
        runner = CliRunner()
        result = runner.invoke(app, ["info"])
        
        assert result.exit_code == 0
        assert "System Info" in result.stdout
        assert "0.1.0" in result.stdout
    
    @pytest.mark.skip(reason="Deploy command not implemented")
    def test_deploy_command(self):
        """Test deploy command"""
        runner = CliRunner()
        result = runner.invoke(app, [
            "deploy", 
            "test-model",
            "--port", "8080"
        ])
        
        assert result.exit_code == 0
        assert "HoopGT Deployer" in result.stdout
        assert "test-model" in result.stdout
        assert "8080" in result.stdout


class TestSystemIntegration:
    """Integration tests for system components"""
    
    def test_pytorch_device_detection(self):
        """Test PyTorch device detection works"""
        # Test that we can detect MPS/CPU
        if torch.backends.mps.is_available():
            assert torch.backends.mps.is_available() is True
        else:
            # Should fallback to CPU
            assert torch.cuda.is_available() is False
    
    def test_model_loading_workflow(self, test_model_path):
        """Test complete model loading workflow"""
        # This would test your optimize_model function when implemented
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        
        # Load model
        model = torch.load(test_model_path, map_location=device, weights_only=False)
        model.eval()
        
        # Verify model is loaded correctly
        assert isinstance(model, nn.Module)
        assert next(model.parameters()).device.type in ["mps", "cpu"]
    
    def test_model_optimization_pipeline(self, test_model_path):
        """Test end-to-end optimization pipeline"""
        # TODO: Implement when optimization logic is added
        
        # For now, just test that model can be loaded
        model = torch.load(test_model_path, map_location="cpu", weights_only=False)
        assert model is not None
        
        # Test that model can make predictions
        test_input = torch.randn(1, 10)
        with torch.no_grad():
            output = model(test_input)
            assert output.shape == (1, 1)


class TestCommandLineIntegration:
    """Test CLI integration with subprocess"""
    
    def test_cli_via_subprocess(self):
        """Test CLI works via subprocess (real usage)"""
        result = subprocess.run(
            ["python", "main.py", "--help"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent
        )
        
        assert result.returncode == 0
        assert "HoopGT SDK" in result.stdout
    
    def test_cli_info_subprocess(self):
        """Test info command via subprocess"""
        result = subprocess.run(
            ["python", "main.py", "info"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent
        )
        
        assert result.returncode == 0
        assert "System Info" in result.stdout


if __name__ == "__main__":
    pytest.main([__file__]) 