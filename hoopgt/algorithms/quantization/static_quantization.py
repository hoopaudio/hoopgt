"""
🏀 HoopGT Static Quantization

Production-ready static quantization for edge devices.
Optimized for CNN models, VLMs, and when you have calibration data.
"""

import torch
import torch.ao.quantization as quant
from typing import Dict, Any, Optional, List, Union, Iterator
from ..hoopgt_base import HoopGTQuantizerBase
from ...types import TargetHardware


class StaticQuantizer(HoopGTQuantizerBase):
    """
    Production static quantization for edge AI.
    
    Perfect for:
    - CNN models (ResNet, EfficientNet, etc.)
    - VLMs with vision backbones
    - When you have calibration data
    - Maximum compression needs
    """
    
    algorithm_name = "static"
    description = "Static quantization - best for CNNs and when calibration data available"
    
    supported_targets = [
        TargetHardware.APPLE_SILICON,
        TargetHardware.X86_SERVER,
        TargetHardware.ARM_MOBILE,
        TargetHardware.NVIDIA_JETSON,
    ]
    
    requires_calibration = True  # Static benefits significantly from calibration
    requires_dataset = False     # But can work without it
    
    def __init__(self):
        super().__init__()
        self.backend_configs = {
            TargetHardware.APPLE_SILICON: {
                "backend": "qnnpack",  # ARM optimized
                "description": "Apple M1/M2/M3/M4 optimization",
                "dtype": torch.qint8,
                "performance_multiplier": 1.2,  # Great static quantization support
            },
            TargetHardware.X86_SERVER: {
                "backend": "fbgemm",  # Intel/AMD optimized  
                "description": "Intel/AMD x86 optimization",
                "dtype": torch.qint8,
                "performance_multiplier": 1.15,
            },
            TargetHardware.ARM_MOBILE: {
                "backend": "qnnpack",
                "description": "ARM mobile/embedded optimization", 
                "dtype": torch.qint8,
                "performance_multiplier": 0.9,  # Conservative for mobile
            },
            TargetHardware.NVIDIA_JETSON: {
                "backend": "qnnpack",  # CPU quantization on Jetson
                "description": "NVIDIA Jetson optimization",
                "dtype": torch.qint8,
                "performance_multiplier": 1.0,
            },
        }
        
        # Default calibration settings
        self.default_calibration_samples = 64
        self.default_calibration_batches = 16
    
    def can_optimize(self, model: torch.nn.Module, target: TargetHardware) -> bool:
        """Check if model can be statically quantized."""
        if target not in self.supported_targets:
            return False
        
        # Static quantization works on models with these layer types
        quantizable_layers = {
            torch.nn.Conv1d,
            torch.nn.Conv2d, 
            torch.nn.Conv3d,
            torch.nn.Linear,
        }
        
        has_quantizable = any(
            isinstance(module, tuple(quantizable_layers))
            for module in model.modules()
        )
        
        return has_quantizable
    
    def _analyze_model_architecture(self, model: torch.nn.Module) -> Dict[str, Any]:
        """Analyze model to understand quantization potential."""
        
        total_params = sum(p.numel() for p in model.parameters())
        
        # Count parameters by layer type
        conv_params = sum(
            sum(p.numel() for p in module.parameters())
            for module in model.modules()
            if isinstance(module, (torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d))
        )
        
        linear_params = sum(
            sum(p.numel() for p in module.parameters())
            for module in model.modules()
            if isinstance(module, torch.nn.Linear)
        )
        
        # Check for CNN patterns
        has_conv = conv_params > 0
        has_batchnorm = any(
            isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d))
            for module in model.modules()
        )
        
        quantizable_params = conv_params + linear_params
        quantizable_ratio = quantizable_params / total_params if total_params > 0 else 0.0
        
        return {
            "total_params": total_params,
            "conv_params": conv_params,
            "linear_params": linear_params,
            "quantizable_params": quantizable_params,
            "quantizable_ratio": quantizable_ratio,
            "has_conv": has_conv,
            "has_batchnorm": has_batchnorm,
            "conv_ratio": conv_params / total_params if total_params > 0 else 0.0,
            "linear_ratio": linear_params / total_params if total_params > 0 else 0.0,
        }
    
    def get_optimization_config(self, model: torch.nn.Module, target: TargetHardware) -> Dict[str, Any]:
        """Get hardware-optimized configuration."""
        
        if target not in self.backend_configs:
            raise ValueError(f"Unsupported target hardware: {target}")
        
        backend_config = self.backend_configs[target]
        arch_info = self._analyze_model_architecture(model)
        
        # Get quantization config for the backend
        qconfig = quant.get_default_qconfig(backend_config["backend"])
        
        return {
            "backend": backend_config["backend"],
            "dtype": backend_config["dtype"],
            "qconfig": qconfig,
            "performance_multiplier": backend_config["performance_multiplier"],
            "description": backend_config["description"],
            "calibration_samples": self.default_calibration_samples,
            "calibration_batches": self.default_calibration_batches,
        }
    
    def _calibrate_model(self, prepared_model: torch.nn.Module, calibration_data: Union[torch.Tensor, Iterator], config: Dict[str, Any]) -> None:
        """Run calibration on prepared model."""
        
        prepared_model.eval()
        samples_used = 0
        batches_used = 0
        max_samples = config.get("calibration_samples", self.default_calibration_samples)
        max_batches = config.get("calibration_batches", self.default_calibration_batches)
        
        print(f"   Running calibration (target: {max_samples} samples, {max_batches} batches)...")
        
        with torch.no_grad():
            if hasattr(calibration_data, "__iter__") and not isinstance(calibration_data, torch.Tensor):
                # Iterable dataset (like DataLoader)
                for batch_idx, batch in enumerate(calibration_data):
                    if batches_used >= max_batches:
                        break
                    
                    # Handle different batch formats
                    if isinstance(batch, (list, tuple)):
                        # (input, target) format
                        batch_input = batch[0]
                    else:
                        # Just input tensor
                        batch_input = batch
                    
                    prepared_model(batch_input)
                    
                    batch_size = batch_input.shape[0] if hasattr(batch_input, 'shape') else 1
                    samples_used += batch_size
                    batches_used += 1
                    
                    if samples_used >= max_samples:
                        break
            else:
                # Single tensor
                prepared_model(calibration_data)
                samples_used = calibration_data.shape[0] if hasattr(calibration_data, 'shape') else 1
                batches_used = 1
        
        print(f"   Calibration completed: {samples_used} samples, {batches_used} batches")
    
    def apply(self, model: torch.nn.Module, target: TargetHardware, config: Optional[Dict[str, Any]] = None) -> torch.nn.Module:
        """Apply static quantization."""
        
        opt_config = self.get_optimization_config(model, target)
        if config:
            opt_config.update(config)
        
        print(f"🏀 Applying static quantization for {target.value}")
        print(f"   Backend: {opt_config['backend']} ({opt_config['description']})")
        print(f"   Best for: CNN models with calibration data")
        
        # Set hardware-optimized backend
        torch.backends.quantized.engine = opt_config["backend"]
        
        try:
            # Prepare model for quantization
            model_copy = torch.nn.Module()
            model_copy.__dict__.update(model.__dict__)
            model_copy.qconfig = opt_config["qconfig"]
            
            prepared_model = quant.prepare(model_copy, inplace=False)
            
            # Run calibration if data is provided
            calibration_data = config.get("calibration_data") if config else None
            if calibration_data is not None:
                self._calibrate_model(prepared_model, calibration_data, opt_config)
            else:
                print("   ⚠️  No calibration data provided - using model's current state")
                print("      Static quantization works best with calibration data!")
            
            # Convert to quantized model
            quantized_model = quant.convert(prepared_model, inplace=False)
            
            print(f"✅ Static quantization completed successfully")
            return quantized_model
            
        except Exception as e:
            print(f"❌ Static quantization failed: {e}")
            print("   Returning original model")
            return model
    
    def get_performance_estimate(self, model: torch.nn.Module, target: TargetHardware) -> Dict[str, float]:
        """Estimate performance gains from static quantization."""
        
        arch_info = self._analyze_model_architecture(model)
        config = self.get_optimization_config(model, target)
        
        # Base performance estimates
        quantizable_ratio = arch_info["quantizable_ratio"]
        conv_ratio = arch_info["conv_ratio"]
        
        # Static quantization typically provides:
        # - 1.5-3x inference speedup on quantizable layers (better than dynamic)
        # - ~2.5x memory reduction on quantizable layers
        # - Higher accuracy impact without calibration (~2-5%)
        
        base_speedup = 1.0 + (quantizable_ratio * 0.5)  # More aggressive than dynamic
        base_memory_reduction = 1.0 + (quantizable_ratio * 0.8)  # Better compression
        
        # CNN models benefit significantly from static quantization
        if conv_ratio > 0.3:  # CNN-heavy model
            base_speedup *= 1.2
            base_memory_reduction *= 1.1
        
        # Apply hardware-specific multiplier
        perf_multiplier = config["performance_multiplier"]
        hardware_speedup = base_speedup * perf_multiplier
        
        # Accuracy impact depends on whether calibration data is available
        # This is an estimate - actual impact depends on calibration quality
        base_accuracy_impact = quantizable_ratio * 0.02  # 2% per fully quantizable model
        
        # CNN models typically handle quantization better
        if conv_ratio > 0.5:
            base_accuracy_impact *= 0.7  # CNNs are more robust to quantization
        
        return {
            "speed_up": min(hardware_speedup, 3.0),  # Static can be faster than dynamic
            "memory_reduction": min(base_memory_reduction, 2.5),  # Better compression
            "accuracy_impact": min(base_accuracy_impact, 0.05),  # Cap at 5%
            "quantizable_ratio": quantizable_ratio,
            "requires_calibration": True,
        }
    
    def create_calibration_dataset(self, sample_inputs: List[torch.Tensor], batch_size: int = 8) -> Iterator[torch.Tensor]:
        """Helper to create calibration dataset from sample inputs."""
        
        for i in range(0, len(sample_inputs), batch_size):
            batch = sample_inputs[i:i + batch_size]
            if len(batch) == 1:
                yield batch[0]
            else:
                yield torch.stack(batch)
    
    def benchmark(self, model: torch.nn.Module, target: TargetHardware, input_shape: tuple, runs: int = 10, calibration_data: Optional[Union[torch.Tensor, Iterator]] = None) -> Dict[str, float]:
        """Benchmark quantized vs original model performance."""
        
        # Create sample input
        sample_input = torch.randn(input_shape)
        
        # Prepare config with calibration data if provided
        config = {"calibration_data": calibration_data} if calibration_data else None
        
        # Quantize model
        quantized_model = self.apply(model, target, config)
        
        # Benchmark original model
        model.eval()
        with torch.no_grad():
            start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
            
            if start_time:
                start_time.record()
                for _ in range(runs):
                    _ = model(sample_input)
                end_time = torch.cuda.Event(enable_timing=True)
                end_time.record()
                torch.cuda.synchronize()
                original_time = start_time.elapsed_time(end_time) / runs
            else:
                import time
                start = time.time()
                for _ in range(runs):
                    _ = model(sample_input)
                original_time = (time.time() - start) / runs * 1000  # Convert to ms
        
        # Benchmark quantized model
        quantized_model.eval()
        with torch.no_grad():
            if start_time:
                start_time = torch.cuda.Event(enable_timing=True)
                start_time.record()
                for _ in range(runs):
                    _ = quantized_model(sample_input)
                end_time = torch.cuda.Event(enable_timing=True)
                end_time.record()
                torch.cuda.synchronize()
                quantized_time = start_time.elapsed_time(end_time) / runs
            else:
                start = time.time()
                for _ in range(runs):
                    _ = quantized_model(sample_input)
                quantized_time = (time.time() - start) / runs * 1000
        
        speedup = original_time / quantized_time if quantized_time > 0 else 1.0
        
        return {
            "original_time_ms": original_time,
            "quantized_time_ms": quantized_time,
            "speedup": speedup,
            "runs": runs,
            "used_calibration": calibration_data is not None,
        } 