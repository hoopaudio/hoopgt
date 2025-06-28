"""
HoopGT SDK - Model Optimization Platform

Clean, production-ready AI model optimization for edge devices.
"""

# from hoopgt.config.hoopgt_config import HoopGTConfig
# from hoopgt.engine.pruna_model import HoopGTModel
# from hoopgt.optimize import optimize 
# from hoopgt.algorithms import HOOPGT_ALGORITHMS
from importlib_metadata import version

__version__ = version(__name__)

# __all__ = ["HoopGTConfig", "HoopGTModel", "optimize", "HOOPGT_ALGORITHMS", "__version__"]
__all__ = ["__version__"]
