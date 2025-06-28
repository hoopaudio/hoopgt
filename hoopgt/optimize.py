from __future__ import annotations

from typing import Any

# from hoopgt import HoopGTModel, HoopGTConfig
# from hoopgt.algorithms import HOOPGT_ALGORITHMS
# from hoopgt.config.compatibility_checks import (
#     check_algorithm_availability,
#     check_model_compatibility,
#     ensure_device_consistency,
# )
# from hoopgt.config.hoopgt_space import ALGORITHM_GROUPS
# from hoopgt.logging.logger import HoopGTLoggerContext, hoopgt_logger
from hoopgt.telemetry import track_usage


# @track_usage
# def optimize(
#     model: Any,
#     hoopgt_config: HoopGTConfig,
#     verbose: bool = False,
#     experimental: bool = False,
# ) -> HoopGTModel:
#     """
#     Optimize an arbitrary model for inference.

#     Parameters
#     ----------
#     model : Any
#         Base model to be optimized.
#     hoopgt_config : HoopGTConfig
#         Configuration settings for quantization, and compilation.
#     verbose : bool
#         Whether to print the progress of the optimization process.
#     experimental : bool
#         Whether to use experimental algorithms, e.g. to avoid checking model compatibility.
#         This can lead to undefined behavior or difficult-to-debug errors.

#     Returns
#     -------
#     HoopGTModel
#         Optimized model wrapped in a `HoopGTModel` object.
#     """
#     with HoopGTLoggerContext(verbose=verbose):
#         # check the device consistency of the model and the hoopgt config
#         ensure_device_consistency(model, hoopgt_config)

#         # check if the model type is compatible with the given configuration
#         if not experimental:
#             check_model_compatibility(model, hoopgt_config)

#         # iterate through all algorithms groups in a predefined order
#         for algorithm_group in ALGORITHM_GROUPS:
#             current_algorithm = hoopgt_config[algorithm_group]

#             if current_algorithm is not None:
#                 check_algorithm_availability(current_algorithm, algorithm_group, HOOPGT_ALGORITHMS)
#                 # apply the active algorithm to the model
#                 hoopgt_logger.info(f"Starting {algorithm_group} {current_algorithm}...")
#                 algorithm_instance = HOOPGT_ALGORITHMS[algorithm_group][current_algorithm]
#                 model = algorithm_instance.apply(model, hoopgt_config=hoopgt_config)
#                 hoopgt_logger.info(f"{algorithm_group} {current_algorithm} was applied successfully.")

#         # wrap the model in a HoopGTModel object before returning
#         optimized_model = HoopGTModel(model, hoopgt_config=hoopgt_config)

#     return optimized_model
