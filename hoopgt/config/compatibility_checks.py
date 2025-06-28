from __future__ import annotations

from typing import Any

from hoopgt import HoopGTConfig
from hoopgt.algorithms import HOOPGT_ALGORITHMS
from hoopgt.config.optimize_space import OPTIMIZE_SPACE
from hoopgt.engine.utils import get_device, move_to_device
from hoopgt.logging.logger import hoopgt_logger


def ensure_device_consistency(model, hoopgt_config):
    """
    Ensure consistency between the device state of the model and the hoopgt config.

    Parameters
    ----------
    model : Any
        The model to check for device consistency.
    hoopgt_config : HoopGTConfig
        The hoopgt config to check for device consistency.
    """
    model_device = get_device(model)

    # model and optimize config devices match
    if model_device == hoopgt_config.device:
        hoopgt_logger.debug("Device consistency check passed.")
        # in case of accelerate, we need to store the device map
        if model_device == "accelerate":
            hoopgt_logger.debug("Device consistency check passed.")
            hf_device_map = get_device(model, return_device_map=True)
            if not all(isinstance(v, int) for v in hf_device_map.values()):
                raise ValueError("Device map indicates CPU offloading, this is not supported at this time.")
            else:
                hoopgt_config.device_map = hf_device_map

    elif hoopgt_config.device in ["cpu", "cuda", "mps"] and model_device in ["cpu", "cuda", "mps"]:
        hoopgt_logger.warning(
            (
                f"Model and HoopGTConfig have different devices. Model: {model_device}, "
                f"HoopGTConfig: {hoopgt_config.device}. Casting model to {hoopgt_config.device}."
                f"If this is not desired, please use HoopGTConfig(device='{model_device}')."
            )
        )
        move_to_device(model, hoopgt_config.device)

    elif hoopgt_config.device == "accelerate" or model_device == "accelerate":
        hoopgt_logger.warning(
            (
                f"Model and HoopGTConfig have different devices. Model: {model_device}, "
                f"HoopGTConfig: {hoopgt_config.device}. Updating HoopGTConfig to device='{model_device}'."
            )
        )   
        hoopgt_config.device = model_device
    else:
        raise ValueError(f"Invalid device: {hoopgt_config.device}")


def check_model_compatibility(
    model: Any,
    hoopgt_config: HoopGTConfig,
    algorithm_dict: dict[str, Any] = HOOPGT_ALGORITHMS,
) -> None:
    """
    Check if the model is compatible with the given configuration.

    Parameters
    ----------
    model : Any
        The model to check for compatibility with the HoopGTConfig.
    hoopgt_config : HoopGTConfig
        The HoopGTConfig to check the model against.
    algorithm_dict : dict[str, Any]
        The algorithm dictionary to hold all algorithm instances.
    """
    # algorithm groups are subject to change, make sure we have the latest version
    from hoopgt.config.optimize_space import ALGORITHM_GROUPS

    # iterate through compiler, quantizer, ...
    for current_group in ALGORITHM_GROUPS:
        algorithm = hoopgt_config[current_group]
        if algorithm is not None:
            check_algorithm_availability(algorithm, current_group, algorithm_dict)
            # test if all required packages are installed, if not this will raise an ImportError
            algorithm_dict[current_group][algorithm].import_algorithm_packages()
            check_argument_compatibility(hoopgt_config, algorithm)
            # check for model-algorithm compatibility with the model_check_fn
            if not algorithm_dict[current_group][algorithm].model_check_fn(model):
                raise ValueError(
                    f"Model is not compatible with {algorithm_dict[current_group][algorithm].algorithm_name}"
                )
            if get_device(model) not in algorithm_dict[current_group][algorithm].runs_on:
                raise ValueError(
                    f"{algorithm} is not compatible with device {get_device(model)}, "
                    f"compatible devices are {algorithm_dict[current_group][algorithm].runs_on}"
                )


def check_argument_compatibility(hoopgt_config: HoopGTConfig, algorithm_name: str) -> None:
    """
    Check if the HoopGTConfig has the required arguments (tokenizer, processor, dataset) for an algorithm.

    Parameters
    ----------
    hoopgt_config : HoopGTConfig
        The HoopGTConfig to check the argument consistency with.
    algorithm_name : str
        The algorithm name that is about to be activated.
    """
    algorithm_requirements = OPTIMIZE_SPACE.model_requirements[algorithm_name]
    if algorithm_requirements["tokenizer_required"] and hoopgt_config.tokenizer is None:
        raise ValueError(f"{algorithm_name} requires a tokenizer. Please provide it with hoopgt_config.add_tokenizer().")
    if algorithm_requirements["processor_required"] and hoopgt_config.processor is None:
        raise ValueError(f"{algorithm_name} requires a processor. Please provide it with hoopgt_config.add_processor().")
    if algorithm_requirements["dataset_required"] and hoopgt_config.data is None:
        raise ValueError(f"{algorithm_name} requires a dataset. Please provide it with hoopgt_config.add_data().")


def check_algorithm_availability(algorithm: str, algorithm_group: str, algorithm_dict: dict[str, Any]) -> None:
    """
    Check if the algorithm is available in the algorithm dictionary.

    Parameters
    ----------
    algorithm : str
        The algorithm to check for availability.
    algorithm_group : str
        The algorithm group to check for availability.
    algorithm_dict : dict[str, Any]
        The algorithm dictionary to check for availability.

    Raises
    ------
    ValueError
        If the algorithm is not available in the algorithm dictionary.
    """
    if algorithm_group not in algorithm_dict:
        raise RuntimeError(f"Algorithm group {algorithm_group} is unavailable with hoopgt.optimize")
    if algorithm not in algorithm_dict[algorithm_group]:
        raise RuntimeError(f"Algorithm {algorithm} is unavailable with hoopgt.optimize")
