from hoopgt.config.hoopgt_config import HoopGTConfig 


def is_empty_config(config: HoopGTConfig) -> bool:
    """
    Check if the HoopGTConfig is empty.

    Parameters
    ----------
    config : HoopGTConfig
        The HoopGTConfig to check.

    Returns
    -------
    bool
        True if the HoopGTConfig is empty, False otherwise.
    """
    empty_config = HoopGTConfig()
    return config == empty_config
