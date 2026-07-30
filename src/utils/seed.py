"""Random seed utility for deterministic execution across PyTorch and Python libraries.

Ensures strict scientific reproducibility across model initialization, data shuffling,
and hardware execution kernels.
"""

import logging
import random
import numpy as np
import torch

logger = logging.getLogger(__name__)


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """Set random seed across all libraries to ensure reproducible experiments.

    Sets seeds for Python built-in random, NumPy, PyTorch CPU, PyTorch CUDA (all GPUs),
    and configures cuDNN backend deterministic execution.

    Args:
        seed: Integer random seed value.
        deterministic: If True, configures PyTorch backends to prioritize deterministic
            algorithms (may slightly reduce performance on specific GPU architectures).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        if hasattr(torch, "use_deterministic_algorithms"):
            try:
                torch.use_deterministic_algorithms(True)
            except RuntimeError as err:
                logger.warning("Deterministic algorithms unavailable for some operations: %s", err)

    logger.info("Random seed set to %d (Deterministic: %s)", seed, deterministic)
