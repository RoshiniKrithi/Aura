"""Unit tests for random seed determinism utilities."""

import random
import numpy as np
import torch

from src.utils.seed import set_seed


def test_set_seed_reproducibility():
    """Verify that set_seed ensures deterministic random number generation across libraries."""
    set_seed(42, deterministic=True)
    py_val1 = random.random()
    np_val1 = np.random.rand(1)[0]
    torch_val1 = torch.rand(1).item()

    set_seed(42, deterministic=True)
    py_val2 = random.random()
    np_val2 = np.random.rand(1)[0]
    torch_val2 = torch.rand(1).item()

    assert py_val1 == py_val2
    assert np_val1 == np_val2
    assert torch_val1 == torch_val2
