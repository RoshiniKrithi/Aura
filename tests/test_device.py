"""Unit tests for device detection utilities."""

import pytest
import torch

from src.utils.device import DeviceManager, get_device


def test_detect_device_cpu():
    """Verify explicit CPU device resolution."""
    device = DeviceManager.detect_device("cpu")
    assert isinstance(device, torch.device)
    assert device.type == "cpu"


def test_detect_device_auto():
    """Verify auto device detection returns valid device."""
    device = get_device("auto")
    assert isinstance(device, torch.device)
    assert device.type in ["cuda", "mps", "cpu"]


def test_invalid_device_raises_error():
    """Verify invalid device string raises ValueError."""
    with pytest.raises(ValueError):
        DeviceManager.detect_device("invalid_device")
