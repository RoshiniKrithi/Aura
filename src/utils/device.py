"""Hardware device detection and configuration utilities.

Provides automatic detection and management of computing devices (CUDA, Apple Silicon MPS, CPU)
for PyTorch tensor execution.
"""

import logging
import torch

logger = logging.getLogger(__name__)


class DeviceManager:
    """Manages PyTorch execution device resolution and hardware diagnostics."""

    @staticmethod
    def detect_device(requested_device: str = "auto") -> torch.device:
        """Detect and return the optimal PyTorch device based on hardware availability.

        Resolution hierarchy when requested_device is "auto":
        1. CUDA (NVIDIA GPUs)
        2. MPS (Apple Silicon GPUs)
        3. CPU (Fallback)

        Args:
            requested_device: Desired target device ("auto", "cuda", "mps", "cpu").

        Returns:
            torch.device instance representing target hardware.

        Raises:
            ValueError: If an unrecognized device string is passed.
        """
        requested = requested_device.lower().strip()

        if requested == "auto":
            if torch.cuda.is_available():
                device = torch.device("cuda")
                logger.info(
                    "Automatic device detection selected CUDA: %s",
                    torch.cuda.get_device_name(0),
                )
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = torch.device("mps")
                logger.info("Automatic device detection selected Apple Silicon MPS.")
            else:
                device = torch.device("cpu")
                logger.info("Automatic device detection selected CPU.")
            return device

        if requested == "cuda":
            if not torch.cuda.is_available():
                logger.warning("CUDA requested but not available. Falling back to CPU.")
                return torch.device("cpu")
            return torch.device("cuda")

        if requested == "mps":
            if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
                logger.warning("MPS requested but not available. Falling back to CPU.")
                return torch.device("cpu")
            return torch.device("mps")

        if requested == "cpu":
            return torch.device("cpu")

        raise ValueError(
            f"Unsupported device '{requested_device}'. Must be one of: 'auto', 'cuda', 'mps', 'cpu'."
        )

    @staticmethod
    def log_device_info(device: torch.device) -> None:
        """Logs detailed diagnostics for the active compute device.

        Args:
            device: Active PyTorch device.
        """
        if device.type == "cuda":
            device_idx = device.index or 0
            name = torch.cuda.get_device_name(device_idx)
            cap = torch.cuda.get_device_capability(device_idx)
            vram = torch.cuda.get_device_properties(device_idx).total_memory / (1024**3)
            logger.info("Active Device: CUDA [%s] | Compute Cap: %d.%d | VRAM: %.2f GB", name, cap[0], cap[1], vram)
        elif device.type == "mps":
            logger.info("Active Device: Apple Silicon MPS")
        else:
            logger.info("Active Device: CPU")


def get_device(requested_device: str = "auto") -> torch.device:
    """Convenience function for resolving PyTorch hardware device.

    Args:
        requested_device: Target device string ("auto", "cuda", "mps", "cpu").

    Returns:
        Configured torch.device instance.
    """
    device = DeviceManager.detect_device(requested_device)
    DeviceManager.log_device_info(device)
    return device
