"""Utilities Module.

WHY THIS MODULE EXISTS:
    Provides infrastructure utilities including hardware device auto-detection (CPU/CUDA/MPS),
    seed setting for strict scientific determinism, path management, configuration loading,
    experiment run tracking, and checkpoint serialization.

HOW FUTURE MODULES WILL PLUG IN:
    - All subsequent phases import `project_paths`, `get_device`, `set_seed`, `load_config`,
      `CheckpointManager`, and `ExperimentManager` to maintain consistent runtime infrastructure.
"""

from src.utils.checkpoint import CheckpointManager
from src.utils.config import AppConfig
from src.utils.config_loader import ConfigLoader, load_config
from src.utils.device import DeviceManager, get_device
from src.utils.experiment import ExperimentManager
from src.utils.paths import PathManager, project_paths
from src.utils.seed import set_seed

__all__ = [
    "PathManager",
    "project_paths",
    "DeviceManager",
    "get_device",
    "set_seed",
    "AppConfig",
    "ConfigLoader",
    "load_config",
    "ExperimentManager",
    "CheckpointManager",
]
