"""Experiment tracking and directory management utilities.

Generates structured, timestamped output directories for tracking run configs,
metrics, and execution artifacts.
"""

from datetime import datetime
import json
import logging
from pathlib import Path
from typing import Any, Dict
import yaml

from src.utils.paths import project_paths

logger = logging.getLogger(__name__)


class ExperimentManager:
    """Manages experiment directory lifecycle, run metadata, and metric persistence."""

    def __init__(self, experiment_name: str = "default_run", base_dir: Path | str | None = None) -> None:
        """Initialize experiment manager.

        Args:
            experiment_name: Name label for the current experiment.
            base_dir: Base directory for experiments. Defaults to outputs/runs.
        """
        if base_dir is None:
            self._base_dir = project_paths.outputs_dir / "runs"
        else:
            self._base_dir = Path(base_dir).resolve()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._run_id = f"{experiment_name}_{timestamp}"
        self._run_dir = self._base_dir / self._run_id

        # Ensure run directory structure
        project_paths.ensure_dir(self._run_dir)
        project_paths.ensure_dir(self._run_dir / "checkpoints")
        project_paths.ensure_dir(self._run_dir / "logs")

        logger.info("Initialized Experiment Run: %s at %s", self._run_id, self._run_dir)

    @property
    def run_id(self) -> str:
        """Returns the unique run identifier."""
        return self._run_id

    @property
    def run_dir(self) -> Path:
        """Returns the root directory for this experiment run."""
        return self._run_dir

    @property
    def checkpoint_dir(self) -> Path:
        """Returns the run-specific checkpoint directory."""
        return self._run_dir / "checkpoints"

    @property
    def log_dir(self) -> Path:
        """Returns the run-specific log directory."""
        return self._run_dir / "logs"

    def save_config(self, config_data: Dict[str, Any]) -> Path:
        """Saves run configuration snapshot to experiment folder.

        Args:
            config_data: Configuration dictionary to serialize.

        Returns:
            Path to saved run_config.yaml.
        """
        config_path = self._run_dir / "run_config.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(config_data, f, default_flow_style=False)
        logger.info("Saved run configuration snapshot to %s", config_path)
        return config_path

    def save_metrics(self, metrics: Dict[str, Any], filename: str = "metrics.json") -> Path:
        """Saves evaluation or training metrics snapshot.

        Args:
            metrics: Metrics dictionary to serialize.
            filename: Target output JSON filename.

        Returns:
            Path to saved metrics file.
        """
        metrics_path = self._run_dir / filename
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        logger.info("Saved experiment metrics to %s", metrics_path)
        return metrics_path
