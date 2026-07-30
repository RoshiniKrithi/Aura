"""Path resolution utilities for the Aura project.

Provides a centralized manager for resolving relative and absolute paths
across all project components (configs, data, checkpoints, outputs, etc.).
"""

import os
from pathlib import Path


class PathManager:
    """Centralized manager for project directory structure and path resolution."""

    def __init__(self, root_dir: Path | str | None = None) -> None:
        """Initialize the path manager.

        Args:
            root_dir: Optional root directory path. If None, infers the project root
                relative to this file.
        """
        if root_dir is None:
            # Resolves from src/utils/paths.py -> src/utils -> src -> Aura Root
            self._root_dir = Path(__file__).resolve().parent.parent.parent
        else:
            self._root_dir = Path(root_dir).resolve()

    @property
    def root(self) -> Path:
        """Returns the project root directory."""
        return self._root_dir

    @property
    def configs_dir(self) -> Path:
        """Returns the configs directory."""
        return self.ensure_dir(self._root_dir / "configs")

    @property
    def data_dir(self) -> Path:
        """Returns the data directory."""
        return self.ensure_dir(self._root_dir / "data")

    @property
    def docs_dir(self) -> Path:
        """Returns the docs directory."""
        return self.ensure_dir(self._root_dir / "docs")

    @property
    def notebooks_dir(self) -> Path:
        """Returns the notebooks directory."""
        return self.ensure_dir(self._root_dir / "notebooks")

    @property
    def src_dir(self) -> Path:
        """Returns the src directory."""
        return self.ensure_dir(self._root_dir / "src")

    @property
    def tests_dir(self) -> Path:
        """Returns the tests directory."""
        return self.ensure_dir(self._root_dir / "tests")

    @property
    def scripts_dir(self) -> Path:
        """Returns the scripts directory."""
        return self.ensure_dir(self._root_dir / "scripts")

    @property
    def checkpoints_dir(self) -> Path:
        """Returns the checkpoints directory."""
        return self.ensure_dir(self._root_dir / "checkpoints")

    @property
    def outputs_dir(self) -> Path:
        """Returns the outputs directory."""
        return self.ensure_dir(self._root_dir / "outputs")

    @property
    def default_config_path(self) -> Path:
        """Returns the default configuration YAML file path."""
        return self.configs_dir / "config.yaml"

    @staticmethod
    def ensure_dir(dir_path: Path | str) -> Path:
        """Ensures that a directory exists, creating parents if necessary.

        Args:
            dir_path: Directory path to ensure exists.

        Returns:
            Resolved Path object.
        """
        path = Path(dir_path).resolve()
        path.mkdir(parents=True, exist_ok=True)
        return path


# Global default path manager instance
project_paths = PathManager()
