"""Checkpoint management system for saving, loading, and pruning PyTorch model states.

Handles serialization of model state_dict, optimizer state, learning rate schedulers,
training step metadata, and configuration snapshots.
"""

import glob
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import torch

from src.utils.paths import project_paths

logger = logging.getLogger(__name__)


class CheckpointManager:
    """Manages model checkpoint serialization, recovery, and retention pruning."""

    def __init__(
        self,
        checkpoint_dir: Path | str | None = None,
        max_to_keep: int = 5,
    ) -> None:
        """Initialize checkpoint manager.

        Args:
            checkpoint_dir: Target directory for storing checkpoints.
            max_to_keep: Maximum number of recent checkpoints to retain before pruning.
        """
        if checkpoint_dir is None:
            self._checkpoint_dir = project_paths.checkpoints_dir
        else:
            self._checkpoint_dir = Path(checkpoint_dir).resolve()

        self._max_to_keep = max_to_keep
        project_paths.ensure_dir(self._checkpoint_dir)

    @property
    def checkpoint_dir(self) -> Path:
        """Returns target checkpoint directory."""
        return self._checkpoint_dir

    def save_checkpoint(
        self,
        state_dict: Dict[str, Any],
        step: int,
        filename: Optional[str] = None,
        is_best: bool = False,
    ) -> Path:
        """Save state dictionary to disk with step metadata and prune older checkpoints.

        Args:
            state_dict: Dictionary containing model, optimizer, scheduler, and step states.
            step: Current global training step.
            filename: Custom filename (optional).
            is_best: If True, creates a copy named 'best_checkpoint.pt'.

        Returns:
            Path to saved checkpoint file.
        """
        if filename is None:
            filename = f"checkpoint_step_{step:07d}.pt"

        save_path = self._checkpoint_dir / filename
        torch.save(state_dict, save_path)
        logger.info("Saved checkpoint at step %d to %s", step, save_path)

        if is_best:
            best_path = self._checkpoint_dir / "best_checkpoint.pt"
            torch.save(state_dict, best_path)
            logger.info("Updated best model checkpoint at %s", best_path)

        self._prune_old_checkpoints()
        return save_path

    def load_checkpoint(
        self,
        checkpoint_path: Path | str,
        device: torch.device | str = "cpu",
    ) -> Dict[str, Any]:
        """Load state dictionary from disk.

        Args:
            checkpoint_path: Path to target checkpoint file.
            device: PyTorch device mapping target.

        Returns:
            Loaded state dictionary.

        Raises:
            FileNotFoundError: If checkpoint file does not exist.
        """
        path = Path(checkpoint_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint file not found: {path}")

        logger.info("Loading checkpoint from %s onto device %s", path, device)
        state_dict = torch.load(path, map_location=device)
        return state_dict

    def get_latest_checkpoint(self) -> Optional[Path]:
        """Finds and returns the most recent checkpoint file based on step number.

        Returns:
            Path to latest checkpoint or None if no checkpoints exist.
        """
        pattern = str(self._checkpoint_dir / "checkpoint_step_*.pt")
        files = glob.glob(pattern)

        if not files:
            return None

        # Sort files by step integer in filename
        files.sort(key=lambda f: int(Path(f).stem.split("_")[-1]))
        return Path(files[-1])

    def _prune_old_checkpoints(self) -> None:
        """Deletes older step checkpoints exceeding self._max_to_keep limit."""
        pattern = str(self._checkpoint_dir / "checkpoint_step_*.pt")
        files = glob.glob(pattern)

        if len(files) <= self._max_to_keep:
            return

        files.sort(key=lambda f: int(Path(f).stem.split("_")[-1]))
        files_to_delete = files[: -self._max_to_keep]

        for file_path in files_to_delete:
            try:
                os.remove(file_path)
                logger.info("Pruned old checkpoint: %s", file_path)
            except OSError as err:
                logger.warning("Failed to delete old checkpoint %s: %s", file_path, err)
