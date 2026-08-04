"""Checkpoint Metadata Schema and Registry Index for Aura LLM Architecture.

Provides CheckpointMetadata records and MetadataRegistry for tracking checkpoint files,
timestamps, metrics, and rotation indexes in metadata.json.
"""

from dataclasses import asdict, dataclass, field
import json
import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class CheckpointMetadata:
    """Metadata record for a saved model checkpoint."""

    checkpoint_name: str
    epoch: int
    global_step: int
    timestamp: str
    metrics: Dict[str, float] = field(default_factory=dict)
    is_best: bool = False
    file_size_bytes: int = 0
    sha256_checksum: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Converts CheckpointMetadata to dictionary format."""
        return asdict(self)


class MetadataRegistry:
    """Manages metadata.json registry index in checkpoint directory."""

    def __init__(self, checkpoint_dir: str = "checkpoints") -> None:
        """Initializes MetadataRegistry.

        Args:
            checkpoint_dir: Path to directory containing checkpoints.
        """
        self.checkpoint_dir = checkpoint_dir
        self.metadata_file = os.path.join(checkpoint_dir, "metadata.json")
        self.records: List[Dict[str, Any]] = []
        self._load()

    def _load(self) -> None:
        """Loads metadata registry from metadata.json if it exists."""
        if os.path.exists(self.metadata_file):
            try:
                with open(self.metadata_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.records = data.get("checkpoints", [])
            except Exception as e:
                logger.warning("Failed to load metadata registry '%s': %s", self.metadata_file, str(e))
                self.records = []

    def save(self) -> None:
        """Persists metadata registry list to metadata.json."""
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        try:
            with open(self.metadata_file, "w", encoding="utf-8") as f:
                json.dump({"checkpoints": self.records}, f, indent=2)
        except Exception as e:
            logger.error("Failed to save metadata registry: %s", str(e))

    def add_record(self, metadata: CheckpointMetadata) -> None:
        """Appends a new checkpoint metadata record and updates registry.

        Args:
            metadata: CheckpointMetadata record object.
        """
        self.records.append(metadata.to_dict())
        self.save()

    def get_best_record(self) -> Optional[Dict[str, Any]]:
        """Returns the metadata record marked as best, if available."""
        for rec in reversed(self.records):
            if rec.get("is_best", False):
                return rec
        return None
