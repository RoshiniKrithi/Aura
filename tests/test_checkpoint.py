"""Unit tests for checkpoint manager utilities."""

import torch

from src.utils.checkpoint import CheckpointManager


def test_checkpoint_manager_save_load_prune(temp_dir):
    """Verify saving, loading, latest identification, and top-k pruning of checkpoints."""
    manager = CheckpointManager(checkpoint_dir=temp_dir, max_to_keep=2)

    # Save 3 checkpoints
    for step in range(1, 4):
        dummy_state = {"step": step, "weights": torch.tensor([float(step)])}
        manager.save_checkpoint(state_dict=dummy_state, step=step)

    # Verify latest checkpoint is step 3
    latest = manager.get_latest_checkpoint()
    assert latest is not None
    assert "0000003" in latest.name

    # Load step 3 state
    loaded_state = manager.load_checkpoint(latest)
    assert loaded_state["step"] == 3
    assert torch.equal(loaded_state["weights"], torch.tensor([3.0]))

    # Verify pruning kept max_to_keep (2) files
    remaining_files = list(temp_dir.glob("checkpoint_step_*.pt"))
    assert len(remaining_files) == 2
