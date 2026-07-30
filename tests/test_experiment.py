"""Unit tests for experiment manager utilities."""

from src.utils.experiment import ExperimentManager


def test_experiment_manager_dir_creation(temp_dir):
    """Verify experiment directory structure creation and metadata saving."""
    exp = ExperimentManager(experiment_name="test_exp", base_dir=temp_dir)

    assert exp.run_dir.exists()
    assert exp.checkpoint_dir.exists()
    assert exp.log_dir.exists()

    cfg_file = exp.save_config({"batch_size": 16, "lr": 0.001})
    assert cfg_file.exists()

    metrics_file = exp.save_metrics({"val_loss": 0.42})
    assert metrics_file.exists()
