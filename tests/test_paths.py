"""Unit tests for path manager utilities."""

from pathlib import Path

from src.utils.paths import PathManager, project_paths


def test_project_paths_resolution():
    """Verify PathManager resolves existing directory structure."""
    assert project_paths.root.exists()
    assert project_paths.src_dir.exists()
    assert project_paths.configs_dir.exists()


def test_custom_root_path_manager(temp_dir):
    """Verify PathManager initializes correctly with custom root directory."""
    pm = PathManager(root_dir=temp_dir)
    assert pm.root == temp_dir
    assert pm.data_dir.exists()
    assert pm.checkpoints_dir.exists()
