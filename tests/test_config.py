"""Unit tests for configuration loader and dataclass schema."""

import pytest

from src.utils.config import AppConfig, ModelConfig
from src.utils.config_loader import ConfigLoader, load_config


def test_model_config_head_dim_calculation():
    """Verify head_dim calculation and divison validation."""
    cfg = ModelConfig(d_model=768, n_heads=12)
    assert cfg.head_dim == 64

    invalid_cfg = ModelConfig(d_model=768, n_heads=11)
    with pytest.raises(ValueError):
        _ = invalid_cfg.head_dim


def test_config_loader_from_yaml(sample_config_yaml):
    """Verify loading AppConfig from custom YAML path."""
    app_cfg = ConfigLoader.from_yaml(sample_config_yaml)
    assert isinstance(app_cfg, AppConfig)
    assert app_cfg.system.seed == 123
    assert app_cfg.model.d_model == 64
    assert app_cfg.training.max_steps == 100


def test_config_file_not_found():
    """Verify FileNotFoundError is raised when YAML path is invalid."""
    with pytest.raises(FileNotFoundError):
        ConfigLoader.from_yaml("non_existent_config.yaml")
