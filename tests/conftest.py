"""Shared PyTest fixtures for Aura test suite."""

import tempfile
from pathlib import Path
import pytest
import yaml

from src.utils.config import AppConfig
from src.utils.config_loader import ConfigLoader


@pytest.fixture
def temp_dir():
    """Provides a temporary directory for isolation testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_config_yaml(temp_dir):
    """Generates a temporary YAML config file for testing load operations."""
    config_dict = {
        "system": {"seed": 123, "device": "cpu", "num_workers": 2, "pin_memory": False, "deterministic": True},
        "logging": {"level": "DEBUG", "file_logging": False, "log_dir": str(temp_dir / "logs")},
        "model": {"name": "test-gpt", "vocab_size": 1000, "max_sequence_length": 128, "d_model": 64, "n_layers": 2, "n_heads": 4, "d_ff": 256, "dropout": 0.1, "bias": False, "initializer_range": 0.02},
        "tokenizer": {"vocab_size": 1000, "special_tokens": {"pad_token": "<|pad|>"}},
        "dataset": {"dataset_dir": str(temp_dir / "data"), "train_split": "train", "val_split": "val", "test_split": "test"},
        "training": {"batch_size": 4, "gradient_accumulation_steps": 1, "learning_rate": 1e-3, "min_learning_rate": 1e-4, "weight_decay": 0.01, "beta1": 0.9, "beta2": 0.99, "grad_clip": 1.0, "warmup_steps": 10, "max_steps": 100, "eval_interval": 20, "save_interval": 50, "checkpoint_dir": str(temp_dir / "checkpoints")},
        "inference": {"max_new_tokens": 50, "temperature": 0.8, "top_k": 40, "top_p": 0.95},
    }

    yaml_path = temp_dir / "test_config.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(config_dict, f)

    return yaml_path
