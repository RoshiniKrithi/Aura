"""Strongly-typed configuration dataclasses for Aura architecture.

Provides structured validation and explicit typing for all hyperparameter,
system, hardware, logging, dataset, training, and inference settings.
"""

from dataclasses import dataclass, field
from typing import Dict, Any


@dataclass(frozen=True)
class SystemConfig:
    """System and hardware execution settings."""

    seed: int = 42
    device: str = "auto"
    num_workers: int = 4
    pin_memory: bool = True
    deterministic: bool = True


@dataclass(frozen=True)
class LoggingConfig:
    """Logging and experiment monitoring settings."""

    level: str = "INFO"
    file_logging: bool = True
    log_dir: str = "outputs/logs"


@dataclass(frozen=True)
class ModelConfig:
    """GPT-style Transformer architecture hyperparameters."""

    name: str = "aura-gpt-base"
    vocab_size: int = 50257
    max_sequence_length: int = 1024
    d_model: int = 768
    n_layers: int = 12
    n_heads: int = 12
    d_ff: int = 3072
    dropout: float = 0.1
    bias: bool = False
    initializer_range: float = 0.02

    @property
    def head_dim(self) -> int:
        """Calculates dimension per attention head."""
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
            )
        return self.d_model // self.n_heads


@dataclass(frozen=True)
class TokenizerConfig:
    """Tokenizer configuration parameters."""

    vocab_size: int = 50257
    special_tokens: Dict[str, str] = field(
        default_factory=lambda: {
            "pad_token": "<|pad|>",
            "unk_token": "<|unk|>",
            "bos_token": "<|startoftext|>",
            "eos_token": "<|endoftext|>",
        }
    )


@dataclass(frozen=True)
class DatasetConfig:
    """Dataset split and directory configuration."""

    dataset_dir: str = "data"
    train_split: str = "train"
    val_split: str = "validation"
    test_split: str = "test"


@dataclass(frozen=True)
class TrainingConfig:
    """Training optimization and loop hyperparameters."""

    batch_size: int = 16
    gradient_accumulation_steps: int = 4
    learning_rate: float = 6.0e-4
    min_learning_rate: float = 6.0e-5
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    warmup_steps: int = 2000
    max_steps: int = 100000
    eval_interval: int = 500
    save_interval: int = 1000
    checkpoint_dir: str = "checkpoints"


@dataclass(frozen=True)
class InferenceConfig:
    """Inference and generation decoding options."""

    max_new_tokens: int = 512
    temperature: float = 0.7
    top_k: int = 50
    top_p: float = 0.9


@dataclass(frozen=True)
class AppConfig:
    """Root configuration aggregating all Aura system modules."""

    system: SystemConfig = field(default_factory=SystemConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
