"""PEFT and LoRA Configuration Dataclasses for Aura EXP-007.

Provides LoRAConfig and PEFTTrainingConfig dataclasses.
"""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class LoRAConfig:
    """Configuration container for Low-Rank Adaptation (LoRA) hyperparameters.

    Attributes:
        r: Rank dimension of decomposition matrices A and B (default: 16).
        alpha: Constant scaling factor for adapter outputs (default: 32.0).
        dropout: Dropout probability applied to adapter input features (default: 0.05).
        target_modules: List of module name substrings to replace with LoRALinear layers.
        bias: Bias configuration mode ("none", "all", "lora_only").
    """

    r: int = 16
    alpha: float = 32.0
    dropout: float = 0.05
    target_modules: List[str] = field(
        default_factory=lambda: ["c_attn", "c_proj", "w1", "w2", "q_proj", "v_proj"]
    )
    bias: str = "none"

    def to_dict(self) -> Dict[str, Any]:
        """Converts LoRAConfig to dictionary representation."""
        return asdict(self)


@dataclass
class PEFTTrainingConfig:
    """Configuration container for Parameter-Efficient Fine-Tuning execution.

    Attributes:
        experiment_id: Unique string identifier for experiment tracking.
        adapter_name: Standalone adapter name tag (e.g. "aura-dsa-adapter").
        version: Version string tag for adapter tracking (e.g. "v1.0").
        phase: Project hierarchy phase tag.
        seed: Random seed for reproducibility.
        device: Computation target device ("cuda", "cpu", "auto").
        pretrained_checkpoint_path: Optional path to pre-trained base model weights (.pt).
        data_dir: Path to instruction tuning data directory.
        tokenizer_dir: Path to BPE tokenizer directory.
        lora_config: Embedded LoRAConfig instance.
        learning_rate: Optimizer learning rate for adapter parameters.
        weight_decay: L2 regularization coefficient.
        max_steps: Maximum training steps.
        eval_interval: Evaluation interval in steps.
        save_interval: Checkpoint saving interval in steps.
        micro_batch_size: Per-GPU micro-batch size.
        gradient_accumulation_steps: Gradient accumulation steps.
        max_sequence_length: Maximum sequence context window length L.
        d_model: Base model embedding size.
        n_layers: Transformer layers.
        n_heads: Attention heads.
        d_ff: Feed-forward dimension.
        warmup_steps: Learning rate warmup steps.
        output_dir: Output directory path for adapter artifacts.
    """

    experiment_id: str = "EXP-007_LoRA_PEFT_v1.0"
    adapter_name: str = "aura_dsa_adapter"
    version: str = "v1.0"
    phase: str = "Phase 26"
    seed: int = 42
    device: str = "auto"

    pretrained_checkpoint_path: Optional[str] = None
    data_dir: str = "data/instructions"
    tokenizer_dir: str = "data/tokenizer"

    lora_config: LoRAConfig = field(default_factory=LoRAConfig)

    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    max_steps: int = 20
    eval_interval: int = 5
    save_interval: int = 5
    micro_batch_size: int = 4
    gradient_accumulation_steps: int = 2

    vocab_size: int = 50260
    max_sequence_length: int = 512
    d_model: int = 768
    n_layers: int = 12
    n_heads: int = 12
    d_ff: int = 3072
    warmup_steps: int = 2

    output_dir: str = "outputs/experiments/EXP-007_LoRA_PEFT_v1.0"

    def to_dict(self) -> Dict[str, Any]:
        """Converts PEFTTrainingConfig to dictionary representation."""
        res = asdict(self)
        res["lora_config"] = self.lora_config.to_dict()
        return res
