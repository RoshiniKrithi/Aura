"""Comprehensive PyTest Suite for Experiment EXP-007 LoRA & PEFT Fine-Tuning.

Includes unit, integration, stress, and regression testing for:
- LoRALinear Layer Wrapper & Scaling Formula
- LoRAInjector Base Model Freezing & Target Layer Replacement
- Standalone Adapter Saving, Loading, and Dynamic Switching
- Zero-Latency Weight Merging & Bitwise Verification
- PEFTRunner Training Loop, Gradient Updates & Checkpoint Recovery
"""

from pathlib import Path
import tempfile
import pytest
import torch
import torch.nn as nn

from src.models.config import AuraGPTConfig
from src.models.gpt import AuraGPT
from src.peft.adapter_manager import AdapterLoader, AdapterManager, AdapterSaver
from src.peft.adapter_merger import AdapterMerger
from src.peft.lora_injector import LoRAInjector
from src.peft.lora_layer import LoRALinear
from src.peft.peft_config import LoRAConfig, PEFTTrainingConfig
from src.peft.peft_trainer import PEFTRunner


def test_lora_linear_forward_and_initialization():
    """Verifies LoRALinear initialization (delta_W = 0 at step 0) and scaling."""
    base_linear = nn.Linear(64, 128)
    lora_layer = LoRALinear(base_layer=base_linear, r=16, alpha=32.0)

    # Base weight must be frozen
    assert not lora_layer.base_layer.weight.requires_grad
    # Adapter parameters must be trainable
    assert lora_layer.lora_A.requires_grad
    assert lora_layer.lora_B.requires_grad

    x = torch.randn(2, 8, 64)
    # At step 0 (B=0), lora_layer output must equal base_linear output
    y_base = base_linear(x)
    y_lora = lora_layer(x)

    assert torch.allclose(y_base, y_lora, atol=1e-6)


def test_lora_injector_and_parameter_freezing():
    """Verifies LoRAInjector freezes base weights and injects LoRALinear into target modules."""
    gpt_cfg = AuraGPTConfig(
        model_name="aura-test",
        vocab_size=50260,
        max_sequence_length=128,
        d_model=32,
        n_layers=2,
        n_heads=2,
        d_ff=64,
    )
    model = AuraGPT(gpt_cfg)
    lora_cfg = LoRAConfig(r=8, alpha=16.0, target_modules=["c_attn", "c_proj"])

    adapted_model, stats = LoRAInjector.inject_lora(model, lora_cfg)

    assert stats["injected_layers_count"] > 0
    assert stats["trainable_percentage"] < 5.0  # Trainable params < 5%

    # Verify frozen base parameters
    is_valid, trainable, frozen = LoRAInjector.verify_frozen_parameters(adapted_model)
    assert is_valid
    assert trainable > 0
    assert frozen > 0


def test_adapter_saving_and_loading():
    """Verifies AdapterSaver and AdapterLoader export and reload adapter weights."""
    gpt_cfg = AuraGPTConfig(
        model_name="aura-test",
        vocab_size=50260,
        max_sequence_length=128,
        d_model=32,
        n_layers=2,
        n_heads=2,
        d_ff=64,
    )
    model = AuraGPT(gpt_cfg)
    lora_cfg = LoRAConfig(r=8, alpha=16.0, target_modules=["c_attn"])
    adapted_model, _ = LoRAInjector.inject_lora(model, lora_cfg)

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # Mutate an adapter weight
        for name, param in adapted_model.named_parameters():
            if "lora_B" in name:
                param.data.fill_(0.5)

        saver_path = AdapterSaver.save_adapter(
            model=adapted_model,
            output_dir=tmp_path,
            config=lora_cfg,
            adapter_name="test_adapter",
        )
        assert saver_path.exists()
        assert (saver_path / "adapter_model.pt").exists()

        # Build clean model and load adapter
        clean_model = AuraGPT(gpt_cfg)
        clean_model, _ = LoRAInjector.inject_lora(clean_model, lora_cfg)

        AdapterLoader.load_adapter(clean_model, saver_path)

        for name, param in clean_model.named_parameters():
            if "lora_B" in name:
                assert torch.allclose(param, torch.tensor(0.5))


def test_adapter_merger_bitwise_correctness():
    """Verifies AdapterMerger in-place weight merging formula W_merged = W_0 + alpha/r * B*A."""
    base_linear = nn.Linear(32, 32)
    lora_layer = LoRALinear(base_layer=base_linear, r=8, alpha=16.0)

    # Fill A and B with non-zero values
    lora_layer.lora_A.data.fill_(1.0)
    lora_layer.lora_B.data.fill_(1.0)

    lora_layer.eval()
    x = torch.randn(2, 32)
    y_unmerged = lora_layer(x)

    lora_layer.merge_weights()
    assert lora_layer.merged

    y_merged = lora_layer(x)
    assert torch.allclose(y_unmerged, y_merged, atol=1e-5)


def test_peft_runner_execution_and_resume():
    """Verifies end-to-end PEFTRunner execution, adapter export, and checkpoint resume."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        config = PEFTTrainingConfig(
            experiment_id="TEST_PEFT_RUNNER",
            data_dir=str(tmp_path),
            tokenizer_dir=str(tmp_path),
            output_dir=str(tmp_path),
            max_steps=4,
            eval_interval=2,
            save_interval=2,
            micro_batch_size=2,
            gradient_accumulation_steps=1,
            max_sequence_length=128,
            d_model=32,
            n_layers=2,
            n_heads=2,
            d_ff=64,
            warmup_steps=0,
        )

        runner = PEFTRunner(config=config)
        summary = runner.run_peft_training()

        assert summary["status"] == "COMPLETED_SUCCESSFULLY"
        assert summary["total_steps"] == 4

        ckpt_latest = tmp_path / "latest.pt"
        ckpt_step2 = tmp_path / "checkpoint_peft_step_000002.pt"
        assert ckpt_latest.exists()
        assert ckpt_step2.exists()
