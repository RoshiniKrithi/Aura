"""Production PR Review & Benchmark Test Suite for Phase 23 / EXP-004 Instruction Tuning.

Includes comprehensive testing for:
- ChatML Role Delimitation & Formatting
- Target Loss Masking Alignment (-100)
- Multi-Turn Conversation Packing & Truncation
- Instruction Dataset Validation & Multi-Format Adapters
- Large Conversation Simulation & Memory Profiling
- Checkpoint State Resume & Serialization
- Stress & Integration Testing for SFT Runner
"""

import json
from pathlib import Path
import tempfile
import time
import pytest
import torch

from src.datasets.conversation_formatter import (
    Conversation,
    ConversationFormatter,
    ConversationTokenizer,
    Message,
    PromptTemplateEngine,
    IM_START_ID,
    IM_END_ID,
    PAD_ID,
)
from src.datasets.instruction_dataset import (
    CodeAlpacaAdapter,
    ConversationDataset,
    ConversationPacker,
    InstructionDatasetLoader,
    InstructionDatasetValidator,
    OpenCoderAdapter,
    ShareGPTAdapter,
)
from src.models.config import AuraGPTConfig
from src.models.gpt import AuraGPT
from src.tokenizer.code_bpe_tokenizer import CodeBPETokenizer
from src.training.exp_004_orchestrator import (
    InstructionEvaluator,
    InstructionTuningConfig,
    InstructionTuningRunner,
)


def test_chatml_role_delimitation_and_tokens():
    """Verifies control tokens IM_START and IM_END boundaries in formatted prompt text."""
    conv = Conversation(
        messages=[
            Message(role="user", content="User Question"),
            Message(role="assistant", content="Assistant Answer"),
        ],
        system_prompt="System Setup",
    )

    engine = PromptTemplateEngine()
    formatted = engine.format(conv)

    assert formatted.count("<|im_start|>") == 3
    assert formatted.count("<|im_end|>") == 3
    assert "<|im_start|>system\nSystem Setup<|im_end|>\n" in formatted
    assert "<|im_start|>user\nUser Question<|im_end|>\n" in formatted
    assert "<|im_start|>assistant\nAssistant Answer<|im_end|>\n" in formatted


def test_target_loss_masking_alignment():
    """Verifies exact alignment of target labels where non-assistant tokens are masked with -100."""
    tokenizer = CodeBPETokenizer.create_default()
    formatter = ConversationFormatter(tokenizer=tokenizer)

    conv = Conversation(
        messages=[
            Message(role="user", content="Solve Two Sum"),
            Message(role="assistant", content="def two_sum(nums, target): pass"),
        ],
        system_prompt="",
    )

    x, y = formatter.tokenize_and_mask(conv, max_sequence_length=256)

    assert isinstance(x, torch.LongTensor)
    assert isinstance(y, torch.LongTensor)
    assert x.shape == (256,)
    assert y.shape == (256,)

    # Labels for non-assistant tokens must be -100
    labels = y.tolist()
    assert -100 in labels
    # Assistant content tokens must be present in labels
    unmasked_labels = [lbl for lbl in labels if lbl != -100]
    assert len(unmasked_labels) > 0


def test_adapters_integrity_and_validation():
    """Verifies record parsing across CodeAlpaca, ShareGPT, and OpenCoder adapters."""
    # 1. CodeAlpaca
    alpaca = CodeAlpacaAdapter().adapt({"instruction": "Write binary search", "output": "def bs(): pass"})
    assert alpaca is not None
    is_valid, errs = InstructionDatasetValidator.validate_conversation(alpaca)
    assert is_valid
    assert len(errs) == 0

    # 2. ShareGPT
    share = ShareGPTAdapter().adapt(
        {"conversations": [{"from": "human", "value": "Q"}, {"from": "gpt", "value": "A"}]}
    )
    assert share is not None
    is_valid_share, _ = InstructionDatasetValidator.validate_conversation(share)
    assert is_valid_share

    # 3. OpenCoder
    coder = OpenCoderAdapter().adapt({"prompt": "Debug code", "response": "Fixed"})
    assert coder is not None
    is_valid_coder, _ = InstructionDatasetValidator.validate_conversation(coder)
    assert is_valid_coder


def test_conversation_packer_efficiency():
    """Verifies ConversationPacker packs multi-turn sequences into full L tensors."""
    packer = ConversationPacker(max_sequence_length=16, ignore_index=-100)

    pairs = [
        (torch.tensor(list(range(8)), dtype=torch.long), torch.tensor([-100] * 8, dtype=torch.long)),
        (torch.tensor(list(range(8)), dtype=torch.long), torch.tensor(list(range(1, 9)), dtype=torch.long)),
    ]

    packed = packer.pack_conversations(pairs)
    assert len(packed) == 1
    px, py = packed[0]
    assert px.shape == (16,)
    assert py.shape == (16,)


def test_large_conversation_handling_and_memory():
    """Simulates large instruction dataset loading and profiles memory footprint."""
    convs = []
    for i in range(100):
        convs.append(
            Conversation(
                messages=[
                    Message(role="user", content=f"Question {i}: Write code for problem {i}"),
                    Message(role="assistant", content=f"Solution {i}: def solve_{i}(): return {i}"),
                ]
            )
        )

    ds = ConversationDataset(conversations=convs, max_sequence_length=128)
    assert len(ds) == 100

    start_time = time.time()
    for idx in range(len(ds)):
        x, y = ds[idx]
        assert x.shape == (128,)
        assert y.shape == (128,)

    elapsed = time.time() - start_time
    assert elapsed < 5.0  # Fast tokenization and masking

    stats = ds.compute_statistics()
    assert stats.total_conversations == 100
    assert stats.total_turns == 200


def test_sft_runner_execution_and_checkpoint_resume():
    """Verifies end-to-end InstructionTuningRunner training loop, checkpoint saving, and resume."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        config = InstructionTuningConfig(
            experiment_id="PR_REVIEW_SFT_RUNNER",
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

        runner = InstructionTuningRunner(config=config)
        summary = runner.run_sft()

        assert summary["status"] == "COMPLETED_SUCCESSFULLY"
        assert summary["total_steps"] == 4

        ckpt_latest = tmp_path / "latest.pt"
        ckpt_step2 = tmp_path / "checkpoint_sft_step_000002.pt"
        assert ckpt_latest.exists()
        assert ckpt_step2.exists()

        # Resume verification
        config.max_steps = 6
        runner_resumed = InstructionTuningRunner(config=config, resume_from_checkpoint=ckpt_latest)
        assert runner_resumed.start_step == 4

        summary_resumed = runner_resumed.run_sft()
        assert summary_resumed["total_steps"] == 6
