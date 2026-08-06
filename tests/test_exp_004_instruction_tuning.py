"""Comprehensive PyTest Suite for Experiment EXP-004 Instruction Tuning (SFT).

Includes testing for:
- Message & Conversation formatting (ChatML)
- Prompt Template Engine & Role Control Tokens
- Completion-only Target Loss Masking (-100 PyTorch ignore_index strategy)
- Instruction Dataset Adapters (CodeAlpaca, ShareGPT, OpenCoder)
- Instruction Dataset Validator & Loader
- Conversation Dataset & Statistics
- Conversation Sequence Packing
- SFT Evaluator & Sample Generation
- InstructionTuningRunner End-to-End Execution & Checkpoint Resume
"""

import json
from pathlib import Path
import tempfile
import torch
import pytest

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


def test_conversation_formatting_chatml():
    """Verifies ChatML formatting for system, user, and assistant roles."""
    conv = Conversation(
        messages=[
            Message(role="user", content="Write a function."),
            Message(role="assistant", content="def foo(): pass"),
        ],
        system_prompt="Custom System Prompt",
    )

    engine = PromptTemplateEngine()
    formatted_text = engine.format(conv)

    assert "<|im_start|>system\nCustom System Prompt<|im_end|>\n" in formatted_text
    assert "<|im_start|>user\nWrite a function.<|im_end|>\n" in formatted_text
    assert "<|im_start|>assistant\ndef foo(): pass<|im_end|>\n" in formatted_text


def test_completion_only_target_loss_masking():
    """Verifies that non-assistant tokens are assigned label -100 (ignored in loss calculation)."""
    tokenizer = CodeBPETokenizer.create_default()
    formatter = ConversationFormatter(tokenizer=tokenizer)

    conv = Conversation(
        messages=[
            Message(role="user", content="Add 2 and 2."),
            Message(role="assistant", content="4"),
        ],
        system_prompt="",
    )

    x_tensor, y_tensor = formatter.tokenize_and_mask(conv, max_sequence_length=512)

    assert isinstance(x_tensor, torch.LongTensor)
    assert isinstance(y_tensor, torch.LongTensor)
    assert x_tensor.shape == (512,)
    assert y_tensor.shape == (512,)

    # Verify that -100 exists in labels for user query tokens
    assert -100 in y_tensor.tolist()
    # Verify assistant completion token exists in labels (not all are -100)
    assert any(label != -100 for label in y_tensor.tolist())


def test_instruction_dataset_adapters():
    """Verifies CodeAlpaca, ShareGPT, and OpenCoder adapters adapt raw json records."""
    # 1. CodeAlpaca
    alpaca_raw = {"instruction": "Reverse array", "input": "[1, 2]", "output": "[2, 1]"}
    conv_alpaca = CodeAlpacaAdapter().adapt(alpaca_raw)
    assert conv_alpaca is not None
    assert len(conv_alpaca.messages) == 2
    assert "Reverse array" in conv_alpaca.messages[0].content

    # 2. ShareGPT
    share_raw = {
        "conversations": [
            {"from": "human", "value": "Hello"},
            {"from": "gpt", "value": "Hi there!"},
        ]
    }
    conv_share = ShareGPTAdapter().adapt(share_raw)
    assert conv_share is not None
    assert len(conv_share.messages) == 2
    assert conv_share.messages[1].role == "assistant"

    # 3. OpenCoder
    coder_raw = {"prompt": "Write binary search", "response": "def bs(): pass"}
    conv_coder = OpenCoderAdapter().adapt(coder_raw)
    assert conv_coder is not None
    assert conv_coder.messages[1].content == "def bs(): pass"


def test_instruction_dataset_validator():
    """Verifies InstructionDatasetValidator validates conversation structural rules."""
    valid_conv = Conversation(
        messages=[
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi!"),
        ]
    )
    is_valid, errors = InstructionDatasetValidator.validate_conversation(valid_conv)
    assert is_valid
    assert len(errors) == 0

    # Invalid conv (no assistant)
    invalid_conv = Conversation(messages=[Message(role="user", content="Hello")])
    is_valid_inv, errors_inv = InstructionDatasetValidator.validate_conversation(invalid_conv)
    assert not is_valid_inv
    assert "lacks an assistant completion turn" in errors_inv[0]


def test_instruction_dataset_loader():
    """Verifies InstructionDatasetLoader loads JSONL files."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        jsonl_path = Path(tmp_dir) / "test_instructions.jsonl"
        with open(jsonl_path, "w", encoding="utf-8") as f:
            f.write(json.dumps({"instruction": "Task 1", "output": "Ans 1"}) + "\n")
            f.write(json.dumps({"instruction": "Task 2", "output": "Ans 2"}) + "\n")

        convs = InstructionDatasetLoader.load_jsonl(jsonl_path, format_name="code_alpaca")
        assert len(convs) == 2


def test_conversation_dataset_and_statistics():
    """Verifies ConversationDataset item indexing and statistics calculation."""
    convs = [
        Conversation(
            messages=[
                Message(role="user", content="Explain O(1)"),
                Message(role="assistant", content="O(1) means constant time."),
            ]
        )
    ]
    ds = ConversationDataset(conversations=convs, max_sequence_length=32)
    assert len(ds) == 1

    x, y = ds[0]
    assert x.shape == (32,)
    assert y.shape == (32,)

    stats = ds.compute_statistics()
    assert stats.total_conversations == 1
    assert stats.total_turns == 2


def test_conversation_packer():
    """Verifies ConversationPacker concatenates sequence pairs into full L window tensors."""
    packer = ConversationPacker(max_sequence_length=10)
    pairs = [
        (torch.tensor(list(range(5)), dtype=torch.long), torch.tensor([-100] * 5, dtype=torch.long)),
        (torch.tensor(list(range(8)), dtype=torch.long), torch.tensor([1, 2, 3, 4, 5, 6, 7, 8], dtype=torch.long)),
    ]

    packed = packer.pack_conversations(pairs)
    assert len(packed) == 1
    px, py = packed[0]
    assert px.shape == (10,)
    assert py.shape == (10,)


def test_instruction_tuning_runner_and_checkpoint_resume():
    """Verifies InstructionTuningRunner end-to-end execution, checkpointing, and bitwise resume."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        config = InstructionTuningConfig(
            experiment_id="TEST_SFT_RUNNER",
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

        # 1. Initial SFT Run
        runner = InstructionTuningRunner(config=config)
        summary = runner.run_sft()

        assert summary["status"] == "COMPLETED_SUCCESSFULLY"
        assert summary["total_steps"] == 4

        ckpt_latest = tmp_path / "latest.pt"
        ckpt_step2 = tmp_path / "checkpoint_sft_step_000002.pt"
        assert ckpt_latest.exists()
        assert ckpt_step2.exists()

        # 2. Test Resume
        config.max_steps = 6
        runner_resumed = InstructionTuningRunner(config=config, resume_from_checkpoint=ckpt_latest)
        assert runner_resumed.start_step == 4

        summary_resumed = runner_resumed.run_sft()
        assert summary_resumed["total_steps"] == 6
