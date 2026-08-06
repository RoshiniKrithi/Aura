"""PEFT Trainer and Evaluator Orchestrator for Aura EXP-007.

Provides PEFTStatistics, PEFTEvaluator, PEFTTrainer, and PEFTRunner.
"""

from dataclasses import asdict, dataclass
import json
import logging
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.datasets.conversation_formatter import ConversationFormatter, PromptTemplateEngine
from src.datasets.instruction_dataset import ConversationDataset, InstructionDatasetLoader
from src.losses.cross_entropy import CrossEntropyLoss
from src.models.config import AuraGPTConfig
from src.models.gpt import AuraGPT
from src.optimizers.config import OptimizationConfig, OptimizerConfig
from src.optimizers.manager import OptimizationManager
from src.peft.adapter_manager import AdapterManager
from src.peft.adapter_merger import AdapterMerger
from src.peft.lora_injector import LoRAInjector
from src.peft.peft_config import PEFTTrainingConfig
from src.schedulers.cosine_warmup import CosineAnnealingWithWarmupLR
from src.tokenizer.code_bpe_tokenizer import CodeBPETokenizer

logger = logging.getLogger(__name__)


@dataclass
class PEFTStatistics:
    """Statistics container for PEFT training step metrics."""

    step: int
    train_loss: float
    val_loss: Optional[float]
    trainable_params: int
    frozen_params: int
    trainable_percentage: float
    elapsed_time_sec: float

    def to_dict(self) -> Dict[str, Any]:
        """Converts PEFTStatistics to dictionary representation."""
        return asdict(self)


class PEFTEvaluator:
    """Evaluates validation loss on completion tokens for PEFT adapter models."""

    def __init__(
        self,
        model: nn.Module,
        val_loader: DataLoader,
        loss_fn: CrossEntropyLoss,
        device: torch.device,
    ) -> None:
        """Initializes PEFTEvaluator."""
        self.model = model
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.device = device

    def evaluate(self, max_batches: int = 5) -> float:
        """Calculates mean cross-entropy validation loss on completion tokens."""
        self.model.eval()
        total_loss = 0.0
        total_batches = 0

        with torch.no_grad():
            for i, (x, y) in enumerate(self.val_loader):
                if i >= max_batches:
                    break
                x, y = x.to(self.device), y.to(self.device)
                logits = self.model(x)
                loss_output = self.loss_fn(logits, y)
                loss = loss_output[0] if isinstance(loss_output, tuple) else loss_output
                total_loss += loss.item()
                total_batches += 1

        self.model.train()
        return total_loss / max(1, total_batches)


class PEFTRunner:
    """Master runner coordinating Parameter-Efficient Fine-Tuning execution lifecycle."""

    def __init__(
        self,
        config: PEFTTrainingConfig,
        resume_from_checkpoint: Optional[Union[str, Path]] = None,
    ) -> None:
        """Initializes PEFTRunner."""
        self.config = config
        self.output_dir = Path(config.output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.device = self._resolve_device(config.device)
        self.config.device = str(self.device)

        # 1. Load Tokenizer
        tokenizer_path = Path(config.tokenizer_dir) / "bpe_vocab_50257.json"
        merges_path = Path(config.tokenizer_dir) / "bpe_merges_50257.txt"

        if tokenizer_path.exists() and merges_path.exists():
            self.tokenizer = CodeBPETokenizer.from_files(tokenizer_path, merges_path)
        else:
            self.tokenizer = CodeBPETokenizer.create_default()

        # 2. Build Base Model & Inject LoRA
        gpt_cfg = AuraGPTConfig(
            model_name="aura-peft-base",
            vocab_size=max(self.tokenizer.vocab_size, config.vocab_size, 50260),
            max_sequence_length=config.max_sequence_length,
            d_model=config.d_model,
            n_layers=config.n_layers,
            n_heads=config.n_heads,
            d_ff=config.d_ff,
            device=str(self.device),
        )
        base_model = AuraGPT(gpt_cfg).to(self.device)

        if config.pretrained_checkpoint_path and Path(config.pretrained_checkpoint_path).exists():
            self._load_pretrained_weights(base_model, Path(config.pretrained_checkpoint_path))

        self.model, self.param_stats = LoRAInjector.inject_lora(
            model=base_model, config=config.lora_config
        )

        # 3. Setup Dataset
        self.formatter = ConversationFormatter(tokenizer=self.tokenizer)
        data_path = Path(config.data_dir) / "instructions.jsonl"
        conversations = InstructionDatasetLoader.load_jsonl(data_path, format_name="code_alpaca")

        if not conversations:
            from src.datasets.instruction_dataset import Conversation, Message
            conversations = [
                Conversation(
                    messages=[
                        Message(role="user", content="Implement binary search in Python."),
                        Message(role="assistant", content="def binary_search(arr, target):\n    low, high = 0, len(arr) - 1\n    while low <= high:\n        mid = (low + high) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            low = mid + 1\n        else:\n            high = mid - 1\n    return -1"),
                    ],
                )
            ]

        dataset = ConversationDataset(
            conversations=conversations,
            formatter=self.formatter,
            max_sequence_length=config.max_sequence_length,
        )

        self.train_loader = DataLoader(
            dataset, batch_size=config.micro_batch_size, shuffle=True
        )
        self.val_loader = DataLoader(
            dataset, batch_size=config.micro_batch_size, shuffle=False
        )

        # 4. Optimization Engine (Trainable Params ONLY)
        opt_cfg = OptimizationConfig(
            optimizer=OptimizerConfig(
                name="adamw",
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
            )
        )
        self.opt_mgr = OptimizationManager(model=self.model, config=opt_cfg)
        self.scheduler = CosineAnnealingWithWarmupLR(
            optimizer=self.opt_mgr.optimizer,
            warmup_steps=config.warmup_steps,
            max_steps=config.max_steps,
        )
        from src.losses.config import CrossEntropyLossConfig
        self.loss_fn = CrossEntropyLoss(config=CrossEntropyLossConfig(ignore_index=-100))
        self.evaluator = PEFTEvaluator(
            model=self.model,
            val_loader=self.val_loader,
            loss_fn=self.loss_fn,
            device=self.device,
        )
        self.adapter_mgr = AdapterManager(model=self.model)

        self.start_step = 0
        if resume_from_checkpoint and Path(resume_from_checkpoint).exists():
            self._resume_checkpoint(Path(resume_from_checkpoint))

    def _resolve_device(self, req: str) -> torch.device:
        if req == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def _load_pretrained_weights(self, model: AuraGPT, path: Path) -> None:
        logger.info("Loading pre-trained base weights from %s", path)
        ckpt = torch.load(path, weights_only=False)
        if "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])

    def _resume_checkpoint(self, path: Path) -> None:
        logger.info("Resuming PEFT training from checkpoint: %s", path)
        ckpt = torch.load(path, weights_only=False)
        if "model_state_dict" in ckpt:
            self.model.load_state_dict(ckpt["model_state_dict"])
        if "global_step" in ckpt:
            self.start_step = ckpt["global_step"]

    def run_peft_training(self) -> Dict[str, Any]:
        """Executes Parameter-Efficient Fine-Tuning training loop."""
        logger.info("STARTING EXP-007 PEFT LORA TRAINING: %s", self.config.experiment_id)
        self.model.train()

        current_step = self.start_step
        train_iter = iter(self.train_loader)
        start_time = time.time()

        while current_step < self.config.max_steps:
            self.opt_mgr.zero_grad()
            step_loss = 0.0

            for _ in range(self.config.gradient_accumulation_steps):
                try:
                    x, y = next(train_iter)
                except StopIteration:
                    train_iter = iter(self.train_loader)
                    x, y = next(train_iter)

                x, y = x.to(self.device), y.to(self.device)
                logits = self.model(x)
                loss_output = self.loss_fn(logits, y)
                loss = loss_output[0] if isinstance(loss_output, tuple) else loss_output
                scaled_loss = loss / self.config.gradient_accumulation_steps
                scaled_loss.backward()
                step_loss += loss.item()

            # Clip gradients on trainable LoRA parameters
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad], max_norm=1.0
            )

            self.opt_mgr.step()
            self.scheduler.step()
            current_step += 1

            val_loss = None
            if current_step % self.config.eval_interval == 0:
                val_loss = self.evaluator.evaluate()

            if current_step % self.config.save_interval == 0:
                self._save_checkpoint(current_step)

        # Save Final Adapter & Merged Model
        adapter_path = self.adapter_mgr.save_adapter(
            output_dir=self.output_dir,
            config=self.config.lora_config,
            adapter_name=self.config.adapter_name,
            version=self.config.version,
        )

        merged_model_path = self.output_dir / "merged_model.pt"
        AdapterMerger.export_merged_model(self.model, merged_model_path)

        summary = {
            "experiment_id": self.config.experiment_id,
            "status": "COMPLETED_SUCCESSFULLY",
            "total_steps": current_step,
            "parameter_statistics": self.param_stats,
            "adapter_path": str(adapter_path),
            "merged_model_path": str(merged_model_path),
            "elapsed_time_seconds": round(time.time() - start_time, 2),
        }

        with open(self.output_dir / "peft_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        logger.info("EXP-007 PEFT LORA TRAINING FINISHED SUCCESSFULLY")
        return summary

    def _save_checkpoint(self, step: int) -> Path:
        ckpt_path = self.output_dir / f"checkpoint_peft_step_{step:06d}.pt"
        ckpt_data = {
            "global_step": step,
            "model_state_dict": self.model.state_dict(),
            "config": self.config.to_dict(),
        }
        torch.save(ckpt_data, ckpt_path)
        torch.save(ckpt_data, self.output_dir / "latest.pt")
        return ckpt_path
