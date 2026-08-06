#!/usr/bin/env python3
"""Aura System Validation & Pre-Flight Health Check Engine.

Executes a 13-stage verification suite validating project structure, configuration,
tokenizer, dataset, model initialization, forward pass, cross-entropy loss, backward pass,
optimizer steps, learning rate schedule, checkpointing, autoregressive inference,
and automated pytest execution before training. Automatically generates reports/validation_report.md.
"""

from dataclasses import dataclass, field
import datetime
import importlib.util
import json
import logging
import os
from pathlib import Path
import platform
import subprocess
import sys

# Ensure workspace root is in sys.path for direct script execution
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import datetime
import importlib.util
import json
import logging
import tempfile
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

# Import Aura internal subsystems
from src.datasets.text_dataset import AuraTextDataset
from src.inference.engine import InferenceEngine
from src.losses.cross_entropy import CrossEntropyLoss
from src.models.config import AuraGPTConfig
from src.models.gpt import AuraGPT
from src.optimizers.config import OptimizationConfig, OptimizerConfig, SchedulerConfig
from src.optimizers.manager import OptimizationManager
from src.tokenizer.char_tokenizer import CharacterTokenizer
from src.utils.config_loader import ConfigLoader

logger = logging.getLogger("aura.validator")


@dataclass
class CheckResult:
    """Dataclass holding validation check outcomes."""

    name: str
    passed: bool
    details: str = ""
    warning: str = ""
    failure_reason: str = ""
    suggested_fix: str = ""
    file_involved: str = ""
    root_cause: str = ""


@dataclass
class ValidationReport:
    """Dataclass aggregating system validation report data."""

    timestamp: str
    python_version: str
    pytorch_version: str
    cuda_available: bool
    device_name: str
    git_hash: str
    results: List[CheckResult] = field(default_factory=list)

    @property
    def total_passed(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def total_failed(self) -> int:
        return sum(1 for r in self.results if not r.passed)

    @property
    def total_warnings(self) -> int:
        return sum(1 for r in self.results if r.warning)

    @property
    def is_ready_for_training(self) -> bool:
        return self.total_failed == 0


class AuraSystemValidator:
    """Pre-flight validation engine performing complete end-to-end system checks."""

    def __init__(self, root_dir: Optional[Path] = None) -> None:
        """Initializes system validator binding workspace root directory.

        Args:
            root_dir: Optional root directory Path. Defaults to current working directory.
        """
        self.root_dir = Path(root_dir or os.getcwd()).resolve()
        self.reports_dir = self.root_dir / "reports"
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        self.results: List[CheckResult] = []
        self.app_config: Optional[Any] = None
        self.tokenizer: Optional[Any] = None
        self.dataset: Optional[Any] = None
        self.model: Optional[Any] = None
        self.loss_module: Optional[Any] = None
        self.opt_manager: Optional[Any] = None

    def run_all_checks(self) -> ValidationReport:
        """Executes all 13 validation checks sequentially in order.

        Returns:
            ValidationReport object summarizing all check results.
        """
        checks = [
            ("Project Structure", self.check_project_structure),
            ("Configuration", self.check_configuration),
            ("Tokenizer", self.check_tokenizer),
            ("Dataset", self.check_dataset),
            ("Model Initialization", self.check_model_initialization),
            ("Forward Pass", self.check_forward_pass),
            ("Loss", self.check_loss_computation),
            ("Backward Pass", self.check_backward_pass),
            ("Optimizer", self.check_optimizer_step),
            ("Scheduler", self.check_scheduler_step),
            ("Checkpoint", self.check_checkpoint_serialization),
            ("Inference", self.check_inference_generation),
            ("Unit Tests", self.check_unit_tests),
        ]

        for name, check_fn in checks:
            try:
                res = check_fn()
                self.results.append(res)
            except Exception as e:
                self.results.append(
                    CheckResult(
                        name=name,
                        passed=False,
                        failure_reason=str(e),
                        suggested_fix="Inspect unexpected exception stack trace and fix underlying error.",
                        file_involved=__file__,
                        root_cause=f"Unhandled exception in {name} check: {type(e).__name__}",
                    )
                )

        report = self._build_report()
        self.generate_markdown_report(report)
        return report

    def check_project_structure(self) -> CheckResult:
        """Check 1: Verifies all required workspace directories exist."""
        required_dirs = [
            "src",
            "tests",
            "configs",
            "scripts",
            "logs",
            "checkpoints",
            "outputs",
        ]
        missing: List[str] = []

        for folder in required_dirs:
            p = self.root_dir / folder
            if not p.exists():
                # Auto-create runtime directories if missing
                if folder in ["logs", "checkpoints", "outputs"]:
                    p.mkdir(parents=True, exist_ok=True)
                else:
                    missing.append(folder)

        if missing:
            return CheckResult(
                name="Project Structure",
                passed=False,
                failure_reason=f"Missing required workspace directories: {missing}",
                suggested_fix=f"Create missing directories using 'mkdir {', '.join(missing)}'.",
                file_involved=str(self.root_dir),
                root_cause="Incomplete project workspace layout.",
            )

        return CheckResult(
            name="Project Structure",
            passed=True,
            details=f"All {len(required_dirs)} required directories verified.",
        )

    def check_configuration(self) -> CheckResult:
        """Check 2: Loads and validates YAML configuration schema."""
        config_path = self.root_dir / "configs" / "config.yaml"
        if not config_path.exists():
            return CheckResult(
                name="Configuration",
                passed=False,
                failure_reason=f"Configuration file not found: {config_path}",
                suggested_fix="Ensure configs/config.yaml exists in workspace root.",
                file_involved=str(config_path),
                root_cause="Missing configuration file.",
            )

        try:
            self.app_config = ConfigLoader.from_yaml(config_path)

            # Validate key attributes and types
            model_cfg = self.app_config.model
            assert isinstance(model_cfg.max_sequence_length, int), "max_sequence_length must be int"
            assert isinstance(model_cfg.d_model, int), "d_model must be int"
            assert isinstance(model_cfg.n_layers, int), "n_layers must be int"
            assert isinstance(model_cfg.n_heads, int), "n_heads must be int"

            return CheckResult(
                name="Configuration",
                passed=True,
                details=f"Loaded config '{model_cfg.name}': d_model={model_cfg.d_model}, n_layers={model_cfg.n_layers}.",
            )
        except Exception as e:
            return CheckResult(
                name="Configuration",
                passed=False,
                failure_reason=str(e),
                suggested_fix="Check configs/config.yaml syntax and schema field names.",
                file_involved=str(config_path),
                root_cause=f"Configuration validation error: {type(e).__name__}",
            )

    def check_tokenizer(self) -> CheckResult:
        """Check 3: Verifies tokenizer encoding and decoding roundtrips."""
        sample_text = "def binary_search(arr, target):\n    return -1\n"
        try:
            self.tokenizer = CharacterTokenizer.from_corpus(sample_text)
            token_ids = self.tokenizer.encode(sample_text)
            decoded_text = self.tokenizer.decode(token_ids)

            if decoded_text != sample_text:
                return CheckResult(
                    name="Tokenizer",
                    passed=False,
                    failure_reason=f"Decoded text mismatch. Expected '{sample_text}', got '{decoded_text}'",
                    suggested_fix="Fix CharacterTokenizer decode index mapping logic.",
                    file_involved="src/tokenizer/char_tokenizer.py",
                    root_cause="Lossy roundtrip token encoding/decoding.",
                )

            return CheckResult(
                name="Tokenizer",
                passed=True,
                details=f"CharacterTokenizer verified. Vocab size: {self.tokenizer.vocab_size}, Tokens: {len(token_ids)}.",
            )
        except Exception as e:
            return CheckResult(
                name="Tokenizer",
                passed=False,
                failure_reason=str(e),
                suggested_fix="Check tokenizer module initialization.",
                file_involved="src/tokenizer/char_tokenizer.py",
                root_cause=f"Tokenizer error: {type(e).__name__}",
            )

    def check_dataset(self) -> CheckResult:
        """Check 4: Validates dataset sequence slicing and item retrieval."""
        try:
            sample_corpus = ("ROMEO: Hear me speak.\nJULIET: Speak on.\n") * 20
            tokens = self.tokenizer.encode(sample_corpus)

            self.dataset = AuraTextDataset(
                token_ids=tokens,
                window_size=32,
                stride=32,
                name="ValidationDS",
            )

            if len(self.dataset) <= 0:
                return CheckResult(
                    name="Dataset",
                    passed=False,
                    failure_reason="Dataset length is zero.",
                    suggested_fix="Ensure token count exceeds window_size.",
                    file_involved="src/datasets/text_dataset.py",
                    root_cause="Empty sequence array.",
                )

            x, y = self.dataset[0]
            assert isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor), "Item outputs must be PyTorch Tensors"
            assert x.dtype == torch.long and y.dtype == torch.long, "Tensor dtype must be torch.long"
            assert x.shape == (32,) and y.shape == (32,), f"Expected shape (32,), got X:{x.shape}, Y:{y.shape}"

            return CheckResult(
                name="Dataset",
                passed=True,
                details=f"AuraTextDataset verified. Length: {len(self.dataset)} sequences, Window: {x.shape[0]}.",
            )
        except Exception as e:
            return CheckResult(
                name="Dataset",
                passed=False,
                failure_reason=str(e),
                suggested_fix="Inspect dataset sequence construction logic.",
                file_involved="src/datasets/text_dataset.py",
                root_cause=f"Dataset validation failure: {type(e).__name__}",
            )

    def check_model_initialization(self) -> CheckResult:
        """Check 5: Initializes AuraGPT model and validates parameter stats."""
        try:
            gpt_config = AuraGPTConfig(
                model_name="aura-tiny",
                vocab_size=self.tokenizer.vocab_size,
                max_sequence_length=128,
                d_model=128,
                n_layers=2,
                n_heads=2,
                d_ff=512,
                dropout=0.1,
                device="cpu",
            )

            self.model = AuraGPT(config=gpt_config)
            num_params = self.model.get_num_params()

            if num_params <= 0:
                return CheckResult(
                    name="Model Initialization",
                    passed=False,
                    failure_reason="Model parameter count is zero or negative.",
                    suggested_fix="Check AuraGPT submodule layer initializations.",
                    file_involved="src/models/gpt.py",
                    root_cause="Uninitialized model parameters.",
                )

            return CheckResult(
                name="Model Initialization",
                passed=True,
                details=f"AuraGPT initialized. Total trainable parameters: {num_params:,}.",
            )
        except Exception as e:
            return CheckResult(
                name="Model Initialization",
                passed=False,
                failure_reason=str(e),
                suggested_fix="Check model architecture configuration parameters.",
                file_involved="src/models/gpt.py",
                root_cause=f"Model initialization failure: {type(e).__name__}",
            )

    def check_forward_pass(self) -> CheckResult:
        """Check 6: Runs forward pass and validates output tensor numerical health."""
        try:
            dummy_inputs = torch.randint(0, self.tokenizer.vocab_size, (2, 32))
            logits = self.model(dummy_inputs)

            expected_shape = (2, 32, self.tokenizer.vocab_size)
            if logits.shape != expected_shape:
                return CheckResult(
                    name="Forward Pass",
                    passed=False,
                    failure_reason=f"Logits shape mismatch. Expected {expected_shape}, got {logits.shape}",
                    suggested_fix="Check LM head linear projection shape.",
                    file_involved="src/models/gpt.py",
                    root_cause="Incorrect tensor shape projection.",
                )

            if torch.isnan(logits).any() or torch.isinf(logits).any():
                return CheckResult(
                    name="Forward Pass",
                    passed=False,
                    failure_reason="Logits tensor contains NaN or Inf values.",
                    suggested_fix="Inspect weight initializations and scale operations.",
                    file_involved="src/models/gpt.py",
                    root_cause="Numerical instability in forward propagation.",
                )

            return CheckResult(
                name="Forward Pass",
                passed=True,
                details=f"Forward pass successful. Output Logits Shape: {tuple(logits.shape)}.",
            )
        except Exception as e:
            return CheckResult(
                name="Forward Pass",
                passed=False,
                failure_reason=str(e),
                suggested_fix="Debug AuraGPT.forward() execution.",
                file_involved="src/models/gpt.py",
                root_cause=f"Forward pass exception: {type(e).__name__}",
            )

    def check_loss_computation(self) -> CheckResult:
        """Check 7: Computes cross-entropy loss and verifies finiteness."""
        try:
            self.loss_module = CrossEntropyLoss()
            dummy_inputs = torch.randint(0, self.tokenizer.vocab_size, (2, 32))
            dummy_targets = torch.randint(0, self.tokenizer.vocab_size, (2, 32))

            logits = self.model(dummy_inputs)
            loss, metrics = self.loss_module(logits, dummy_targets)

            loss_val = loss.item()
            if not torch.isfinite(loss) or loss_val <= 0.0:
                return CheckResult(
                    name="Loss",
                    passed=False,
                    failure_reason=f"Invalid loss value: {loss_val}",
                    suggested_fix="Check cross entropy loss computation log-softmax stability.",
                    file_involved="src/losses/cross_entropy.py",
                    root_cause="Non-finite or non-positive loss output.",
                )

            return CheckResult(
                name="Loss",
                passed=True,
                details=f"CrossEntropyLoss verified. Loss: {loss_val:.4f} nats, Perplexity: {metrics.get('perplexity', 0):.2f}.",
            )
        except Exception as e:
            return CheckResult(
                name="Loss",
                passed=False,
                failure_reason=str(e),
                suggested_fix="Check CrossEntropyLoss.forward() implementation.",
                file_involved="src/losses/cross_entropy.py",
                root_cause=f"Loss computation error: {type(e).__name__}",
            )

    def check_backward_pass(self) -> CheckResult:
        """Check 8: Executes backward pass and validates parameter gradients."""
        try:
            self.model.zero_grad(set_to_none=True)
            dummy_inputs = torch.randint(0, self.tokenizer.vocab_size, (2, 32))
            dummy_targets = torch.randint(0, self.tokenizer.vocab_size, (2, 32))

            logits = self.model(dummy_inputs)
            loss, _ = self.loss_module(logits, dummy_targets)
            loss.backward()

            missing_grads = 0
            invalid_grads = 0
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    if param.grad is None:
                        missing_grads += 1
                    elif torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                        invalid_grads += 1

            if missing_grads > 0 or invalid_grads > 0:
                return CheckResult(
                    name="Backward Pass",
                    passed=False,
                    failure_reason=f"Gradient checks failed: missing={missing_grads}, invalid={invalid_grads}",
                    suggested_fix="Ensure autograd graph continuity across model blocks.",
                    file_involved="src/models/gpt.py",
                    root_cause="Broken backpropagation flow or gradient exploding.",
                )

            return CheckResult(
                name="Backward Pass",
                passed=True,
                details="Backward pass verified. All parameter gradients exist and are finite.",
            )
        except Exception as e:
            return CheckResult(
                name="Backward Pass",
                passed=False,
                failure_reason=str(e),
                suggested_fix="Inspect autograd graph step.",
                file_involved="src/models/gpt.py",
                root_cause=f"Backward pass exception: {type(e).__name__}",
            )

    def check_optimizer_step(self) -> CheckResult:
        """Check 9: Performs optimizer step and verifies parameter updates."""
        try:
            opt_cfg = OptimizationConfig(
                optimizer=OptimizerConfig(name="adamw", lr=1.0e-3),
                scheduler=SchedulerConfig(name="cosine_warmup", warmup_steps=0, max_steps=100),
                max_grad_norm=1.0,
                gradient_accumulation_steps=1,
            )

            self.opt_manager = OptimizationManager(model=self.model, config=opt_cfg)

            # 1. Populate gradients via forward and backward pass
            self.model.zero_grad(set_to_none=True)
            dummy_inputs = torch.randint(0, self.tokenizer.vocab_size, (2, 32))
            dummy_targets = torch.randint(0, self.tokenizer.vocab_size, (2, 32))
            logits = self.model(dummy_inputs)
            loss, _ = self.loss_module(logits, dummy_targets)
            loss.backward()

            # 2. Store baseline weights across all parameters
            p_old_list = [p.clone().detach() for p in self.model.parameters()]

            # 3. Execute step at micro_step=1 (where LR > 0 during warmup)
            did_step, current_lr, grad_norm = self.opt_manager.step(micro_step=1)

            p_new_list = [p.clone().detach() for p in self.model.parameters()]

            weights_changed = any(not torch.equal(p1, p2) for p1, p2 in zip(p_old_list, p_new_list))

            if not weights_changed:
                return CheckResult(
                    name="Optimizer",
                    passed=False,
                    failure_reason="Model weights did not change after optimizer.step()",
                    suggested_fix="Verify optimizer parameter group gradients and learning rate.",
                    file_involved="src/optimizers/manager.py",
                    root_cause="Zero parameter update or frozen gradients.",
                )

            return CheckResult(
                name="Optimizer",
                passed=True,
                details=f"Optimizer step verified. Weight update detected (GradNorm: {grad_norm:.4f}, LR: {current_lr:.3e}).",
            )
        except Exception as e:
            return CheckResult(
                name="Optimizer",
                passed=False,
                failure_reason=str(e),
                suggested_fix="Inspect OptimizationManager.step() method.",
                file_involved="src/optimizers/manager.py",
                root_cause=f"Optimizer execution failure: {type(e).__name__}",
            )

    def check_scheduler_step(self) -> CheckResult:
        """Check 10: Runs learning rate scheduler step and verifies schedule trajectory."""
        try:
            initial_lr = self.opt_manager.get_lr()

            # Run 5 steps to trigger scheduler warmup update
            for i in range(1, 6):
                # Trigger backward to populate dummy gradients
                dummy_inputs = torch.randint(0, self.tokenizer.vocab_size, (2, 32))
                dummy_targets = torch.randint(0, self.tokenizer.vocab_size, (2, 32))
                logits = self.model(dummy_inputs)
                loss, _ = self.loss_module(logits, dummy_targets)
                loss.backward()
                self.opt_manager.step(micro_step=i)

            updated_lr = self.opt_manager.get_lr()

            if updated_lr == initial_lr:
                return CheckResult(
                    name="Scheduler",
                    passed=False,
                    failure_reason=f"Learning rate did not update after 5 scheduler steps (LR={updated_lr})",
                    suggested_fix="Check scheduler warmup logic and step calls.",
                    file_involved="src/schedulers/cosine_warmup.py",
                    root_cause="Static learning rate schedule.",
                )

            return CheckResult(
                name="Scheduler",
                passed=True,
                details=f"Scheduler step verified. Warmup LR updated from {initial_lr:.3e} to {updated_lr:.3e}.",
            )
        except Exception as e:
            return CheckResult(
                name="Scheduler",
                passed=False,
                failure_reason=str(e),
                suggested_fix="Check OptimizationManager scheduler integration.",
                file_involved="src/optimizers/manager.py",
                root_cause=f"Scheduler step failure: {type(e).__name__}",
            )

    def check_checkpoint_serialization(self) -> CheckResult:
        """Check 11: Saves and reloads model checkpoint, verifying weight identity."""
        try:
            with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp_file:
                tmp_ckpt_path = Path(tmp_file.name)

            # Save checkpoint state dict
            state_dict = {
                "model_state_dict": self.model.state_dict(),
                "opt_state_dict": self.opt_manager.state_dict(),
                "global_step": 10,
            }
            torch.save(state_dict, tmp_ckpt_path)

            # Instantiate new model and load state
            reloaded_model = AuraGPT(config=self.model.config)
            loaded_checkpoint = torch.load(tmp_ckpt_path, weights_only=False)
            reloaded_model.load_state_dict(loaded_checkpoint["model_state_dict"])

            # Clean up temp file
            if tmp_ckpt_path.exists():
                tmp_ckpt_path.unlink()

            # Verify parameter equality
            for p1, p2 in zip(self.model.parameters(), reloaded_model.parameters()):
                if not torch.equal(p1, p2):
                    return CheckResult(
                        name="Checkpoint",
                        passed=False,
                        failure_reason="Reloaded checkpoint parameters do not match original model weights.",
                        suggested_fix="Inspect checkpoint state_dict save/load methods.",
                        file_involved="src/utils/checkpoint.py",
                        root_cause="Corrupted or non-identical checkpoint state loading.",
                    )

            return CheckResult(
                name="Checkpoint",
                passed=True,
                details="Checkpoint serialization verified. Loaded model parameters are bitwise identical.",
            )
        except Exception as e:
            return CheckResult(
                name="Checkpoint",
                passed=False,
                failure_reason=str(e),
                suggested_fix="Check torch.save / torch.load configuration.",
                file_involved="src/utils/checkpoint.py",
                root_cause=f"Checkpoint error: {type(e).__name__}",
            )

    def check_inference_generation(self) -> CheckResult:
        """Check 12: Instantiates InferenceEngine and validates text generation."""
        try:
            engine = InferenceEngine(
                model=self.model,
                tokenizer=self.tokenizer,
                device="cpu",
            )

            prompt = "ROMEO:"
            generated_text = engine.generate(prompt=prompt, max_new_tokens=10, do_sample=False)

            if not generated_text or not isinstance(generated_text, str):
                return CheckResult(
                    name="Inference",
                    passed=False,
                    failure_reason="Generated text response is empty or non-string.",
                    suggested_fix="Check InferenceEngine autoregressive generation loop.",
                    file_involved="src/inference/engine.py",
                    root_cause="Autoregressive sampling failure.",
                )

            return CheckResult(
                name="Inference",
                passed=True,
                details=f"InferenceEngine verified. Prompt: '{prompt}' -> Completion: '{generated_text[:40]}...'",
            )
        except Exception as e:
            return CheckResult(
                name="Inference",
                passed=False,
                failure_reason=str(e),
                suggested_fix="Inspect InferenceEngine.generate() implementation.",
                file_involved="src/inference/engine.py",
                root_cause=f"Inference generation crash: {type(e).__name__}",
            )

    def check_unit_tests(self) -> CheckResult:
        """Check 13: Automatically executes pytest suite and parses summary results."""
        try:
            cmd = [sys.executable, "-m", "pytest", "--no-cov", "-q"]
            res = subprocess.run(cmd, cwd=str(self.root_dir), capture_output=True, text=True, timeout=120)

            stdout = res.stdout
            if res.returncode == 0:
                summary_line = stdout.strip().split("\n")[-1] if stdout else "All unit tests passed."
                return CheckResult(
                    name="Unit Tests",
                    passed=True,
                    details=f"pytest suite executed successfully. Result: {summary_line}",
                )
            else:
                stderr_summary = res.stderr or stdout
                last_lines = "\n".join(stderr_summary.strip().split("\n")[-5:])
                return CheckResult(
                    name="Unit Tests",
                    passed=False,
                    failure_reason=f"Pytest suite failed with return code {res.returncode}.\n{last_lines}",
                    suggested_fix="Run 'pytest -v' in terminal to identify failing test cases.",
                    file_involved="tests/",
                    root_cause="Failing unit test assertions in workspace.",
                )
        except Exception as e:
            return CheckResult(
                name="Unit Tests",
                passed=False,
                failure_reason=str(e),
                suggested_fix="Verify pytest is installed in Python environment.",
                file_involved="tests/",
                root_cause=f"Subprocess test execution error: {type(e).__name__}",
            )

    def _build_report(self) -> ValidationReport:
        """Constructs system metadata and builds ValidationReport container."""
        git_hash = "Unknown"
        try:
            git_res = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True)
            if git_res.returncode == 0:
                git_hash = git_res.stdout.strip()
        except Exception:
            pass

        device_name = "CPU"
        cuda_avail = torch.cuda.is_available()
        if cuda_avail:
            device_name = torch.cuda.get_device_name(0)

        return ValidationReport(
            timestamp=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            python_version=platform.python_version(),
            pytorch_version=torch.__version__,
            cuda_available=cuda_avail,
            device_name=device_name,
            git_hash=git_hash,
            results=self.results,
        )

    def print_terminal_output(self, report: ValidationReport) -> None:
        """Renders formatted terminal status table to stdout."""
        print("\n" + "=" * 52)
        print("Aura System Validation")
        print("=" * 52)

        for res in report.results:
            dots = "." * (30 - len(res.name))
            status_str = "PASS" if res.passed else "FAIL"
            print(f"{res.name} {dots} {status_str}")

        print("=" * 52)
        print("Total Checks")
        print(f"  Passed:   {report.total_passed}")
        print(f"  Failed:   {report.total_failed}")
        print(f"  Warnings: {report.total_warnings}")
        print("=" * 52)
        print("Aura Status")
        if report.is_ready_for_training:
            print("  READY FOR TRAINING")
        else:
            print("  NEEDS ATTENTION (System Check Failed)")
        print("=" * 52 + "\n")

        # If any failed, print failure diagnostics block
        if report.total_failed > 0:
            print("\n" + "!" * 60)
            print("FAILURE DIAGNOSTICS DETECTED")
            print("!" * 60)
            for res in report.results:
                if not res.passed:
                    print(f"\n[FAIL] {res.name}")
                    print(f"  Reason:        {res.failure_reason}")
                    print(f"  Suggested Fix: {res.suggested_fix}")
                    print(f"  File Involved: {res.file_involved}")
                    print(f"  Possible Cause:{res.root_cause}")
            print("!" * 60 + "\n")

    def generate_markdown_report(self, report: ValidationReport) -> Path:
        """Generates reports/validation_report.md markdown file.

        Args:
            report: ValidationReport object.

        Returns:
            Path to generated markdown file.
        """
        report_path = self.reports_dir / "validation_report.md"

        status_text = "READY FOR TRAINING" if report.is_ready_for_training else "NEEDS ATTENTION"

        md_content = f"""# Aura System Validation Report

**Generated Date**: {report.timestamp}  
**Git Commit Hash**: `{report.git_hash}`  
**System Status**: **{status_text}**  

---

## 1. Environment & Hardware Specifications

| System Attribute | Recorded Value |
| :--- | :--- |
| **Python Version** | `{report.python_version}` |
| **PyTorch Version** | `{report.pytorch_version}` |
| **CUDA Available** | `{report.cuda_available}` |
| **Compute Device** | `{report.device_name}` |
| **OS Platform** | `{platform.system()} {platform.release()}` |

---

## 2. Validation Summary

| Metric | Count |
| :--- | :--- |
| **Total Checks Performed** | {len(report.results)} |
| **Passed Checks** | {report.total_passed} |
| **Failed Checks** | {report.total_failed} |
| **Warnings** | {report.total_warnings} |

---

## 3. Detailed Check Results

| Check Name | Status | Details / Diagnostic Notes |
| :--- | :--- | :--- |
"""
        for r in report.results:
            status_badge = "✅ PASS" if r.passed else "❌ FAIL"
            details = r.details if r.passed else f"**Error**: {r.failure_reason}"
            md_content += f"| **{r.name}** | {status_badge} | {details} |\n"

        if report.total_failed > 0:
            md_content += "\n---\n\n## 4. Failure Diagnostic & Remediation Guide\n\n"
            for r in report.results:
                if not r.passed:
                    md_content += f"""### ❌ Check Failed: {r.name}
- **Failure Reason**: {r.failure_reason}
- **Suggested Fix**: {r.suggested_fix}
- **File Involved**: `{r.file_involved}`
- **Root Cause**: {r.root_cause}

"""
        else:
            md_content += "\n---\n\n## 4. System Readiness Sign-off\n\n"
            md_content += "All 13 system validation stages completed successfully with zero errors. The Aura codebase, data pipelines, model architecture, optimization engine, checkpointing, and inference systems are verified **READY FOR TRAINING**.\n"

        with open(report_path, "w", encoding="utf-8") as f:
            f.write(md_content)

        return report_path


def main() -> None:
    """Main entry point executing Aura system validation script."""
    validator = AuraSystemValidator()
    report = validator.run_all_checks()
    validator.print_terminal_output(report)

    if not report.is_ready_for_training:
        sys.exit(1)


if __name__ == "__main__":
    main()
