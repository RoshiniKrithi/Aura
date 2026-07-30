<div align="center">

# ⚡ Aura

**A Custom GPT-Style Programming & DSA Large Language Model Built Completely From Scratch in PyTorch**

[![Python 3.12+](https://img.shields.io/badge/Python-3.12%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.2+](https://img.shields.io/badge/PyTorch-2.2%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Linter: Ruff](https://img.shields.io/badge/linter-ruff-brightgreen.svg)](https://github.com/astral-sh/ruff)

</div>

---

## 🎯 Project Vision

**Aura** is an open, production-grade Large Language Model engineered specifically for **Programming, Data Structures, and Algorithms (DSA)**.

The objective of Aura is to build a domain-specialized autoregressive transformer completely from first principles in PyTorch—**without using external LLM wrappers or third-party training frameworks** (e.g., *no* Hugging Face Trainer, *no* nanoGPT, *no* minGPT). Every layer, attention calculation, tokenization byte-pair merge, loss function, and optimization step is implemented natively to ensure maximum mechanical transparency and architectural mastery.

### Capabilities target:
- 💡 **Explaining Algorithms**: Deep step-by-step mathematical and logical explanations of complex algorithms.
- ⚡ **Solving DSA Problems**: Generating optimal code solutions for competitive programming and LeetCode-style challenges.
- 🛠️ **High-Quality Code Generation**: Clean, idiomatic, typed, and docstring-annotated code production.
- 🐞 **Debugging Code**: Identifying root causes of runtime exceptions, memory leaks, and logic bugs.
- ⏱️ **Complexity Analysis**: Computing exact Big-O time and space complexity with formal mathematical justifications.
- 🎓 **Interview Preparation**: Interactive pair-programming and technical mock interview assistance.

---

## 🏗️ Architecture

Aura implements a decoder-only GPT-style transformer architecture optimized for structural code modeling:

```text
┌─────────────────────────────────────────────────────────┐
│                      Input Text                         │
└────────────────────────────┬────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────┐
│              Byte-Pair Encoding Tokenizer               │
└────────────────────────────┬────────────────────────────┘
                             │  [batch_size, seq_len]
                             ▼
┌─────────────────────────────────────────────────────────┐
│        Token Embeddings + Positional Encodings          │
└────────────────────────────┬────────────────────────────┘
                             │  [batch_size, seq_len, d_model]
                             ▼
┌─────────────────────────────────────────────────────────┐
│            Stack of N x Transformer Blocks              │
│  ┌───────────────────────────────────────────────────┐  │
│  │ RMSNorm / LayerNorm                               │  │
│  │ Causal Multi-Head Self-Attention (MHA)            │  │
│  │ Residual Add                                      │  │
│  │ RMSNorm / LayerNorm                               │  │
│  │ SwiGLU / GeLU Feed-Forward MLP                    │  │
│  │ Residual Add                                      │  │
│  └───────────────────────────────────────────────────┘  │
└────────────────────────────┬────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────┐
│                 Final Layer Normalization               │
└────────────────────────────┬────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────┐
│          Linear LM Head (Vocab Projection)              │
└────────────────────────────┬────────────────────────────┘
                             │  [batch_size, seq_len, vocab_size]
                             ▼
┌─────────────────────────────────────────────────────────┐
│                Next-Token Cross-Entropy                 │
└─────────────────────────────────────────────────────────┘
```

---

## 🛠️ Technology Stack

- **Core Engine**: Python 3.12+
- **Deep Learning Framework**: PyTorch 2.2+ (Pure PyTorch, C++ extensions compatible)
- **Mathematical Utilities**: NumPy 1.26+
- **Configuration & Serialization**: PyYAML, JSON
- **Testing & Quality Assurance**: PyTest, PyTest-Cov, MyPy (Strict Typing)
- **Formatting & Linting**: Black, Ruff, Pre-Commit

---

## 🗺️ Project Roadmap (Phases 1–18)

- [x] **Phase 1: Project Foundation** — Configuration loader, logging system, device detection, reproducibility seeds, path management, and checkpointing.
- [ ] **Phase 2: Tokenizer** — Custom Byte-Pair Encoding (BPE) tokenizer built from scratch.
- [ ] **Phase 3: Dataset Pipeline** — Custom PyTorch streaming dataset loaders, token chunking, and collators.
- [ ] **Phase 4: Embeddings** — Token embedding lookup layers with scaling factor.
- [ ] **Phase 5: Positional Embeddings** — Absolute, Sinusoidal, and Rotary Positional Embeddings (RoPE).
- [ ] **Phase 6: Self-Attention** — Scaled Dot-Product Attention mechanism with causal lower-triangular masking.
- [ ] **Phase 7: Multi-Head Attention** — Multi-head linear projection splitting, parallel head computation, and output projection.
- [ ] **Phase 8: Feed-Forward Network** — Position-wise feed-forward networks (GeLU / SwiGLU projection layers).
- [ ] **Phase 9: Layer Normalization** — Standard LayerNorm and RMSNorm implemented from scratch without autograd dependency.
- [ ] **Phase 10: Residual Connections** — Pre-LN / Post-LN residual skip connection wrappers.
- [ ] **Phase 11: Transformer Block** — Unified decoder layer assembling Attention, FFN, Normalization, and Residuals.
- [ ] **Phase 12: GPT Model** — Full top-level AuraGPT architecture assembly with weight initialization and parameter counting.
- [ ] **Phase 13: Training Loop** — Custom training loop featuring gradient accumulation, AMP, grad clipping, and validation.
- [ ] **Phase 14: Inference** — Autoregressive generation engine with KV-Caching, Temperature, Top-k, and Top-p sampling.
- [ ] **Phase 15: Programming Dataset** — Dataset scraping, deduplication, filtering, and instruction-formatting for DSA text.
- [ ] **Phase 16: Fine-tuning** — Supervised Fine-Tuning (SFT) for instruction-following code alignment.
- [ ] **Phase 17: Evaluation** — HumanEval/MBPP benchmark suite, sandboxed Python code execution tester, and perplexity evaluation.
- [ ] **Phase 18: Deployment** — Local web API server, streaming terminal CLI interface, and model quantization.

---

## 📁 Directory Structure

```text
Aura/
├── configs/                    # YAML configuration files
│   └── config.yaml             # Main project config schema
├── data/                       # Raw, processed, and tokenized datasets
├── docs/                       # Architectural specifications and design docs
├── notebooks/                  # Experimental analysis and research notebooks
├── src/                        # Main source code package
│   ├── __init__.py
│   ├── attention/              # Self-Attention & Multi-Head Attention modules
│   ├── datasets/               # PyTorch Datasets & DataLoaders
│   ├── embeddings/             # Token & Positional Embedding layers
│   ├── evaluation/             # HumanEval, MBPP & performance metrics
│   ├── inference/              # Generation engine & KV-cache decoder
│   ├── logging/                # Structured logging system
│   ├── losses/                 # Loss functions (CrossEntropyLoss from scratch)
│   ├── models/                 # Complete AuraGPT model architecture
│   ├── optimizers/             # Custom optimizers (AdamW from scratch)
│   ├── schedulers/             # Learning rate schedulers (Cosine Warmup)
│   ├── tokenizer/              # BPE Tokenizer engine
│   ├── training/               # Distributed training loops & execution
│   ├── transformer/            # Transformer Blocks, LayerNorm, & MLPs
│   └── utils/                  # Device, Seed, Path, Config & Checkpoint tools
├── tests/                      # PyTest unit and integration tests
├── scripts/                    # CLI execution entry points (train.py, infer.py)
├── checkpoints/                # Model weight checkpoints
└── outputs/                    # Logs, experiment outputs, and metrics
```

---

## 📏 Development Standards & Conventions

### 1. Code Style & Formatting
- **PEP 8**: Strict adherence to standard Python style guidelines.
- **Formatter**: `black` with 100 character line limit.
- **Linter**: `ruff` enforcing strict linting, import sorting, and bug detection.
- **Typing**: Mandatory strict type hints for all function arguments and return types.

### 2. Documentation Convention
- **Google Style Docstrings**: Required for all classes, methods, and functions.
```python
def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """Set random seed across all libraries to ensure reproducible experiments.

    Args:
        seed: Integer random seed value.
        deterministic: If True, configures PyTorch backends for determinism.
    """
```

### 3. Naming Conventions
- **Modules / Files**: `snake_case.py` (e.g., `config_loader.py`)
- **Classes**: `PascalCase` (e.g., `CheckpointManager`)
- **Functions & Variables**: `snake_case` (e.g., `detect_device`)
- **Constants**: `UPPER_SNAKE_CASE` (e.g., `DEFAULT_VOCAB_SIZE`)

### 4. Git Branching & Commit Message Convention
- **Branches**: `feature/phase-X-name`, `bugfix/issue-description`, `docs/readme-update`
- **Conventional Commits**:
  - `feat(tokenizer): implement byte-pair encoding merge algorithm`
  - `fix(device): resolve MPS detection issue on macOS`
  - `docs(readme): add phase 1 architecture diagram`
  - `test(config): add unit tests for yaml loader`

---

## ⚡ Installation Guide

### Prerequisites
- Python 3.12+
- PyTorch 2.2+ (CUDA 12.x compatible if using NVIDIA GPUs)

### 1. Clone Repository & Setup Virtual Environment
```bash
git clone https://github.com/your-username/Aura.git
cd Aura

python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate
```

### 2. Install Package & Dependencies
```bash
pip install --upgrade pip
pip install -e ".[dev]"
```

### 3. Setup Pre-Commit Hooks
```bash
pre-commit install
```

---

## 🚀 Usage Guide

### Running Test Suite
Execute unit tests across all foundation modules:
```bash
pytest
```

### Initializing Foundation Utilities
```python
from src.utils import get_device, set_seed, load_config, project_paths
from src.logging import setup_logger

# 1. Setup Structured Logger
logger = setup_logger(name="aura", level="INFO")

# 2. Set Seed for Reproducibility
set_seed(42, deterministic=True)

# 3. Detect Optimal Compute Hardware (CUDA -> MPS -> CPU)
device = get_device("auto")

# 4. Load Strongly-Typed Configuration
config = load_config()
logger.info(f"Loaded model: {config.model.name} with d_model={config.model.d_model}")
```

---

## 🤝 Contributing & License

Contributions are welcome! Please ensure all code passes `black`, `ruff`, `mypy`, and all tests in `pytest` before submitting pull requests.

Distributed under the **MIT License**. See [`LICENSE`](file:///d:/Aura/LICENSE) for details.
