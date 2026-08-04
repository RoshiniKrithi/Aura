"""Comprehensive PyTest Suite for Phase 18 Inference & Text Generation Subsystem.

Validates Greedy, Temperature, Top-K, and Top-P decoding strategies,
end-to-end autoregressive text generation, streaming generators, repetition penalties,
context window truncation, InferenceFactory builders, and validator checks.
"""

import pytest
import torch
import torch.nn as nn

from src.inference import (
    CompositeSamplingStrategy,
    GreedyStrategy,
    InferenceConfig,
    InferenceEngine,
    InferenceFactory,
    InferenceStatistics,
    InferenceUtilities,
    InferenceValidationError,
    InferenceValidator,
    TemperatureStrategy,
    TopKStrategy,
    TopPStrategy,
)
from src.models import AuraGPT, AuraGPTConfig


class MockTokenizer:
    """Simple mock tokenizer for testing text generation engines."""

    def __init__(self, vocab_size: int = 100):
        self.vocab_size = vocab_size
        self.eos_token_id = 2
        self.pad_token_id = 0

    def encode(self, text: str):
        return [1, 10, 20, 30]

    def decode(self, token_ids):
        return f"decoded_text_{'_'.join(str(t) for t in token_ids)}"


@pytest.fixture
def mock_tokenizer():
    return MockTokenizer(vocab_size=500)


@pytest.fixture
def small_aura_model():
    cfg = AuraGPTConfig(vocab_size=500, d_model=32, n_layers=2, n_heads=2, d_ff=128)
    return AuraGPT(config=cfg)


# 1. Configuration Schema Tests
def test_inference_config_defaults():
    cfg = InferenceConfig()
    assert cfg.max_new_tokens == 256
    assert cfg.temperature == 0.7
    assert cfg.top_k == 50
    assert cfg.top_p == 0.9
    assert cfg.do_sample is True


# 2. Decoding Strategies Unit Tests
def test_greedy_strategy():
    logits = torch.tensor([[1.0, 5.0, 2.0]])
    token = GreedyStrategy.select_token(logits)
    assert token.item() == 1


def test_temperature_strategy():
    strat = TemperatureStrategy()
    cfg = InferenceConfig(temperature=0.5)
    logits = torch.tensor([[1.0, 2.0]])
    scaled = strat.process_logits(logits, cfg)
    assert torch.allclose(scaled, torch.tensor([[2.0, 4.0]]))


def test_top_k_strategy():
    strat = TopKStrategy()
    cfg = InferenceConfig(top_k=2)
    logits = torch.tensor([[1.0, 10.0, 5.0, 2.0]])
    filtered = strat.process_logits(logits, cfg)
    # Positions 0 and 3 (values 1.0, 2.0) should be -inf
    assert filtered[0, 0].item() == float("-inf")
    assert filtered[0, 3].item() == float("-inf")
    assert filtered[0, 1].item() == 10.0
    assert filtered[0, 2].item() == 5.0


def test_top_p_strategy():
    strat = TopPStrategy()
    cfg = InferenceConfig(top_p=0.8)
    logits = torch.tensor([[10.0, 1.0, 0.1]])
    filtered = strat.process_logits(logits, cfg)
    # Position 2 (value 0.1) should be masked out to -inf
    assert filtered[0, 2].item() == float("-inf")


# 3. Inference Utilities Tests
def test_repetition_penalty():
    logits = torch.tensor([[2.0, 4.0, -1.0]])
    history = torch.tensor([[1]])  # token 1 was generated
    penalized = InferenceUtilities.apply_repetition_penalty(logits, history, penalty=2.0)
    assert penalized[0, 1].item() == 2.0  # Positive logit divided by 2.0 -> 2.0


def test_prompt_truncation():
    tokens = torch.tensor([[1, 2, 3, 4, 5]])
    truncated = InferenceUtilities.truncate_prompt_tokens(tokens, max_allowed_tokens=3)
    assert truncated.shape == (1, 3)
    assert truncated[0].tolist() == [3, 4, 5]


# 4. End-to-End Generation Tests
def test_inference_engine_generate(small_aura_model, mock_tokenizer):
    cfg = InferenceConfig(max_new_tokens=5, do_sample=False)
    engine = InferenceEngine(model=small_aura_model, tokenizer=mock_tokenizer, config=cfg)

    result_text = engine.generate("def foo():", max_new_tokens=5)
    assert isinstance(result_text, str)
    assert "decoded_text" in result_text


def test_inference_engine_stream(small_aura_model, mock_tokenizer):
    cfg = InferenceConfig(max_new_tokens=3, do_sample=False)
    engine = InferenceEngine(model=small_aura_model, tokenizer=mock_tokenizer, config=cfg)

    chunks = list(engine.generate_stream("def foo():", max_new_tokens=3))
    assert len(chunks) > 0
    assert all(isinstance(c, str) for c in chunks)


# 5. Factory & Validator Tests
def test_inference_factory(small_aura_model, mock_tokenizer):
    engine = InferenceFactory.create_engine(model=small_aura_model, tokenizer=mock_tokenizer)
    assert isinstance(engine, InferenceEngine)


def test_inference_validator_raises(small_aura_model):
    with pytest.raises(InferenceValidationError):
        InferenceEngine(model=small_aura_model, tokenizer="not_a_tokenizer")
