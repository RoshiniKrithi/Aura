"""Models Subsystem Module for Aura LLM Architecture.

Provides GPTConfig, GPTModel decoder trunk, TransformerStack composition module,
AuraGPT complete model, LMHeadConfig, LanguageModelingHead, LanguageModelHead,
LMHeadFactory, LMHeadValidator, LMHeadUtilities, and LMHeadStatistics.
"""

from src.models.config import AuraGPTConfig, GPTConfig
from src.models.exceptions import (
    LMHeadConfigError,
    LMHeadError,
    LMHeadValidationError,
    ModelConfigError,
    ModelError,
    ModelValidationError,
)
from src.models.factory import AuraGPTFactory, ModelFactory
from src.models.gpt import AuraGPT
from src.models.gpt_model import GPTModel
from src.models.initializer import ModelInitializer
from src.models.lm_head import LanguageModelHead, LanguageModelingHead
from src.models.lm_head_config import LMHeadConfig
from src.models.lm_head_factory import LMHeadFactory
from src.models.lm_head_statistics import LMHeadStats, LMHeadStatistics
from src.models.lm_head_utilities import LMHeadUtilities
from src.models.lm_head_validator import LMHeadValidationResult, LMHeadValidator
from src.models.statistics import ModelStats, ModelStatistics
from src.models.transformer_stack import TransformerStack
from src.models.utilities import ModelUtilities
from src.models.validator import ModelValidationResult, ModelValidator

__all__ = [
    "GPTConfig",
    "AuraGPTConfig",
    "GPTModel",
    "TransformerStack",
    "AuraGPT",
    "LMHeadConfig",
    "LanguageModelingHead",
    "LanguageModelHead",
    "LMHeadFactory",
    "LMHeadValidator",
    "LMHeadValidationResult",
    "LMHeadUtilities",
    "LMHeadStatistics",
    "LMHeadStats",
    "ModelFactory",
    "AuraGPTFactory",
    "ModelInitializer",
    "ModelStatistics",
    "ModelStats",
    "ModelValidator",
    "ModelValidationResult",
    "ModelUtilities",
    "ModelError",
    "ModelValidationError",
    "ModelConfigError",
    "LMHeadError",
    "LMHeadValidationError",
    "LMHeadConfigError",
]
