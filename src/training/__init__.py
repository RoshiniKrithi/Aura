"""Training Engine Subsystem Module for Aura LLM Architecture.

Provides TrainingEngineConfig, TrainingEngine, EpochRunner, BatchRunner, ValidationRunner,
MetricsManager, ProgressTracker, TrainingLogger, TrainingStatistics, EngineFactory, EngineValidator, and EngineUtilities.
"""

from src.training.config import TrainingEngineConfig
from src.training.engine import TrainingEngine
from src.training.exceptions import (
    EngineConfigError,
    EngineError,
    EngineValidationError,
)
from src.training.factory import EngineFactory
from src.training.logger import TrainingLogger
from src.training.metrics import MetricSummary, MetricTracker, MetricTracker as MetricsManager
from src.training.progress import ProgressTracker
from src.training.runners import BatchRunner, EpochRunner, ValidationRunner
from src.training.statistics import TrainingStatistics, TrainingStats
from src.training.utilities import EngineUtilities
from src.training.validator import EngineValidationResult, EngineValidator

from src.training.exp_001 import (
    ArtifactManager,
    ExperimentConfig,
    ExperimentRunner,
    ExperimentStatistics,
    MetricLogger,
    SampleGenerator,
    TensorBoardLogger,
    TrainingSession,
    ValidationSession,
)

from src.training.exp_003_orchestrator import (
    CurriculumScheduler,
    DatasetMixer,
    DynamicBatchBuilder,
    EvaluationManager,
    ExperimentTracker,
    ProgrammingPretrainingConfig,
    ProgrammingPretrainingRunner,
    SequencePacker,
)

__all__ = [
    "TrainingEngineConfig",
    "TrainingEngine",
    "EpochRunner",
    "BatchRunner",
    "ValidationRunner",
    "MetricsManager",
    "MetricTracker",
    "MetricSummary",
    "ProgressTracker",
    "TrainingLogger",
    "TrainingStatistics",
    "TrainingStats",
    "EngineFactory",
    "EngineValidator",
    "EngineValidationResult",
    "EngineUtilities",
    "EngineError",
    "EngineValidationError",
    "EngineConfigError",
    "ExperimentConfig",
    "ExperimentRunner",
    "TrainingSession",
    "ValidationSession",
    "ArtifactManager",
    "MetricLogger",
    "TensorBoardLogger",
    "SampleGenerator",
    "ExperimentStatistics",
    "ProgrammingPretrainingConfig",
    "ProgrammingPretrainingRunner",
    "DatasetMixer",
    "CurriculumScheduler",
    "DynamicBatchBuilder",
    "SequencePacker",
    "ExperimentTracker",
    "EvaluationManager",
]

