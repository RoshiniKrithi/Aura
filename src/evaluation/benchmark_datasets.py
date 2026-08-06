"""Benchmark Dataset Loaders and Problem Registry for Aura EXP-005 Evaluation Suite.

Provides BenchmarkProblem, BenchmarkRegistry, BenchmarkLoader, HumanEvalLoader,
MBPPLoader, APPSLoader, and CustomBenchmarkLoader.
"""

from dataclasses import dataclass, field
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Union

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkProblem:
    """Represents a single programming evaluation benchmark problem instance.

    Attributes:
        task_id: Unique string identifier (e.g. "HumanEval/0").
        prompt: Task specification or docstring prompt.
        entry_point: Expected target function or class entry point name.
        canonical_solution: Ground-truth baseline solution code string.
        test_cases: List of unit test assertion strings.
        metadata: Optional dictionary metadata (difficulty, language, domain).
    """

    task_id: str
    prompt: str
    entry_point: str
    canonical_solution: str
    test_cases: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseBenchmarkLoader:
    """Abstract base class for benchmark problem loaders."""

    def load_problems(self, file_path: Union[str, Path]) -> List[BenchmarkProblem]:
        """Loads problems from benchmark file path."""
        raise NotImplementedError


class HumanEvalLoader(BaseBenchmarkLoader):
    """Loader for OpenAI HumanEval benchmark format."""

    def load_problems(self, file_path: Union[str, Path]) -> List[BenchmarkProblem]:
        path = Path(file_path).resolve()
        if not path.exists():
            logger.warning("HumanEval dataset file not found: %s", path)
            return []

        problems: List[BenchmarkProblem] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                prob = BenchmarkProblem(
                    task_id=data.get("task_id", f"HumanEval/{len(problems)}"),
                    prompt=data.get("prompt", ""),
                    entry_point=data.get("entry_point", ""),
                    canonical_solution=data.get("canonical_solution", ""),
                    test_cases=[data.get("test", "")],
                    metadata={"benchmark": "humaneval"},
                )
                problems.append(prob)

        logger.info("Loaded %d HumanEval problems from %s", len(problems), path.name)
        return problems


class MBPPLoader(BaseBenchmarkLoader):
    """Loader for Mostly Basic Python Problems (MBPP) benchmark format."""

    def load_problems(self, file_path: Union[str, Path]) -> List[BenchmarkProblem]:
        path = Path(file_path).resolve()
        if not path.exists():
            logger.warning("MBPP dataset file not found: %s", path)
            return []

        problems: List[BenchmarkProblem] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                tests = data.get("test_list", [])
                prob = BenchmarkProblem(
                    task_id=f"MBPP/{data.get('task_id', len(problems))}",
                    prompt=data.get("text", ""),
                    entry_point=data.get("entry_point", "solution"),
                    canonical_solution=data.get("code", ""),
                    test_cases=tests,
                    metadata={"benchmark": "mbpp"},
                )
                problems.append(prob)

        logger.info("Loaded %d MBPP problems from %s", len(problems), path.name)
        return problems


class APPSLoader(BaseBenchmarkLoader):
    """Loader for APPS competitive programming benchmark format."""

    def load_problems(self, file_path: Union[str, Path]) -> List[BenchmarkProblem]:
        path = Path(file_path).resolve()
        if not path.exists():
            logger.warning("APPS dataset file not found: %s", path)
            return []

        problems: List[BenchmarkProblem] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                prob = BenchmarkProblem(
                    task_id=f"APPS/{data.get('problem_id', len(problems))}",
                    prompt=data.get("question", ""),
                    entry_point=data.get("entry_point", "solve"),
                    canonical_solution=data.get("solutions", [""])[0],
                    test_cases=[data.get("input_output", "")],
                    metadata={"benchmark": "apps", "difficulty": data.get("difficulty", "medium")},
                )
                problems.append(prob)

        logger.info("Loaded %d APPS problems from %s", len(problems), path.name)
        return problems


class CustomBenchmarkLoader(BaseBenchmarkLoader):
    """Loader for custom JSONL programming & DSA benchmark files."""

    def load_problems(self, file_path: Union[str, Path]) -> List[BenchmarkProblem]:
        path = Path(file_path).resolve()
        if not path.exists():
            logger.warning("Custom benchmark dataset file not found: %s", path)
            return []

        problems: List[BenchmarkProblem] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                prob = BenchmarkProblem(
                    task_id=data.get("task_id", f"Custom/{len(problems)}"),
                    prompt=data.get("prompt", data.get("question", "")),
                    entry_point=data.get("entry_point", "solution"),
                    canonical_solution=data.get("canonical_solution", data.get("solution", "")),
                    test_cases=data.get("test_cases", [data.get("test", "")]),
                    metadata={"benchmark": "custom_dsa"},
                )
                problems.append(prob)

        logger.info("Loaded %d custom benchmark problems from %s", len(problems), path.name)
        return problems


class BenchmarkRegistry:
    """Central registry mapping benchmark names to loader implementations."""

    LOADER_MAP: Dict[str, Type[BaseBenchmarkLoader]] = {
        "humaneval": HumanEvalLoader,
        "mbpp": MBPPLoader,
        "apps": APPSLoader,
        "custom": CustomBenchmarkLoader,
        "custom_dsa": CustomBenchmarkLoader,
    }

    @classmethod
    def get_loader(cls, benchmark_name: str) -> BaseBenchmarkLoader:
        """Returns instantiated loader for given benchmark name.

        Args:
            benchmark_name: Key string (e.g. "humaneval", "mbpp").

        Returns:
            Instantiated BaseBenchmarkLoader subclass.
        """
        loader_cls = cls.LOADER_MAP.get(benchmark_name.lower(), CustomBenchmarkLoader)
        return loader_cls()


class BenchmarkLoader:
    """Unified helper for loading benchmark datasets across formats."""

    @staticmethod
    def load_benchmark(
        file_path: Union[str, Path], benchmark_type: str = "custom"
    ) -> List[BenchmarkProblem]:
        """Loads benchmark problem dataset.

        Args:
            file_path: Path to dataset JSONL file.
            benchmark_type: Key name ("humaneval", "mbpp", "apps", "custom").

        Returns:
            List of BenchmarkProblem objects.
        """
        loader = BenchmarkRegistry.get_loader(benchmark_type)
        return loader.load_problems(file_path)
