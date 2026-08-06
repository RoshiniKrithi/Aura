"""Agentic Programming Assistant Master Orchestrator for Aura EXP-009.

Coordinates task decomposition, ReAct Chain-of-Thought reasoning steps,
tool execution dispatch, observation capture, and self-correction reflection loops.
"""

from typing import Any, Dict, List, Optional, Union
import json
import logging
from pathlib import Path
import time
import torch
import torch.nn as nn

from src.agent.config import AgentConfig
from src.agent.exceptions import AgentException, MaxStepsExceededError
from src.agent.memory import MemoryManager
from src.agent.planner import TaskPlanner
from src.agent.reasoning import ReasoningEngine
from src.agent.reflection import ReflectionEngine
from src.agent.tools import ToolManager
from src.agent.validator import AgentValidator
from src.optimization.inference_optimizer import OptimizedInferenceEngine

logger = logging.getLogger(__name__)


class AgenticProgrammingAssistant:
    """Master Autonomous Software Engineering Agent for Aura LLM."""

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        rag_orchestrator: Optional[Any] = None,
        config: Optional[AgentConfig] = None,
    ) -> None:
        """Initializes AgenticProgrammingAssistant.

        Args:
            model: PyTorch nn.Module instance (AuraGPT or Optimized model).
            tokenizer: CodeBPETokenizer or BPE Tokenizer instance.
            rag_orchestrator: Optional RAGOrchestrator instance for knowledge lookup.
            config: Optional AgentConfig configuration container.
        """
        self.config = config or AgentConfig()

        val_res = AgentValidator.validate_config(self.config)
        if not val_res.is_valid:
            raise AgentException(f"Invalid AgentConfig: {val_res.errors}")

        self.model = model
        self.tokenizer = tokenizer

        # Bind existing EXP-008 OptimizedInferenceEngine
        self.inference_engine = OptimizedInferenceEngine(model=model, tokenizer=tokenizer)

        # Agent Subsystems
        self.planner = TaskPlanner(planning_depth=self.config.planning_depth)
        self.tool_manager = ToolManager(
            workspace_root=self.config.workspace_root,
            timeout_sec=self.config.execution_timeout_sec,
            rag_orchestrator=rag_orchestrator,
        )
        self.memory = MemoryManager(max_memory_tokens=self.config.max_memory_tokens)

        logger.info("Initialized AgenticProgrammingAssistant with workspace '%s'.", self.config.workspace_root)

    def run_task(self, task_description: str) -> Dict[str, Any]:
        """Executes an autonomous multi-step software engineering task.

        Args:
            task_description: Natural language instruction task prompt string.

        Returns:
            Dict containing final_solution, execution_trace, and performance metrics.
        """
        start_time = time.time()
        self.memory.clear()

        # 1. Task Decomposition Planning
        subtasks = self.planner.decompose_task(task_description)
        system_prompt = ReasoningEngine.format_system_prompt(self.config.allowed_tools)

        step_idx = 0
        final_answer = None
        trace = []

        logger.info("Starting agent ReAct execution loop for task: %s", task_description[:60])

        while step_idx < self.config.max_reasoning_steps:
            step_idx += 1
            logger.info("--- ReAct Reasoning Step %d/%d ---", step_idx, self.config.max_reasoning_steps)

            # Build prompt history
            working_context = self.memory.get_working_context_prompt()
            full_prompt = (
                f"{system_prompt}\n"
                f"Task: {task_description}\n\n"
                f"{working_context}\n"
                "Thought:"
            )

            # Generate candidate step using OptimizedInferenceEngine
            raw_response = self._generate_step_completion(full_prompt)

            # Parse ReAct Step
            parsed_step = ReasoningEngine.parse_model_response("Thought: " + raw_response)

            # Final Answer Check
            if parsed_step.is_final:
                final_answer = parsed_step.final_answer
                self.memory.add_step(
                    step_index=step_idx,
                    thought=parsed_step.thought,
                    observation="Completed task.",
                )
                trace.append({
                    "step": step_idx,
                    "thought": parsed_step.thought,
                    "final_answer": final_answer,
                })
                logger.info("Agent reached Final Answer at step %d.", step_idx)
                break

            # Tool Selection & Execution
            tool_name = parsed_step.action
            tool_input = parsed_step.action_input or {}
            observation_str = ""
            reflection_str = None

            if tool_name:
                # Permission check
                perm_res = AgentValidator.validate_tool_permission(tool_name, self.config.allowed_tools)
                if not perm_res.is_valid:
                    observation_str = f"Error: {perm_res.errors[0]}"
                else:
                    tool_res = self.tool_manager.execute_tool(tool_name, **tool_input)
                    observation_str = tool_res.output if tool_res.exit_code == 0 else f"Error: {tool_res.error}"

                    # Execute Self-Correction Reflection if tool failed
                    if tool_res.exit_code != 0 and self.config.enable_reflection:
                        reflection_str = ReflectionEngine.diagnose_failure(
                            tool_name=tool_name,
                            error_msg=tool_res.error or "Unknown error",
                            output=tool_res.output,
                        )
                        self.memory.add_reflection(reflection_str)
            else:
                observation_str = "No tool action specified by model."

            # Update Short-Term Memory
            self.memory.add_step(
                step_index=step_idx,
                thought=parsed_step.thought,
                action=tool_name,
                tool_args=tool_input,
                observation=observation_str,
                reflection=reflection_str,
            )

            trace.append({
                "step": step_idx,
                "thought": parsed_step.thought,
                "action": tool_name,
                "action_input": tool_input,
                "observation": observation_str,
                "reflection": reflection_str,
            })

        elapsed_sec = time.time() - start_time

        if final_answer is None and step_idx >= self.config.max_reasoning_steps:
            final_answer = "Max reasoning steps reached without explicit final answer."
            logger.warning("Reached max reasoning steps (%d).", self.config.max_reasoning_steps)

        result_payload = {
            "task": task_description,
            "final_solution": final_answer,
            "total_steps": step_idx,
            "execution_time_sec": round(elapsed_sec, 3),
            "execution_trace": trace,
            "plan_status": "completed" if self.planner.is_plan_completed() else "in_progress",
        }

        # Export trace log
        self._export_trace(result_payload)
        return result_payload

    def _generate_step_completion(self, prompt_text: str) -> str:
        """Helper to generate step text completion using InferenceEngine."""
        try:
            res = self.inference_engine.generate(
                prompt=prompt_text,
                max_new_tokens=128,
                temperature=self.config.temperature,
            )
            if isinstance(res, torch.Tensor):
                return str(res.tolist())
            # Truncate prompt from response
            if prompt_text in res:
                res = res.replace(prompt_text, "")
            return res.strip()
        except Exception as e:
            logger.warning("Model inference call failed: %s. Using default ReAct fallback.", str(e))
            return "Thought: Executing tool inspection.\nAction: workspace_search\nAction Input: {\"pattern\": \"def \"}"

    def _export_trace(self, payload: Dict[str, Any]) -> Path:
        """Exports agent session execution trace report to JSON."""
        out_dir = Path(self.config.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        file_path = out_dir / "agent_session_trace.json"

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        logger.info("Exported agent session trace to '%s'.", file_path)
        return file_path
