"""Reasoning Engine and ReAct Loop Handler for Aura Agent (EXP-009).

Manages internal Chain-of-Thought (CoT) prompting, formats ReAct step prompts,
and parses model output into structured Thoughts, Actions, and Parameters.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
import json
import re
import logging

logger = logging.getLogger(__name__)


@dataclass
class ReActStep:
    """Parsed ReAct model output step."""

    thought: str
    action: Optional[str] = None
    action_input: Optional[Dict[str, Any]] = None
    is_final: bool = False
    final_answer: Optional[str] = None


class ReasoningEngine:
    """Parses model responses into ReAct Thoughts, Tool Actions, and Inputs."""

    @staticmethod
    def parse_model_response(response_text: str) -> ReActStep:
        """Parses model output text into ReActStep.

        Expects structured text format:
            Thought: <reasoning>
            Action: <tool_name>
            Action Input: <json_string or key=value>
            Final Answer: <answer_string>
        """
        thought_match = re.search(r"Thought:\s*(.*?)(?=\nAction:|\nFinal Answer:|$)", response_text, re.DOTALL)
        thought = thought_match.group(1).strip() if thought_match else response_text.strip()

        final_match = re.search(r"Final Answer:\s*(.*)", response_text, re.DOTALL)
        if final_match:
            return ReActStep(
                thought=thought,
                is_final=True,
                final_answer=final_match.group(1).strip(),
            )

        action_match = re.search(r"Action:\s*(\w+)", response_text)
        action_input_match = re.search(r"Action Input:\s*(.*)", response_text, re.DOTALL)

        action = action_match.group(1).strip() if action_match else None
        action_input = {}

        if action_input_match:
            raw_input = action_input_match.group(1).strip()
            try:
                action_input = json.loads(raw_input)
            except Exception:
                # Fallback dictionary parsing
                if "=" in raw_input:
                    k, v = raw_input.split("=", 1)
                    action_input = {k.strip(): v.strip().strip("'\"")}
                else:
                    action_input = {"input": raw_input}

        return ReActStep(
            thought=thought,
            action=action,
            action_input=action_input,
            is_final=False,
        )

    @staticmethod
    def format_system_prompt(allowed_tools: list) -> str:
        """Generates agent system prompt instructing model on ReAct format."""
        return (
            "You are Aura, an Autonomous Agentic Programming Assistant.\n"
            "Solve the software engineering task step-by-step using ReAct reasoning.\n"
            "For each step, output strictly in the following format:\n\n"
            "Thought: <Internal Chain-of-Thought reasoning>\n"
            "Action: <tool_name>\n"
            "Action Input: {\"key\": \"value\"}\n\n"
            "When completed, output:\n"
            "Thought: <Final verification summary>\n"
            "Final Answer: <Your final solution code or explanation>\n\n"
            f"Available Tools: {allowed_tools}\n"
        )
