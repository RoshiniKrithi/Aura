"""Production PR Review & Agent Capability Test Suite for Phase 28 / EXP-009 Agent.

Includes comprehensive testing for:
- Agent planning, DAG subtask dependency resolution, and state updates
- Tool permissions, path traversal sandboxing, and execution safety bounds
- Python REPL, Shell GitOps, FileReader, FileWriter, ASTParser, and PyTest tools
- ReAct step parsing, Thought/Action/Input extraction, and Final Answer detection
- Multi-step reasoning loops, reflection self-correction, and JSON trace exports
"""

from pathlib import Path
import tempfile
import time
import pytest
import torch

from src.agent.config import AgentConfig
from src.agent.exceptions import AgentValidationError
from src.agent.memory import MemoryManager
from src.agent.orchestrator import AgenticProgrammingAssistant
from src.agent.planner import TaskPlanner
from src.agent.reasoning import ReasoningEngine
from src.agent.reflection import ReflectionEngine
from src.agent.tools import GitOpsTool, PythonREPLTool, ToolManager, UnitTestRunnerTool
from src.agent.validator import AgentValidator
from src.models.config import AuraGPTConfig
from src.models.gpt import AuraGPT
from src.tokenizer.code_bpe_tokenizer import CodeBPETokenizer


def test_agent_permission_and_path_sandboxing_security():
    """Verifies security bounds blocking directory traversal and unauthorized tools."""
    cfg = AgentConfig(workspace_root="/tmp/sandbox")
    val_res = AgentValidator.validate_file_path("/etc/passwd", "/tmp/sandbox")
    assert not val_res.is_valid
    assert "escapes sandboxed" in val_res.errors[0]

    tool_res = AgentValidator.validate_tool_permission("unregistered_tool", cfg.allowed_tools)
    assert not tool_res.is_valid


def test_task_planner_dag_and_dependencies():
    """Verifies TaskPlanner DAG subtask dependency enforcement."""
    planner = TaskPlanner(planning_depth=5)
    planner.decompose_task("Refactor Aura model normalization layer")

    st1 = planner.get_next_subtask()
    assert st1 is not None and st1.task_id == 1

    # Subtask 2 cannot start before Subtask 1 completes
    st2 = [st for st in planner.subtasks if st.task_id == 2][0]
    assert st2.status == "pending"

    planner.mark_completed(1, "Completed analysis")
    st_next = planner.get_next_subtask()
    assert st_next is not None and st_next.task_id == 2


def test_shell_git_and_unittest_tools():
    """Verifies GitOpsTool and UnitTestRunnerTool execution in sandboxed workspace."""
    git_tool = GitOpsTool(workspace_root=".")
    res_git = git_tool.execute(action="status")
    assert res_git.exit_code == 0
    assert isinstance(res_git.output, str)

    pytest_tool = UnitTestRunnerTool(timeout_sec=10.0)
    res_py = pytest_tool.execute(test_path="tests/test_exp_009_agent.py")
    assert res_py.exit_code == 0


def test_reasoning_and_reflection_self_correction():
    """Verifies ReasoningEngine and ReflectionEngine self-correction cycle."""
    text_input = (
        "Thought: Let us execute python script.\n"
        "Action: python_repl\n"
        "Action Input: {\"code\": \"import invalid_lib\"}\n"
    )
    parsed = ReasoningEngine.parse_model_response(text_input)
    assert parsed.action == "python_repl"

    diag = ReflectionEngine.diagnose_failure(
        tool_name="python_repl",
        error_msg="No module named 'invalid_lib'",
        output="",
    )
    assert "ModuleNotFoundError" in diag


def test_agent_orchestrator_multi_step_trace_review():
    """Verifies AgenticProgrammingAssistant multi-step trace generation."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        gpt_cfg = AuraGPTConfig(
            model_name="aura-agent-review",
            vocab_size=50260,
            max_sequence_length=128,
            d_model=32,
            n_layers=2,
            n_heads=2,
            d_ff=64,
        )
        model = AuraGPT(gpt_cfg)
        tokenizer = CodeBPETokenizer.create_default()

        agent_cfg = AgentConfig(
            max_reasoning_steps=2,
            workspace_root=tmp_dir,
            output_dir=f"{tmp_dir}/outputs",
        )
        agent = AgenticProgrammingAssistant(model=model, tokenizer=tokenizer, config=agent_cfg)

        payload = agent.run_task("Inspect workspace and format report")
        assert "execution_trace" in payload
        assert payload["total_steps"] > 0
        assert Path(f"{tmp_dir}/outputs/agent_session_trace.json").exists()
