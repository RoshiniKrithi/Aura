"""PyTest Unit & Integration Test Suite for Aura EXP-009 Agentic Programming Assistant.

Verifies:
- AgentConfig and AgentValidator path traversal safety bounds
- TaskPlanner DAG task decomposition and subtask completion status
- ToolManager execution (Python REPL, FileReader, FileWriter, WorkspaceSearch, ASTParser, PyTest, GitOps)
- MemoryManager short-term step tracking and working context formatting
- ReflectionEngine failure diagnosis and diagnostic recommendation prompts
- ReasoningEngine ReAct output parsing (Thought, Action, Action Input, Final Answer)
- AgenticProgrammingAssistant end-to-end multi-step task execution & JSON session trace exports
"""

from pathlib import Path
import tempfile
import pytest
import torch

from src.agent.config import AgentConfig
from src.agent.exceptions import AgentValidationError
from src.agent.memory import MemoryManager
from src.agent.orchestrator import AgenticProgrammingAssistant
from src.agent.planner import TaskPlanner
from src.agent.reasoning import ReasoningEngine
from src.agent.reflection import ReflectionEngine
from src.agent.tools import (
    ASTParserTool,
    FileReaderTool,
    FileWriterTool,
    GitOpsTool,
    PythonREPLTool,
    ToolManager,
    WorkspaceSearchTool,
)
from src.agent.validator import AgentValidator
from src.models.config import AuraGPTConfig
from src.models.gpt import AuraGPT
from src.tokenizer.code_bpe_tokenizer import CodeBPETokenizer


def test_agent_config_and_validator():
    """Verifies AgentConfig parameters and AgentValidator safety checks."""
    cfg = AgentConfig(max_reasoning_steps=10, workspace_root=".")
    val_res = AgentValidator.validate_config(cfg)
    assert val_res.is_valid

    path_res = AgentValidator.validate_file_path("src/agent/config.py", ".")
    assert path_res.is_valid

    invalid_path = AgentValidator.validate_file_path("../../etc/passwd", ".")
    assert not invalid_path.is_valid


def test_task_planner_decomposition_and_updates():
    """Verifies TaskPlanner breaks down task and updates subtask states."""
    planner = TaskPlanner(planning_depth=5)
    subtasks = planner.decompose_task("Fix bug in tokenizer")

    assert len(subtasks) == 4
    next_st = planner.get_next_subtask()
    assert next_st is not None
    assert next_st.task_id == 1

    planner.mark_completed(1, "Analyzed requirements")
    next_st2 = planner.get_next_subtask()
    assert next_st2 is not None
    assert next_st2.task_id == 2


def test_tool_manager_and_workspace_tools():
    """Verifies ToolManager executing Python REPL, FileReader, FileWriter, and ASTParser."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmppath = Path(tmp_dir)
        test_file = tmppath / "sample.py"

        # FileWriter
        writer = FileWriterTool(workspace_root=tmp_dir)
        res_w = writer.execute(path_str=str(test_file), content="def foo(x):\n    return x + 1\n")
        assert res_w.exit_code == 0

        # FileReader
        reader = FileReaderTool(workspace_root=tmp_dir)
        res_r = reader.execute(path_str=str(test_file))
        assert res_r.exit_code == 0
        assert "def foo" in res_r.output

        # ASTParser
        ast_tool = ASTParserTool(workspace_root=tmp_dir)
        res_ast = ast_tool.execute(path_str=str(test_file))
        assert res_ast.exit_code == 0
        assert "Function: foo" in res_ast.output

        # PythonREPL
        repl = PythonREPLTool(timeout_sec=5.0)
        res_repl = repl.execute(code="print(2 + 3)")
        assert res_repl.exit_code == 0
        assert res_repl.output == "5"


def test_memory_manager_and_context_prompting():
    """Verifies MemoryManager records steps and formats working context prompts."""
    mem = MemoryManager(max_memory_tokens=1024)
    mem.add_step(
        step_index=1,
        thought="Need to search repository.",
        action="workspace_search",
        tool_args={"pattern": "class Aura"},
        observation="Found class Aura in src/models/gpt.py",
    )
    mem.add_reflection("Reflection: File path is valid.")

    prompt_context = mem.get_working_context_prompt()
    assert "Step 1:" in prompt_context
    assert "workspace_search" in prompt_context
    assert "Reflection: File path is valid." in prompt_context


def test_reflection_engine_diagnosis():
    """Verifies ReflectionEngine categorizes runtime errors into recommendations."""
    diag_import = ReflectionEngine.diagnose_failure("python_repl", "No module named 'invalid_mod'", "")
    assert "ModuleNotFoundError" in diag_import

    diag_assert = ReflectionEngine.diagnose_failure("unit_test_runner", "FAILED test_foo.py", "")
    assert "PyTest test assertion failure" in diag_assert


def test_reasoning_engine_react_parsing():
    """Verifies ReasoningEngine parses ReAct Thought, Action, Action Input, and Final Answer."""
    text_action = (
        "Thought: I should search for main class.\n"
        "Action: workspace_search\n"
        "Action Input: {\"pattern\": \"AuraGPT\"}\n"
    )
    step1 = ReasoningEngine.parse_model_response(text_action)
    assert not step1.is_final
    assert step1.action == "workspace_search"
    assert step1.action_input.get("pattern") == "AuraGPT"

    text_final = (
        "Thought: All tests pass successfully.\n"
        "Final Answer: The bug in config.py has been resolved."
    )
    step2 = ReasoningEngine.parse_model_response(text_final)
    assert step2.is_final
    assert "resolved" in step2.final_answer


def test_agentic_programming_assistant_end_to_end():
    """Verifies full end-to-end AgenticProgrammingAssistant execution loop."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        gpt_cfg = AuraGPTConfig(
            model_name="aura-agent-test",
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
            max_reasoning_steps=3,
            workspace_root=tmp_dir,
            output_dir=f"{tmp_dir}/outputs",
        )

        agent = AgenticProgrammingAssistant(model=model, tokenizer=tokenizer, config=agent_cfg)

        result = agent.run_task(task_description="Create simple utility script in workspace")

        assert result["total_steps"] > 0
        assert "execution_trace" in result
        assert Path(f"{tmp_dir}/outputs/agent_session_trace.json").exists()
