"""Tool Registry and Execution Infrastructure for Aura Agent (EXP-009).

Provides sandboxed workspace tools:
- PythonREPLTool: Subprocess Python execution with timeout
- FileReaderTool: File reading with line number bounds
- FileWriterTool: File writing within sandboxed workspace_root
- WorkspaceSearchTool: Regex & keyword file content search
- ASTParserTool: Python AST structure & signature extraction
- UnitTestRunnerTool: Automated PyTest test suite execution
- GitOpsTool: Workspace git status and diff inspection
- RAGSearchTool: Knowledge retrieval via RAGOrchestrator
"""

import ast
from dataclasses import dataclass
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Callable, Dict, List, Optional
import logging

from src.agent.exceptions import ToolExecutionError
from src.agent.validator import AgentValidator

logger = logging.getLogger(__name__)


@dataclass
class ToolResult:
    """Standardized tool output result structure."""

    tool_name: str
    output: str
    exit_code: int = 0
    error: Optional[str] = None


class BaseTool:
    """Abstract base class for Agent tools."""

    name: str = "base_tool"
    description: str = "Base tool interface."

    def execute(self, **kwargs) -> ToolResult:
        raise NotImplementedError


class PythonREPLTool(BaseTool):
    """Executes Python code in a sandboxed subprocess with timeout controls."""

    name = "python_repl"
    description = "Executes Python code string and returns stdout/stderr."

    def __init__(self, timeout_sec: float = 30.0) -> None:
        self.timeout_sec = timeout_sec

    def execute(self, code: str, **kwargs) -> ToolResult:
        """Runs python code string in subprocess."""
        try:
            cmd = [sys.executable, "-c", code]
            res = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout_sec,
            )
            out = res.stdout
            if res.stderr:
                out += f"\n[stderr]\n{res.stderr}"
            return ToolResult(
                tool_name=self.name,
                output=out.strip(),
                exit_code=res.returncode,
                error=res.stderr.strip() if res.returncode != 0 else None,
            )
        except subprocess.TimeoutExpired:
            return ToolResult(
                tool_name=self.name,
                output="",
                exit_code=1,
                error=f"Execution timed out after {self.timeout_sec}s.",
            )
        except Exception as e:
            return ToolResult(
                tool_name=self.name, output="", exit_code=1, error=str(e)
            )


class FileReaderTool(BaseTool):
    """Reads workspace text file contents with line limits."""

    name = "file_reader"
    description = "Reads text file content from path_str."

    def __init__(self, workspace_root: str = ".") -> None:
        self.workspace_root = workspace_root

    def execute(self, path_str: str, start_line: int = 1, end_line: Optional[int] = None, **kwargs) -> ToolResult:
        val_res = AgentValidator.validate_file_path(path_str, self.workspace_root)
        if not val_res.is_valid:
            return ToolResult(tool_name=self.name, output="", exit_code=1, error=str(val_res.errors))

        try:
            path = Path(path_str)
            if not path.exists():
                return ToolResult(tool_name=self.name, output="", exit_code=1, error=f"File '{path_str}' does not exist.")

            lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
            sl = max(1, start_line) - 1
            el = end_line if end_line is not None else len(lines)
            content = "\n".join(lines[sl:el])

            return ToolResult(tool_name=self.name, output=content, exit_code=0)
        except Exception as e:
            return ToolResult(tool_name=self.name, output="", exit_code=1, error=str(e))


class FileWriterTool(BaseTool):
    """Writes or overwrites file content within workspace_root bounds."""

    name = "file_writer"
    description = "Writes content string to file at path_str."

    def __init__(self, workspace_root: str = ".") -> None:
        self.workspace_root = workspace_root

    def execute(self, path_str: str, content: str, **kwargs) -> ToolResult:
        val_res = AgentValidator.validate_file_path(path_str, self.workspace_root)
        if not val_res.is_valid:
            return ToolResult(tool_name=self.name, output="", exit_code=1, error=str(val_res.errors))

        try:
            path = Path(path_str)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
            return ToolResult(
                tool_name=self.name,
                output=f"Successfully wrote {len(content)} bytes to '{path_str}'.",
                exit_code=0,
            )
        except Exception as e:
            return ToolResult(tool_name=self.name, output="", exit_code=1, error=str(e))


class WorkspaceSearchTool(BaseTool):
    """Searches workspace files using pattern string."""

    name = "workspace_search"
    description = "Searches codebase files for regex pattern."

    def __init__(self, workspace_root: str = ".") -> None:
        self.workspace_root = workspace_root

    def execute(self, pattern: str, extension: str = ".py", **kwargs) -> ToolResult:
        try:
            matches = []
            regex = re.compile(pattern, re.IGNORECASE)
            root = Path(self.workspace_root)

            for path in root.rglob(f"*{extension}"):
                if any(part.startswith(".") for part in path.parts):
                    continue
                try:
                    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
                    for idx, line in enumerate(lines, 1):
                        if regex.search(line):
                            matches.append(f"{path}:{idx}: {line.strip()}")
                            if len(matches) >= 50:
                                break
                except Exception:
                    continue

            output = "\n".join(matches) if matches else "No matches found."
            return ToolResult(tool_name=self.name, output=output, exit_code=0)
        except Exception as e:
            return ToolResult(tool_name=self.name, output="", exit_code=1, error=str(e))


class ASTParserTool(BaseTool):
    """Parses Python file AST extracting class and function signatures."""

    name = "ast_parser"
    description = "Parses Python AST structure for class/function definitions."

    def __init__(self, workspace_root: str = ".") -> None:
        self.workspace_root = workspace_root

    def execute(self, path_str: str, **kwargs) -> ToolResult:
        val_res = AgentValidator.validate_file_path(path_str, self.workspace_root)
        if not val_res.is_valid:
            return ToolResult(tool_name=self.name, output="", exit_code=1, error=str(val_res.errors))

        try:
            path = Path(path_str)
            if not path.exists():
                return ToolResult(tool_name=self.name, output="", exit_code=1, error=f"File '{path_str}' not found.")

            tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
            nodes_info = []

            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    nodes_info.append(f"Class: {node.name} (Line {node.lineno})")
                elif isinstance(node, ast.FunctionDef):
                    args = [a.arg for a in node.args.args]
                    nodes_info.append(f"  Function: {node.name}({', '.join(args)}) (Line {node.lineno})")

            out = "\n".join(nodes_info) if nodes_info else "No class or function definitions found."
            return ToolResult(tool_name=self.name, output=out, exit_code=0)
        except Exception as e:
            return ToolResult(tool_name=self.name, output="", exit_code=1, error=str(e))


class UnitTestRunnerTool(BaseTool):
    """Runs PyTest test files or functions in subprocess."""

    name = "unit_test_runner"
    description = "Executes pytest test suite on specified test_path."

    def __init__(self, timeout_sec: float = 30.0) -> None:
        self.timeout_sec = timeout_sec

    def execute(self, test_path: str = "tests/", **kwargs) -> ToolResult:
        try:
            cmd = [sys.executable, "-m", "pytest", test_path, "-q", "--no-cov"]
            res = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout_sec,
            )
            out = res.stdout
            if res.stderr:
                out += f"\n[stderr]\n{res.stderr}"
            return ToolResult(
                tool_name=self.name,
                output=out.strip(),
                exit_code=res.returncode,
                error=res.stderr.strip() if res.returncode != 0 else None,
            )
        except subprocess.TimeoutExpired:
            return ToolResult(
                tool_name=self.name,
                output="",
                exit_code=1,
                error=f"PyTest runner timed out after {self.timeout_sec}s.",
            )
        except Exception as e:
            return ToolResult(tool_name=self.name, output="", exit_code=1, error=str(e))


class GitOpsTool(BaseTool):
    """Inspects workspace git status or git diff."""

    name = "git_ops"
    description = "Runs git status or git diff commands in workspace."

    def __init__(self, workspace_root: str = ".") -> None:
        self.workspace_root = workspace_root

    def execute(self, action: str = "status", **kwargs) -> ToolResult:
        try:
            cmd = ["git", action] if action in ("status", "diff") else ["git", "status"]
            res = subprocess.run(
                cmd,
                cwd=self.workspace_root,
                capture_output=True,
                text=True,
                timeout=10.0,
            )
            return ToolResult(
                tool_name=self.name,
                output=res.stdout.strip(),
                exit_code=res.returncode,
                error=res.stderr.strip() if res.returncode != 0 else None,
            )
        except Exception as e:
            return ToolResult(tool_name=self.name, output="", exit_code=1, error=str(e))


class RAGSearchTool(BaseTool):
    """Queries codebase documentation using Aura RAGOrchestrator."""

    name = "rag_search"
    description = "Queries RAGOrchestrator for relevant codebase documentation."

    def __init__(self, rag_orchestrator: Optional[Any] = None) -> None:
        self.rag_orchestrator = rag_orchestrator

    def execute(self, query: str, top_k: int = 3, **kwargs) -> ToolResult:
        if self.rag_orchestrator is None:
            return ToolResult(
                tool_name=self.name,
                output="RAGOrchestrator is not attached to Agent.",
                exit_code=0,
            )
        try:
            res = self.rag_orchestrator.search(query=query, top_k=top_k)
            formatted = "\n---\n".join([f"Doc ID: {r.doc_id}\nContent: {r.content[:200]}" for r in res])
            return ToolResult(tool_name=self.name, output=formatted, exit_code=0)
        except Exception as e:
            return ToolResult(tool_name=self.name, output="", exit_code=1, error=str(e))


class ToolManager:
    """Central registry and executor for all Agent tools."""

    def __init__(
        self,
        workspace_root: str = ".",
        timeout_sec: float = 30.0,
        rag_orchestrator: Optional[Any] = None,
    ) -> None:
        """Initializes ToolManager registering default tools."""
        self.workspace_root = workspace_root
        self.timeout_sec = timeout_sec
        self.tools: Dict[str, BaseTool] = {}

        # Register default tools
        self.register_tool(PythonREPLTool(timeout_sec=timeout_sec))
        self.register_tool(FileReaderTool(workspace_root=workspace_root))
        self.register_tool(FileWriterTool(workspace_root=workspace_root))
        self.register_tool(WorkspaceSearchTool(workspace_root=workspace_root))
        self.register_tool(ASTParserTool(workspace_root=workspace_root))
        self.register_tool(UnitTestRunnerTool(timeout_sec=timeout_sec))
        self.register_tool(GitOpsTool(workspace_root=workspace_root))
        self.register_tool(RAGSearchTool(rag_orchestrator=rag_orchestrator))

    def register_tool(self, tool: BaseTool) -> None:
        """Registers a tool instance."""
        self.tools[tool.name] = tool
        logger.info("Registered tool '%s'.", tool.name)

    def execute_tool(self, tool_name: str, **kwargs) -> ToolResult:
        """Executes a tool by name with arguments."""
        if tool_name not in self.tools:
            return ToolResult(
                tool_name=tool_name,
                output="",
                exit_code=1,
                error=f"Tool '{tool_name}' is not registered in ToolManager.",
            )
        logger.info("Executing tool '%s' with args: %s", tool_name, kwargs)
        return self.tools[tool_name].execute(**kwargs)
