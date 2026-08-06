"""CLI Launcher for Aura EXP-009 Agentic Programming Assistant Pipeline.

Executes autonomous software engineering task benchmarks using AgenticProgrammingAssistant.
"""

import argparse
import logging
import json
from pathlib import Path
import sys

from src.agent.config import AgentConfig
from src.agent.orchestrator import AgenticProgrammingAssistant
from src.models.config import AuraGPTConfig
from src.models.gpt import AuraGPT
from src.tokenizer.code_bpe_tokenizer import CodeBPETokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aura EXP-009 Agentic Programming Assistant Launcher")
    parser.add_argument("--task", type=str, default="Fix syntax error and run tests in src/utils/config.py", help="Task description string")
    parser.add_argument("--max-steps", type=int, default=10, help="Maximum reasoning steps limit")
    parser.add_argument("--workspace", type=str, default=".", help="Target workspace root path")
    parser.add_argument("--output-dir", type=str, default="outputs/experiments/EXP-009_Agent_v1.0", help="Directory for trace output")

    args = parser.parse_args()

    logger.info("Initializing Aura Model & Tokenizer for Agent...")
    gpt_config = AuraGPTConfig(
        model_name="aura-agent-cli",
        vocab_size=50260,
        max_sequence_length=128,
        d_model=32,
        n_layers=2,
        n_heads=2,
        d_ff=64,
    )
    model = AuraGPT(gpt_config)
    tokenizer = CodeBPETokenizer.create_default()

    agent_config = AgentConfig(
        max_reasoning_steps=args.max_steps,
        workspace_root=args.workspace,
        output_dir=args.output_dir,
    )

    agent = AgenticProgrammingAssistant(model=model, tokenizer=tokenizer, config=agent_config)

    logger.info("Running Autonomous Task: '%s'...", args.task)
    results = agent.run_task(task_description=args.task)

    print("\n" + "=" * 50)
    print("AGENT EXECUTION RESULTS SUMMARY")
    print("=" * 50)
    print(f"Task: {results['task']}")
    print(f"Total Steps: {results['total_steps']}")
    print(f"Execution Time: {results['execution_time_sec']}s")
    print(f"Final Solution:\n{results['final_solution']}")
    print("=" * 50)


if __name__ == "__main__":
    main()
