"""CLI Launcher Script for Experiment EXP-008 Performance Benchmarking.

Usage:
    python scripts/run_exp_008_benchmark.py --quantization int8_dynamic --max-new-tokens 32
"""

import argparse
import logging
import sys
from pathlib import Path
import torch

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.config import AuraGPTConfig
from src.models.gpt import AuraGPT
from src.optimization.inference_optimizer import OptimizedInferenceEngine
from src.optimization.optimization_config import PerformanceConfig, QuantizationType
from src.optimization.profiler import PerformanceReporter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Aura EXP-008 Performance Benchmarking")
    parser.add_argument("--quantization", type=str, default="none", choices=["none", "int8_dynamic"], help="Quantization mode")
    parser.add_argument("--max-new-tokens", type=int, default=32, help="Max new tokens to generate")
    parser.add_argument("--batch-size", type=int, default=2, help="Inference batch size")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    qtype = QuantizationType.INT8_DYNAMIC if args.quantization == "int8_dynamic" else QuantizationType.NONE

    config = PerformanceConfig(
        experiment_id="EXP-008_CLI_BENCHMARK",
        quantization_type=qtype,
        use_flash_attention=True,
        use_kv_cache=True,
        use_amp=True,
    )

    logger.info("Initializing AuraGPT base model for EXP-008 benchmarking...")
    gpt_cfg = AuraGPTConfig(
        model_name="aura-opt-base",
        vocab_size=50260,
        max_sequence_length=512,
        d_model=256,
        n_layers=4,
        n_heads=4,
        d_ff=1024,
    )
    base_model = AuraGPT(gpt_cfg)

    engine = OptimizedInferenceEngine(model=base_model, config=config)

    prompt_tokens = torch.randint(0, 50000, (args.batch_size, 16))

    logger.info("Running Latency, Throughput, and VRAM memory profiling...")
    stats = engine.profile(prompt_tokens=prompt_tokens, max_new_tokens=args.max_new_tokens)

    report_path = Path(config.output_dir) / "benchmark_report.json"
    PerformanceReporter.save_report(stats, report_path)

    logger.info("==================================================")
    logger.info("EXP-008 PERFORMANCE BENCHMARKING COMPLETE")
    logger.info("Throughput: %.2f tokens/sec", stats.tokens_per_second)
    logger.info("TTFT: %.2f ms", stats.time_to_first_token_ms)
    logger.info("ITL: %.2f ms", stats.inter_token_latency_ms)
    logger.info("Peak VRAM: %.2f MB", stats.max_allocated_vram_mb)
    logger.info("Report Saved: %s", report_path)
    logger.info("==================================================")


if __name__ == "__main__":
    main()
