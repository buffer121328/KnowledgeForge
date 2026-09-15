"""Validate the checked-in Agent benchmark and trace contract offline."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[2]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from domain.knowledge import QAResult, QueryIntent  # noqa: E402
from domain.trace import TraceRecorder  # noqa: E402
from evaluation.benchmarks.agent_benchmark import evaluate_agent_trace, load_agent_benchmark  # noqa: E402

DEFAULT_BENCHMARK = BACKEND_ROOT / "evaluation" / "data" / "agent-benchmark" / "agent_benchmark.jsonl"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Validate the offline Agent benchmark contract")
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    return parser.parse_args(argv)


def run(benchmark_path: Path) -> int:
    """Validate fixture shape and evaluator semantics without external calls."""
    samples = load_agent_benchmark(benchmark_path)
    for sample in samples:
        recorder = TraceRecorder(trace_id=f"contract_{sample.id}")
        for operation in sample.expected_operations:
            recorder.record(operation)
        result = QAResult(
            question=sample.user_input,
            answer="",
            contexts=[],
            intent=QueryIntent.FACTOID,
            confidence=0.0,
            degradation_code="insufficient_verified_evidence" if sample.expected_refusal else None,
            trace=recorder.build(),
        )
        evaluation = evaluate_agent_trace(sample, result)
        if evaluation.get("status") != "passed":
            raise ValueError(f"Agent contract failed for {sample.id}: {evaluation.get('status')}")
    print(f"agent contract passed: {len(samples)} samples")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the offline contract command."""
    args = parse_args(argv)
    try:
        return run(args.benchmark)
    except (OSError, TypeError, ValueError):
        print("agent contract failed: validation_error", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
