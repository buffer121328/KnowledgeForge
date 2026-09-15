"""Generate fixed, recursive, and semantic retrieval snapshots for one corpus."""

from __future__ import annotations

import argparse
import asyncio
import sys
from collections.abc import Sequence
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[2]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from agents.document_parser import DocParserAgent  # noqa: E402
from evaluation.benchmarks.chunking_benchmark import (  # noqa: E402
    SUPPORTED_CHUNKING_STRATEGIES,
    ChunkingBenchmarkRunner,
    ChunkingBenchmarkValidationError,
)
from langchain_openai import OpenAIEmbeddings  # noqa: E402
from shared.config import settings  # noqa: E402

DEFAULT_CORPUS_ROOT = BACKEND_ROOT / "evaluation" / "data" / "company-demo"
DEFAULT_RESULTS_ROOT = BACKEND_ROOT / "evaluation" / "results" / "chunking"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse controlled offline chunking benchmark arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-root", type=Path, default=DEFAULT_CORPUS_ROOT)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--benchmark", type=Path)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument(
        "--strategies",
        nargs="+",
        choices=SUPPORTED_CHUNKING_STRATEGIES,
        default=list(SUPPORTED_CHUNKING_STRATEGIES),
    )
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args(argv)
    args.manifest = args.manifest or args.corpus_root / "corpus_manifest.json"
    args.benchmark = args.benchmark or args.corpus_root / "benchmark.jsonl"
    return args


def build_runner(results_root: Path) -> ChunkingBenchmarkRunner:
    """Build real DashScope-compatible collaborators without logging secrets."""
    if not settings.dashscope_api_key:
        raise ChunkingBenchmarkValidationError(
            "DASHSCOPE_API_KEY is required for the chunking benchmark"
        )
    embeddings = OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.dashscope_api_key,
        base_url=settings.dashscope_base_url,
        check_embedding_ctx_length=False,
        chunk_size=settings.embedding_batch_size,
    )
    chunker = DocParserAgent(embeddings=embeddings, embedding_cache=None)
    return ChunkingBenchmarkRunner(
        embeddings=embeddings,
        chunker=chunker,
        results_root=results_root,
        embedding_model=settings.embedding_model,
    )


async def run(args: argparse.Namespace) -> Path:
    """Run the offline benchmark and return its saved retrieval snapshot path."""
    runner = build_runner(args.results_root)
    result = await runner.run(
        corpus_root=args.corpus_root,
        manifest_path=args.manifest,
        benchmark_path=args.benchmark,
        strategies=tuple(args.strategies),
        top_k=args.top_k,
    )
    return result.snapshot_path


def main(argv: Sequence[str] | None = None) -> int:
    """Execute the CLI with a concise fail-closed validation boundary."""
    args = parse_args(argv)
    try:
        output_path = asyncio.run(run(args))
    except ChunkingBenchmarkValidationError as error:
        print(f"chunking benchmark failed: {error}", file=sys.stderr)
        return 1
    print(f"Wrote chunking retrieval snapshot to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
