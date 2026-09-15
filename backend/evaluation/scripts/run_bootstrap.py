"""Run the approved paired 1,000-resample offline bootstrap.

Input JSONL records must contain a stable ``question_id`` and reviewed numeric
``vector_score``/``hybrid_score`` fields. The command never reads raw answers,
contexts, or PDFs and writes only the aggregate interval.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

BACKEND_ROOT = Path(__file__).resolve().parents[2]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from evaluation.bootstrap import (  # noqa: E402
    DEFAULT_BOOTSTRAP_RESAMPLES,
    MIN_BOOTSTRAP_PAIRS,
    paired_bootstrap_mean_difference,
)
from evaluation.runner import _write_text_secure  # noqa: E402


def _records(path: Path) -> list[dict[str, Any]]:
    """Return the records."""
    if not path.is_file():
        raise ValueError("bootstrap input does not exist")
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"line {number} must be an object")
        question_id = value.get("question_id")
        if not isinstance(question_id, str) or not question_id or question_id in seen:
            raise ValueError(f"line {number} has an invalid or duplicate question_id")
        seen.add(question_id)
        result.append(value)
    if not result:
        raise ValueError("bootstrap input contains no records")
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the args."""
    parser = argparse.ArgumentParser(description="Run a paired offline bootstrap")
    parser.add_argument("--scores-jsonl", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--resamples", type=int, default=DEFAULT_BOOTSTRAP_RESAMPLES)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-pairs", type=int, default=MIN_BOOTSTRAP_PAIRS)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the command-line entry point."""
    args = parse_args(argv)
    try:
        records = _records(args.scores_jsonl)
        report = paired_bootstrap_mean_difference(
            [record["vector_score"] for record in records],
            [record["hybrid_score"] for record in records],
            resamples=args.resamples,
            seed=args.seed,
            min_pairs=args.min_pairs,
        )
        _write_text_secure(args.output_json, json.dumps(report, indent=2, sort_keys=True) + "\n")
    except (OSError, TypeError, ValueError, KeyError, json.JSONDecodeError) as error:
        print(f"bootstrap failed: {type(error).__name__}", file=sys.stderr)
        return 2
    print(f"bootstrap report written: {args.output_json.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
