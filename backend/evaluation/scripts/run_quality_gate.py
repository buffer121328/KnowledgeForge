"""Build an offline quality report and optionally enforce its release gate."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

BACKEND_ROOT = Path(__file__).resolve().parents[2]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from evaluation.release.quality_gates import (  # noqa: E402
    DEFAULT_SMOKE_THRESHOLD,
    aggregate_quality_report,
    evaluate_release_gate,
)
from evaluation.runner import _write_text_secure  # noqa: E402


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the args."""
    parser = argparse.ArgumentParser(description="Aggregate offline benchmark quality metrics")
    parser.add_argument("--metadata-json", type=Path, required=True, help="Runner run_metadata.json path")
    parser.add_argument("--responses-jsonl", type=Path, required=True, help="Runner responses.jsonl path")
    parser.add_argument("--output-json", type=Path, required=True, help="Owner-only aggregate report path")
    parser.add_argument("--smoke-threshold", type=int, default=DEFAULT_SMOKE_THRESHOLD)
    parser.add_argument("--release-gate", action="store_true", help="Exit non-zero when the report is blocked")
    parser.add_argument(
        "--evidence-thresholds-json",
        type=Path,
        help="JSON object of required evidence-gate metric thresholds",
    )
    return parser.parse_args(argv)


def _load_json(path: Path) -> dict[str, Any]:
    """Load the JSON."""
    if not path.is_file():
        raise ValueError("metadata file does not exist")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("metadata must be a JSON object")
    return value


def _load_thresholds(path: Path | None) -> dict[str, float] | None:
    """Load bounded numeric evidence thresholds from a JSON object."""
    if path is None:
        return None
    value = _load_json(path)
    thresholds: dict[str, float] = {}
    for name, threshold in value.items():
        if (
            not isinstance(name, str)
            or not name
            or isinstance(threshold, bool)
            or not isinstance(threshold, (int, float))
        ):
            raise ValueError("evidence thresholds must map names to numbers")
        thresholds[name] = float(threshold)
    if not thresholds:
        raise ValueError("evidence thresholds must not be empty")
    return thresholds


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load the jsonl."""
    if not path.is_file():
        raise ValueError("responses file does not exist")
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"responses line {line_number} must be an object")
        records.append(value)
    if not records:
        raise ValueError("responses file contains no records")
    return records


def run(args: argparse.Namespace) -> tuple[Path, dict[str, Any] | None]:
    """Run the module operation."""
    metadata = _load_json(args.metadata_json)
    records = _load_jsonl(args.responses_jsonl)
    report = aggregate_quality_report(metadata, records, smoke_threshold=args.smoke_threshold)
    _write_text_secure(
        args.output_json,
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    evidence_thresholds = _load_thresholds(
        getattr(args, "evidence_thresholds_json", None)
    )
    decision = (
        evaluate_release_gate(
            report,
            evidence_thresholds=evidence_thresholds,
        )
        if args.release_gate
        else None
    )
    return args.output_json, decision


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line entry point."""
    args = parse_args(argv)
    try:
        output, decision = run(args)
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        print(f"quality report failed: {type(error).__name__}", file=sys.stderr)
        return 2
    if decision is not None:
        print(f"quality gate {decision['decision']} ({len(decision['reasons'])} reasons)")
        if decision["decision"] == "blocked":
            return 3
    else:
        print(f"quality report written: {output.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
