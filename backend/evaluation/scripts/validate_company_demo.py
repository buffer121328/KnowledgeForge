"""Run the offline validator for the curated Chinese company Demo corpus."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[2]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from evaluation.company_demo.corpus import CompanyDemoValidationError, validate_company_demo

DEFAULT_CORPUS = Path(__file__).resolve().parents[1] / "data" / "company-demo"


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for profile validation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus-root",
        type=Path,
        default=DEFAULT_CORPUS,
        help="company-demo profile directory (default: %(default)s)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Validate the profile and print a concise, non-sensitive summary."""
    args = build_parser().parse_args(argv)
    try:
        report = validate_company_demo(args.corpus_root)
    except CompanyDemoValidationError as error:
        print(f"company demo validation failed: {error}", file=sys.stderr)
        return 1

    departments = ", ".join(sorted(report.departments))
    print(
        "company demo validation passed: "
        f"{report.document_count} documents, {report.benchmark_count} questions, departments={departments}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
