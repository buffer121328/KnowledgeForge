"""Generate normalized TXT files for the curated company Demo corpus."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[2]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from evaluation.company_demo.text_cleaning import CleaningValidationError, clean_company_demo  # noqa: E402

DEFAULT_CORPUS = BACKEND_ROOT / "evaluation" / "data" / "company-demo"


def build_parser() -> argparse.ArgumentParser:
    """Build command-line arguments for company Demo cleaning."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-root", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--normalized-root", type=Path)
    parser.add_argument("--report-path", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the offline cleaner and print only aggregate, non-sensitive counts."""
    args = build_parser().parse_args(argv)
    try:
        report = clean_company_demo(
            args.corpus_root,
            normalized_root=args.normalized_root,
            report_path=args.report_path,
        )
    except CleaningValidationError as error:
        print(f"company demo cleaning failed: {error}", file=sys.stderr)
        return 1
    print(
        "company demo cleaning passed: "
        f"{report.document_count} documents, {report.output_characters} characters, "
        f"{report.paragraph_count} paragraphs, {report.table_row_count} table rows, "
        f"{report.placeholder_hits} placeholder types"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
