"""Build a bounded evidence-gate calibration draft from measured quality reports."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

BACKEND_ROOT = Path(__file__).resolve().parents[2]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from evaluation.evidence_gate.calibration import build_calibration_draft


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quality-report", type=Path, action="append", required=True
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--calibration-version", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = build_calibration_draft(
        args.quality_report,
        output_dir=args.output_dir,
        calibration_version=args.calibration_version,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0 if result["measured"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
