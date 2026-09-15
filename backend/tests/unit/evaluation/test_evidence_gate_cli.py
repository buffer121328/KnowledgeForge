"""CLI acceptance tests for evidence-gate preparation and calibration."""

from __future__ import annotations

import json
from pathlib import Path

from evaluation.scripts.calibrate_evidence_gate import main as calibrate_main
from evaluation.scripts.prepare_evidence_gate_candidates import main as prepare_main


PROJECT_ROOT = Path(__file__).resolve().parents[4]
COMPANY_DEMO_ROOT = PROJECT_ROOT / "backend/evaluation/data/company-demo"


def test_prepare_cli_writes_reviewer_bundle(tmp_path: Path) -> None:
    output = tmp_path / "evidence-gates"

    exit_code = prepare_main(
        [
            "--company-demo-root",
            str(COMPANY_DEMO_ROOT),
            "--output-root",
            str(output),
        ]
    )

    assert exit_code == 0
    assert (output / "manifest.json").is_file()
    assert (output / "fixture-profile.example.json").is_file()


def test_calibration_cli_records_blocked_missing_measurements(tmp_path: Path) -> None:
    report_path = tmp_path / "quality.json"
    report_path.write_text(json.dumps({"evidence_gate": {}}), encoding="utf-8")
    output = tmp_path / "calibration"

    exit_code = calibrate_main(
        [
            "--quality-report",
            str(report_path),
            "--output-dir",
            str(output),
            "--calibration-version",
            "evidence-calibration-2026-08-v1",
        ]
    )

    assert exit_code == 2
    record = json.loads((output / "calibration-record.json").read_text())
    assert record["status"] == "blocked_missing_measurements"
