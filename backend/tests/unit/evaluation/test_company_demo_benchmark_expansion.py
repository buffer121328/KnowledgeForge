"""Tests for deterministic fourth-department benchmark expansion."""

from __future__ import annotations

import json
import shutil
from collections import Counter
from pathlib import Path

from evaluation.company_demo.benchmark_expansion import expand_company_demo_benchmark

PROJECT_ROOT = Path(__file__).resolve().parents[4]
SOURCE = PROJECT_ROOT / "backend/evaluation/data/company-demo"


def test_expansion_preserves_seed_and_generates_fifty_governed_rows(tmp_path: Path) -> None:
    target = tmp_path / "company-demo"
    shutil.copytree(SOURCE, target)
    original = (target / "benchmark.jsonl").read_text(encoding="utf-8").splitlines()[:30]

    report = expand_company_demo_benchmark(target)
    rows = [json.loads(line) for line in (target / "benchmark.jsonl").read_text().splitlines()]

    assert report == {"seed_count": 30, "generated_count": 50, "total_count": 80}
    assert [json.dumps(json.loads(line), ensure_ascii=False, sort_keys=True) for line in original] == [
        json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows[:30]
    ]
    assert len({row["id"] for row in rows}) == 80
    generated = rows[30:]
    assert all(row["evidence_gate"] for row in generated)
    counts = Counter(row["evidence_gate"]["category"] for row in generated)
    assert counts["fully_answerable"] == 10
    assert all(count == 4 for category, count in counts.items() if category != "fully_answerable")
