"""Score saved retrieval records with selected isolated Ragas metrics."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

BACKEND_ROOT = Path(__file__).resolve().parents[2]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from evaluation.ragas.adapter import (
    CONTEXT_QUALITY_METRICS,
    DEFAULT_RAGAS_CONCURRENCY,
    EVIDENCE_GATE_CONTRACT,
    EXTENDED_RAGAS_METRICS,
    MAX_RAGAS_CONCURRENCY,
    MIN_RAGAS_CONCURRENCY,
    FACTUAL_CORRECTNESS,
    RagasConfigurationError,
    build_openai_context_evaluators,
    build_openai_factual_correctness_evaluator,
    build_openai_extended_evaluators,
    build_openai_faithfulness_evaluator,
    OutcomeKey,
    evaluate_context_quality,
    evaluate_evidence_gate_contract,
    evaluate_factual_correctness,
    evaluate_extended_ragas,
    evaluate_faithfulness,
    validate_ragas_configuration,
)
from evaluation.benchmarks.category_metric_policy import category_metric_policy_identity
from evaluation.benchmarks.evaluation_outcomes import (
    attach_diagnostic_identity,
    outcome_checkpoint_key,
)
from evaluation.ragas.metric_capabilities import RAGAS_PACKAGE_VERSION
from evaluation.runner import _write_text_secure


def _concurrency_argument(value: str) -> int:
    """Parse a conservative Ragas Judge concurrency limit."""
    try:
        concurrency = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("concurrency must be an integer") from error
    if not MIN_RAGAS_CONCURRENCY <= concurrency <= MAX_RAGAS_CONCURRENCY:
        raise argparse.ArgumentTypeError(
            f"concurrency must be between {MIN_RAGAS_CONCURRENCY} and "
            f"{MAX_RAGAS_CONCURRENCY}"
        )
    return concurrency


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the args."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--responses-jsonl",
        type=Path,
        required=True,
        help="Path to a BenchmarkRunner responses.jsonl file.",
    )
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        help="Optional output path beside the saved retrieval snapshot.",
    )
    parser.add_argument(
        "--concurrency",
        type=_concurrency_argument,
        default=DEFAULT_RAGAS_CONCURRENCY,
        help=(
            "Maximum in-flight Ragas Judge evaluations per metric "
            f"(default: {DEFAULT_RAGAS_CONCURRENCY}; range: "
            f"{MIN_RAGAS_CONCURRENCY}-{MAX_RAGAS_CONCURRENCY})."
        ),
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        choices=(
            "faithfulness",
            FACTUAL_CORRECTNESS,
            *CONTEXT_QUALITY_METRICS,
            *EXTENDED_RAGAS_METRICS,
            EVIDENCE_GATE_CONTRACT,
        ),
        default=["faithfulness"],
        help="Ragas metrics to run (default: faithfulness).",
    )
    parser.add_argument(
        "--embedding-model",
        help=(
            "Explicit embedding model for answer relevancy and semantic similarity; "
            "required when either metric is selected."
        ),
    )
    parser.add_argument(
        "--variant-id",
        default="observed",
        help="Bounded diagnostic variant identity (default: observed).",
    )
    return parser.parse_args(argv)


def load_response_records(path: Path) -> list[dict[str, Any]]:
    """Load the response records."""
    if not path.is_file():
        raise ValueError(f"responses file does not exist: {path}")
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"responses line {line_number} is not valid JSON") from error
        if not isinstance(record, dict):
            raise ValueError(f"responses line {line_number} must be a JSON object")
        records.append(record)
    if not records:
        raise ValueError("responses file contains no records")
    return records


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    """Write the jsonl."""
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    content = "".join(
        f"{json.dumps(record, ensure_ascii=False, sort_keys=True)}\n"
        for record in records
    )
    _write_text_secure(path, content)


def _default_output_path(args: argparse.Namespace) -> Path:
    """Preserve the legacy filename and choose explicit context/mixed names."""
    if args.output_jsonl is not None:
        return args.output_jsonl
    selected = set(args.metrics)
    if selected == {"faithfulness"}:
        name = "ragas_faithfulness.jsonl"
    elif selected == {FACTUAL_CORRECTNESS}:
        name = "ragas_factual_correctness.jsonl"
    elif selected.issubset(set(CONTEXT_QUALITY_METRICS)):
        name = "ragas_context_quality.jsonl"
    else:
        name = "ragas_scores.jsonl"
    return args.responses_jsonl.with_name(name)


def _score_summary(
    outcomes: list[dict[str, Any]], *, metric_names: Sequence[str] | None = None
) -> dict[str, Any]:
    """Aggregate only finite scored values by metric for concise comparison."""
    metric_names = list(metric_names or dict.fromkeys(outcome["metric"] for outcome in outcomes))
    metrics: dict[str, Any] = {}
    for metric_name in metric_names:
        selected = [
            outcome for outcome in outcomes if outcome["metric"] == metric_name
        ]
        scores = [
            float(outcome["score"])
            for outcome in selected
            if outcome["status"] == "scored"
            and not isinstance(outcome.get("score"), bool)
            and isinstance(outcome.get("score"), (int, float))
            and math.isfinite(float(outcome["score"]))
        ]
        not_applicable_reasons: dict[str, int] = {}
        skipped_reasons: dict[str, int] = {}
        failed_reasons: dict[str, int] = {}
        failure_exceptions: dict[str, int] = {}
        failure_finish_reasons: dict[str, int] = {}
        batch_fallbacks = 0
        mismatch_by_category: dict[str, int] = {}
        route_mismatches: dict[str, int] = {}
        for outcome in selected:
            reason = outcome.get("reason_code")
            if isinstance(reason, str) and reason:
                target = (
                    not_applicable_reasons if outcome.get("status") == "not_applicable"
                    else skipped_reasons if outcome.get("status") == "skipped"
                    else failed_reasons if outcome.get("status") == "failed"
                    else None
                )
                if target is not None:
                    target[reason] = target.get(reason, 0) + 1
            if outcome.get("status") == "failed":
                exception = outcome.get("exception")
                if isinstance(exception, str) and 0 < len(exception) <= 96:
                    failure_exceptions[exception] = failure_exceptions.get(exception, 0) + 1
                diagnostics = outcome.get("judge_diagnostics")
                finish_reason = diagnostics.get("finish_reason") if isinstance(diagnostics, dict) else None
                if isinstance(finish_reason, str) and 0 < len(finish_reason) <= 32:
                    failure_finish_reasons[finish_reason] = (
                        failure_finish_reasons.get(finish_reason, 0) + 1
                    )
            diagnostics = outcome.get("judge_diagnostics")
            if isinstance(diagnostics, dict) and diagnostics.get("batch_fallback") is True:
                batch_fallbacks += 1
            if metric_name == EVIDENCE_GATE_CONTRACT and outcome.get("status") == "scored" and outcome.get("score") == 0:
                category = outcome.get("category")
                expected_status = outcome.get("expected_response_status")
                observed_status = outcome.get("observed_response_status")
                if isinstance(category, str) and 0 < len(category) <= 64:
                    mismatch_by_category[category] = mismatch_by_category.get(category, 0) + 1
                if all(isinstance(value, str) and 0 < len(value) <= 64 for value in (expected_status, observed_status)):
                    route = f"{expected_status}->{observed_status}"
                    route_mismatches[route] = route_mismatches.get(route, 0) + 1
        metrics[metric_name] = {
            "total": len(selected),
            "scored": sum(outcome["status"] == "scored" for outcome in selected),
            "not_applicable": sum(
                outcome["status"] == "not_applicable" for outcome in selected
            ),
            "failed": sum(outcome["status"] == "failed" for outcome in selected),
            "skipped": sum(outcome["status"] == "skipped" for outcome in selected),
            "not_applicable_reasons": not_applicable_reasons,
            "skip_reasons": skipped_reasons,
            "failure_reasons": failed_reasons,
            "failure_exceptions": failure_exceptions,
            "failure_finish_reasons": failure_finish_reasons,
            "batch_fallbacks": batch_fallbacks,
            "mismatch_by_category": mismatch_by_category,
            "route_mismatches": route_mismatches,
            "mean": round(sum(scores) / len(scores), 6) if scores else None,
        }
    result: dict[str, Any] = {"metrics": metrics}
    if any(isinstance(outcome.get("metric_family"), str) for outcome in outcomes):
        families: dict[str, dict[str, int]] = {}
        for family in sorted(
            {
                str(outcome["metric_family"])
                for outcome in outcomes
                if isinstance(outcome.get("metric_family"), str)
            }
        ):
            selected = [
                outcome for outcome in outcomes if outcome.get("metric_family") == family
            ]
            families[family] = {
                status: sum(outcome.get("status") == status for outcome in selected)
                for status in ("failed", "not_applicable", "scored", "unsupported")
            }
        result["families"] = families
    return result


def _validate_selected_metrics(metrics: Sequence[str]) -> None:
    supported = {
        "faithfulness",
        FACTUAL_CORRECTNESS,
        *CONTEXT_QUALITY_METRICS,
        *EXTENDED_RAGAS_METRICS,
        EVIDENCE_GATE_CONTRACT,
    }
    unknown = [metric for metric in metrics if metric not in supported]
    if unknown:
        raise RagasConfigurationError(f"unsupported_metric:{unknown[0]}")


def _formal_record_index(
    records: Sequence[dict[str, Any]],
) -> dict[tuple[str, str], dict[str, Any]] | None:
    """Return the formal category index, or ``None`` for legacy snapshots."""

    categories = [record.get("category") for record in records]
    present = [isinstance(category, str) and bool(category) for category in categories]
    if not any(present):
        return None
    if not all(present):
        raise RagasConfigurationError("category_identity_incomplete")

    index: dict[tuple[str, str], dict[str, Any]] = {}
    for record in records:
        benchmark_id = record.get("benchmark_id")
        retrieval_mode = record.get("retrieval_mode")
        if not isinstance(benchmark_id, str) or not benchmark_id:
            raise RagasConfigurationError("formal_benchmark_id_invalid")
        if not isinstance(retrieval_mode, str) or not retrieval_mode:
            raise RagasConfigurationError("formal_retrieval_mode_invalid")
        key = (benchmark_id, retrieval_mode)
        if key in index:
            raise RagasConfigurationError("formal_record_identity_duplicate")
        index[key] = record
    return index


def _metric_identity(metric: str) -> tuple[str, str]:
    """Return the independent outcome family and bounded metric version."""

    if metric == EVIDENCE_GATE_CONTRACT:
        return "deterministic", "evidence-gate-contract-v1"
    return "ragas", f"ragas-{RAGAS_PACKAGE_VERSION}:{metric}"


def _outcome_key(outcome: dict[str, Any]) -> OutcomeKey | None:
    return outcome_checkpoint_key(outcome)


def _responses_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resume_state_path(output_path: Path) -> Path:
    return output_path.with_suffix(".resume.json")


def _load_resumable_outcomes(
    *, output_path: Path, responses_sha256: str, metrics: Sequence[str], valid_keys: set[OutcomeKey]
) -> list[dict[str, Any]]:
    """Read only valid scored outcomes from an interrupted matching run."""
    state_path = _resume_state_path(output_path)
    if not output_path.is_file() or not state_path.is_file():
        return []
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state != {"responses_sha256": responses_sha256, "metrics": list(metrics)}:
            return []
        candidates = load_response_records(output_path)
    except (OSError, ValueError, json.JSONDecodeError):
        return []
    reusable: list[dict[str, Any]] = []
    valid_scored: list[tuple[OutcomeKey, dict[str, Any]]] = []
    for outcome in candidates:
        key = _outcome_key(outcome)
        score = outcome.get("score")
        if (
            key is None or key not in valid_keys
            or outcome.get("status") != "scored"
            or isinstance(score, bool)
            or not isinstance(score, (int, float)) or not math.isfinite(float(score))
        ):
            continue
        valid_scored.append((key, outcome))
    duplicate_keys = {
        key for key, count in Counter(key for key, _ in valid_scored).items()
        if count > 1
    }
    for key, outcome in valid_scored:
        if key not in duplicate_keys:
            reusable.append(outcome)
    return reusable


def _write_checkpoint(
    *, output_path: Path, responses_sha256: str, metrics: Sequence[str], outcomes: list[dict[str, Any]]
) -> None:
    write_jsonl(output_path, outcomes)
    _write_text_secure(
        _resume_state_path(output_path),
        json.dumps({"responses_sha256": responses_sha256, "metrics": list(metrics)}, sort_keys=True) + "\n",
    )


async def run(args: argparse.Namespace) -> Path:
    """Evaluate a saved snapshot and resume valid completed Judge outcomes."""
    records = load_response_records(args.responses_jsonl)
    max_concurrency = getattr(args, "concurrency", DEFAULT_RAGAS_CONCURRENCY)
    metrics = list(dict.fromkeys(args.metrics))
    _validate_selected_metrics(metrics)
    output_path = _default_output_path(args)
    responses_sha256 = _responses_sha256(args.responses_jsonl)
    formal_records = _formal_record_index(records)
    diagnostic_identity: dict[str, str] | None = None
    judge_identity: dict[str, str] | None = None
    if formal_records is not None:
        variant_id = getattr(args, "variant_id", "observed")
        if not isinstance(variant_id, str) or not 1 <= len(variant_id) <= 128:
            raise RagasConfigurationError("variant_id_invalid")
        policy_identity = category_metric_policy_identity()
        diagnostic_identity = {
            "category_policy_version": policy_identity["category_policy_version"],
            "category_policy_sha256": policy_identity["category_policy_sha256"],
            "variant_id": variant_id,
            "response_snapshot_sha256": responses_sha256,
        }
        if any(metric != EVIDENCE_GATE_CONTRACT for metric in metrics):
            configuration = validate_ragas_configuration()
            judge_identity = {
                "provider": configuration["source"],
                "model": configuration["model"],
            }

    def decorate(outcome: dict[str, Any]) -> None:
        if formal_records is None or diagnostic_identity is None:
            return
        record_key = (outcome.get("benchmark_id"), outcome.get("retrieval_mode"))
        record = formal_records.get(record_key)  # type: ignore[arg-type]
        if record is None:
            raise RagasConfigurationError("formal_outcome_identity_unknown")
        outcome["category"] = record["category"]
        metric = outcome.get("metric")
        if not isinstance(metric, str):
            raise RagasConfigurationError("formal_metric_identity_invalid")
        family, metric_version = _metric_identity(metric)
        attach_diagnostic_identity(
            outcome,
            metric_family=family,
            metric_version=metric_version,
            identity=diagnostic_identity,
            judge_identity=judge_identity if family == "ragas" else None,
        )

    valid_keys: set[OutcomeKey] = set()
    for record in records:
        benchmark_id = record.get("benchmark_id")
        retrieval_mode = record.get("retrieval_mode")
        if not isinstance(benchmark_id, str) or not isinstance(retrieval_mode, str):
            continue
        for metric in metrics:
            seed = {
                "benchmark_id": benchmark_id,
                "retrieval_mode": retrieval_mode,
                "metric": metric,
            }
            decorate(seed)
            if (key := _outcome_key(seed)) is not None:
                valid_keys.add(key)

    outcomes = _load_resumable_outcomes(
        output_path=output_path,
        responses_sha256=responses_sha256,
        metrics=metrics,
        valid_keys=valid_keys,
    )
    completed_keys: set[OutcomeKey] = set()
    for outcome in outcomes:
        key = _outcome_key(outcome)
        if key is None:
            continue
        completed_keys.add(key)
        completed_keys.add(key[:3])

    def checkpoint(outcome: dict[str, Any]) -> None:
        decorate(outcome)
        key = _outcome_key(outcome)
        if key is None:
            return
        for index, existing in enumerate(outcomes):
            if _outcome_key(existing) == key:
                outcomes[index] = dict(outcome)
                break
        else:
            outcomes.append(dict(outcome))
        _write_checkpoint(
            output_path=output_path,
            responses_sha256=responses_sha256,
            metrics=metrics,
            outcomes=outcomes,
        )
        completed_keys.add(key)
        completed_keys.add(key[:3])

    def merge(new_outcomes: list[dict[str, Any]]) -> None:
        for outcome in new_outcomes:
            decorate(outcome)
            key = _outcome_key(outcome)
            if key is None:
                outcomes.append(outcome)
                continue
            for index, existing in enumerate(outcomes):
                if _outcome_key(existing) == key:
                    if existing.get("status") != "scored":
                        outcomes[index] = outcome
                    break
            else:
                outcomes.append(outcome)

    if "faithfulness" in metrics:
        evaluator = build_openai_faithfulness_evaluator()
        merge(await evaluate_faithfulness(
            records, evaluator=evaluator, max_concurrency=max_concurrency,
            completed_keys=completed_keys, on_completed=checkpoint,
        ))
    if FACTUAL_CORRECTNESS in metrics:
        evaluator = build_openai_factual_correctness_evaluator()
        merge(await evaluate_factual_correctness(
            records, evaluator=evaluator, max_concurrency=max_concurrency,
            completed_keys=completed_keys, on_completed=checkpoint,
        ))
    extended_metrics = [
        metric for metric in metrics if metric in EXTENDED_RAGAS_METRICS
    ]
    if extended_metrics:
        extended_evaluators = build_openai_extended_evaluators(
            extended_metrics,
            embedding_model=getattr(args, "embedding_model", None),
            embedding_api_key=getattr(args, "embedding_api_key", None),
            embedding_base_url=getattr(args, "embedding_base_url", None),
        )
        merge(
            await evaluate_extended_ragas(
                records,
                evaluators=extended_evaluators,
                max_concurrency=max_concurrency,
                completed_keys=completed_keys,
                on_completed=checkpoint,
            )
        )
    context_metrics = [metric for metric in metrics if metric in CONTEXT_QUALITY_METRICS]
    if context_metrics:
        context_evaluators = build_openai_context_evaluators(context_metrics)
        merge(await evaluate_context_quality(
            records, evaluators=context_evaluators, max_concurrency=max_concurrency,
            completed_keys=completed_keys, on_completed=checkpoint,
        ))
    if EVIDENCE_GATE_CONTRACT in metrics:
        merge(evaluate_evidence_gate_contract(
            records,
            completed_keys=completed_keys,
            on_completed=checkpoint,
        ))
    _write_checkpoint(
        output_path=output_path,
        responses_sha256=responses_sha256,
        metrics=metrics,
        outcomes=outcomes,
    )
    _write_text_secure(
        output_path.with_suffix(".summary.json"),
        json.dumps(_score_summary(outcomes, metric_names=metrics), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
    )
    return output_path


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line entry point."""
    args = parse_args(argv)
    output_path = asyncio.run(run(args))
    print(f"Wrote Ragas outcomes to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
