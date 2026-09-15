"""对离线评测结果产物的只读访问。

评测结果根目录（``backend/evaluation/results/``）是一个离线、Git 忽略的产物目录，
由 ``BenchmarkRunner``、``run_ragas_benchmark.py`` 和 ``run_quality_gate.py``
写入。本模块为仅管理员可见的结果看板提供严格且容错的读取器：run id 只能
通过服务端持有的直接与发布阶段产物索引解析；每个文件独立读取，单个损坏
产物不会导致整个请求失败；响应绝不包含绝对路径或完整上下文正文。
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from evaluation.release.quality_gates import _safe_metadata

# BenchmarkRunner 写入 `20260801T120000Z-<8位十六进制>`，分块 runner 写入
# `chunking-20260801T120000Z-<8位十六进制>`；只有这些形状才是合法 run id。
RUN_ID_PATTERN = re.compile(r"^(?:[a-z]+-)?\d{8}T\d{6}Z-[0-9a-fA-F]{8}$")  # 合法 run id 的形状
_RELEASE_WORKFLOW_ID_PATTERN = re.compile(r"^erw_[0-9a-f]{32}$")  # 发布工作流目录名形状
_RELEASE_STAGE_DIR_PATTERN = re.compile(r"^(baseline|shadow)-[1-9]\d*$")  # 发布阶段目录名形状（如 baseline-1）
_RELEASE_STAGE_MODES = {"baseline": "off", "shadow": "shadow"}  # 阶段对应的证据门模式目录名
DEFAULT_RESPONSE_LIMIT = 200  # 单条回答文本的默认截断长度（字符）
MAX_PAGE_SIZE = 100  # 分页页大小上限


class EvaluationResultError(ValueError):
    """当评测结果产物无法被安全读取（非法 run id、越界或歧义）时抛出。"""


def _contained_directory(root: Path, candidate: Path) -> Path | None:
    """仅当 candidate 是非符号链接目录且解析后仍位于 root 之内时返回其路径，否则返回 None。

    Args:
        root: 已解析的结果根目录。
        candidate: 待检查的目录。
    """
    # 安全关卡：符号链接与普通文件一律拒绝，防止借链接逃出 root
    if candidate.is_symlink() or not candidate.is_dir():
        return None
    try:
        resolved = candidate.resolve(strict=True)
        # relative_to 失败即说明越出 root
        resolved.relative_to(root)
    except (OSError, ValueError):
        return None
    return resolved


def _run_directories(results_root: Path) -> dict[str, list[Path]]:
    """构建标准 run 与发布阶段 run 产物的有界索引（run id -> 目录列表）。

    Args:
        results_root: 评测结果根目录。
    """
    root = results_root.resolve()
    if not root.is_dir():
        return {}
    indexed: dict[str, list[Path]] = {}

    def add(run_id: str, candidate: Path) -> None:
        """把通过 root 包含校验的目录登记到该 run id 的索引列表。

        Args:
            run_id: run 标识。
            candidate: 待登记的候选目录。
        """
        resolved = _contained_directory(root, candidate)
        if resolved is not None:
            indexed.setdefault(run_id, []).append(resolved)

    # ① 索引根目录下形状合法的直接子目录
    for candidate in root.iterdir():
        if RUN_ID_PATTERN.fullmatch(candidate.name):
            add(candidate.name, candidate)

    # ② 索引 release-workflows/<工作流>/<阶段>/<模式>/ 下的 run 目录
    release_root = root / "release-workflows"
    if not release_root.is_dir() or release_root.is_symlink():
        return indexed
    for workflow_dir in release_root.iterdir():
        if not _RELEASE_WORKFLOW_ID_PATTERN.fullmatch(workflow_dir.name):
            continue
        if _contained_directory(root, workflow_dir) is None:
            continue
        for stage_dir in workflow_dir.iterdir():
            matched = _RELEASE_STAGE_DIR_PATTERN.fullmatch(stage_dir.name)
            if matched is None or _contained_directory(root, stage_dir) is None:
                continue
            # 每个阶段只索引其模式目录（baseline->off, shadow->shadow）
            mode_dir = stage_dir / _RELEASE_STAGE_MODES[matched.group(1)]
            if _contained_directory(root, mode_dir) is None:
                continue
            for candidate in mode_dir.iterdir():
                if RUN_ID_PATTERN.fullmatch(candidate.name):
                    add(candidate.name, candidate)
    return indexed


def _release_stage(entry: Path, root: Path) -> str | None:
    """解析发布阶段 run 所属的阶段名（baseline/shadow），非发布 run 返回 None。

    Args:
        entry: 已解析的单个 run 目录。
        root: 已解析的结果根目录。

    返回阶段名或 None；不通过安全校验的路径一律视为非发布 run。
    """
    # release-workflows/<workflow_id>/<stage>-<n>/<mode>/<run_id>
    stage_mode_dir = entry.parent  # <mode>（off/shadow）
    stage_dir = stage_mode_dir.parent  # <stage>-<n>
    if not _RELEASE_STAGE_DIR_PATTERN.fullmatch(stage_dir.name):
        return None
    if _RELEASE_STAGE_MODES.get(stage_dir.name.partition("-")[0]) != stage_mode_dir.name:
        return None
    workflow_dir = stage_dir.parent
    if not _RELEASE_WORKFLOW_ID_PATTERN.fullmatch(workflow_dir.name):
        return None
    if workflow_dir.parent != root / "release-workflows":
        return None
    return stage_dir.name.partition("-")[0]


def _resolve_run_dir(results_root: Path, run_id: str) -> Path:
    """解析唯一被索引的 run 目录，拒绝非法 id、不存在与同名歧义。

    Args:
        results_root: 评测结果根目录。
        run_id: run 标识。
    """
    # 安全关卡：id 必须先通过形状校验，杜绝路径拼接穿越
    if not RUN_ID_PATTERN.fullmatch(run_id):
        raise EvaluationResultError("invalid run_id")
    candidates = _run_directories(results_root).get(run_id, [])
    if not candidates:
        raise EvaluationResultError("run does not exist")
    # 同名目录出现多处视为歧义，拒绝猜测
    if len(candidates) != 1:
        raise EvaluationResultError("ambiguous run id")
    return candidates[0]


def _read_json(path: Path) -> dict[str, Any] | None:
    """读取 JSON 对象文件；文件缺失或格式损坏时返回 None 而非抛错。

    Args:
        path: 目标文件路径。
    """
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _read_jsonl(path: Path) -> tuple[list[dict[str, Any]], int]:
    """逐行读取 JSONL 对象，跳过并计数损坏行，返回 (记录列表, 跳过行数)。

    Args:
        path: 目标 JSONL 文件路径。
    """
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError):
        return [], 0
    records: list[dict[str, Any]] = []
    skipped = 0
    for line in lines:
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            skipped += 1  # 单行损坏只计数跳过，不影响其余记录
            continue
        if isinstance(value, dict):
            records.append(value)
        else:
            skipped += 1  # 非对象行同样视为损坏
    return records, skipped


def _load_ragas_scores(run_dir: Path) -> dict[tuple[str, str], dict[str, float]]:
    """把 (benchmark_id, retrieval_mode) 映射到逐指标的 Ragas 分数字典。

    Args:
        run_dir: run 目录（读取其中的 ragas_*.jsonl 文件）。
    """
    scores: dict[tuple[str, str], dict[str, float]] = {}
    for path in sorted(run_dir.glob("ragas_*.jsonl")):
        if path.name.endswith(".summary.json"):
            continue  # 摘要文件另行处理，这里只读明细
        records, _ = _read_jsonl(path)
        for outcome in records:
            benchmark_id = outcome.get("benchmark_id")
            mode = outcome.get("retrieval_mode")
            metric = outcome.get("metric")
            score = outcome.get("score")
            # 字段类型不合规的行直接忽略
            if not isinstance(benchmark_id, str) or not isinstance(mode, str):
                continue
            if not isinstance(metric, str) or not isinstance(score, (int, float)):
                continue
            scores.setdefault((benchmark_id, mode), {})[metric] = round(float(score), 6)
    return scores


def _load_ragas_metric_states(
    run_dir: Path,
) -> dict[tuple[str, str], dict[str, tuple[str, str | None]]]:
    """把 (benchmark_id, retrieval_mode) 映射到逐指标 (状态, 原因码)。

    覆盖 scored / not_applicable / failed 全部状态，让前端能区分
    "按评审契约不适用"与"应评未评"。损坏行跳过，不抛错。

    Args:
        run_dir: run 目录（读取其中的 ragas_*.jsonl 文件）。
    """
    states: dict[tuple[str, str], dict[str, tuple[str, str | None]]] = {}
    for path in sorted(run_dir.glob("ragas_*.jsonl")):
        if path.name.endswith(".summary.json"):
            continue
        records, _ = _read_jsonl(path)
        for outcome in records:
            benchmark_id = outcome.get("benchmark_id")
            mode = outcome.get("retrieval_mode")
            metric = outcome.get("metric")
            status = outcome.get("status")
            if (
                not isinstance(benchmark_id, str)
                or not isinstance(mode, str)
                or not isinstance(metric, str)
                or not isinstance(status, str)
            ):
                continue
            reason = outcome.get("reason_code")
            states.setdefault((benchmark_id, mode), {})[metric] = (
                status,
                reason if isinstance(reason, str) else None,
            )
    return states


def _record_summary(
    record: dict[str, Any],
    *,
    response_limit: int,
    scores: dict[tuple[str, str], dict[str, float]],
    metric_states: dict[tuple[str, str], dict[str, tuple[str, str | None]]] | None = None,
) -> dict[str, Any]:
    """构造有界的单条记录摘要，不包含完整上下文正文与绝对路径。

    Args:
        record: responses.jsonl 中的单条记录。
        response_limit: 回答文本的截断长度（字符）。
        scores: (benchmark_id, retrieval_mode) -> 指标分数映射。
    """
    # contexts 只保留有限字段，不外泄正文
    contexts = [
        {
            "rank": context.get("rank"),
            "source": context.get("source"),
            "score": context.get("score"),
            "retrieval_type": context.get("retrieval_type"),
            "source_document_id": context.get("source_document_id"),
        }
        for context in record.get("contexts", [])
        if isinstance(context, dict)
    ]
    response = record.get("response")
    # 超长回答截断并加省略号标记
    if isinstance(response, str) and len(response) > response_limit:
        response = response[:response_limit] + "…"
    key = (str(record.get("benchmark_id", "")), str(record.get("retrieval_mode", "")))
    states = (metric_states or {}).get(key, {})
    # 按指标输出 {状态, 原因码}；契约性 not_applicable 与 failed 都如实透出，
    # 让前端能区分"按评审契约不评分"与"应评未评"。
    ragas_metric_states = {
        metric: {"status": status, "reason_code": reason}
        for metric, (status, reason) in sorted(states.items())
    }
    return {
        "benchmark_id": record.get("benchmark_id"),
        "retrieval_mode": record.get("retrieval_mode"),
        "category": record.get("category"),
        "status": record.get("status"),
        "exception": record.get("exception"),
        "refused": record.get("refused"),
        "latency_ms": record.get("latency_ms"),
        "question": record.get("question"),
        "response": response,
        "context_count": len(contexts),
        "contexts": contexts[:3],  # 最多返回前 3 条上下文
        "retrieved_context_ids": list(record.get("retrieved_context_ids") or []),
        "reference_context_ids": list(record.get("reference_context_ids") or []),
        "ragas_scores": scores.get(key, {}),
        "ragas_metric_states": ragas_metric_states,
    }


def list_runs(results_root: str | Path) -> list[dict[str, Any]]:
    """列出全部评测 run 及其有界摘要，按开始时间倒序排列。

    发布评测 baseline 阶段（校准对照）的 run 不进入该列表，只保留 shadow
    与根目录标准离线 run；baseline 产物仍可通过 run id 直接访问，服务发布
    评测 attempt 深链与哈希审计。

    Args:
        results_root: 评测结果根目录。
    """
    root = Path(results_root)
    if not root.is_dir():
        return []
    runs: list[dict[str, Any]] = []
    # 索引产物均为已解析路径，阶段归属判定需要同等解析的根目录
    resolved_root = root.resolve()
    for run_id, candidates in _run_directories(root).items():
        # 同名多处（歧义）的 run 一律跳过，不猜测具体目录
        if len(candidates) != 1:
            continue
        entry = candidates[0]
        # 发布阶段 run 按目录形状归属阶段；baseline 是校准对照，不在列表展示
        stage = _release_stage(entry, resolved_root)
        if stage == "baseline":
            continue
        # 每个文件独立读取：单个产物损坏只影响对应字段，不让整个请求失败
        metadata = _read_json(entry / "run_metadata.json")
        report = _read_json(entry / "quality-report.json") or _read_json(entry / "quality_report.json")
        records, skipped = _read_jsonl(entry / "responses.jsonl")
        runs.append(
            {
                "run_id": run_id,
                "started_at": metadata.get("started_at") if metadata else None,
                "incomplete": metadata is None,  # 元数据缺失视为未完成 run
                "retrieval_modes": list(metadata.get("retrieval_modes") or [])
                if metadata
                else [],
                "run_classification": report.get("run_classification")
                if report
                else None,
                "record_count": len(records),
                "completed": sum(1 for r in records if r.get("status") == "succeeded"),
                "failed": sum(1 for r in records if r.get("status") == "failed"),
                "invalid_provenance": sum(
                    1 for r in records if r.get("status") == "invalid_provenance"
                ),
                "skipped_lines": skipped,
            }
        )
    # 无 started_at 的排最后；同为空字符串按稳定次序
    runs.sort(key=lambda run: run["started_at"] or "", reverse=True)
    return runs


def load_run(results_root: str | Path, run_id: str) -> dict[str, Any]:
    """加载单个 run 的元数据、质量报告与逐文件 Ragas 摘要。

    Args:
        results_root: 评测结果根目录。
        run_id: run 标识。
    """
    run_dir = _resolve_run_dir(Path(results_root), run_id)
    # 元数据经 _safe_metadata 脱敏后才返回，避免泄露敏感配置
    metadata = _read_json(run_dir / "run_metadata.json")
    report = _read_json(run_dir / "quality-report.json") or _read_json(run_dir / "quality_report.json")
    ragas_summaries: dict[str, Any] = {}
    for path in sorted(run_dir.glob("ragas_*.summary.json")):
        summary = _read_json(path)
        if summary is not None:
            ragas_summaries[path.name.removesuffix(".summary.json")] = summary
    return {
        "run_id": run_id,
        "incomplete": metadata is None,
        "metadata": _safe_metadata(metadata) if metadata else {},
        "quality_report": report,
        "ragas_summaries": ragas_summaries,
    }


def list_records(
    results_root: str | Path,
    run_id: str,
    *,
    page: int = 1,
    page_size: int = 20,
    response_limit: int = DEFAULT_RESPONSE_LIMIT,
) -> dict[str, Any]:
    """返回单个 run 中按分页组织的逐问题记录页（字段有界）。

    Args:
        results_root: 评测结果根目录。
        run_id: run 标识。
        page: 页码，从 1 开始。
        page_size: 每页记录数，不得超过 MAX_PAGE_SIZE。
        response_limit: 回答文本截断长度（字符），不允许为负。
    """
    # 安全关卡：分页参数必须落在合法区间
    if page < 1 or page_size < 1 or page_size > MAX_PAGE_SIZE or response_limit < 0:
        raise EvaluationResultError("invalid pagination parameters")
    run_dir = _resolve_run_dir(Path(results_root), run_id)
    records, skipped = _read_jsonl(run_dir / "responses.jsonl")
    scores = _load_ragas_scores(run_dir)
    metric_states = _load_ragas_metric_states(run_dir)
    total = len(records)
    # 切片分页：页码超出范围时自然返回空页
    start = (page - 1) * page_size
    page_records = records[start : start + page_size]
    return {
        "run_id": run_id,
        "page": page,
        "page_size": page_size,
        "total": total,
        "skipped_lines": skipped,
        "records": [
            _record_summary(
                record,
                response_limit=response_limit,
                scores=scores,
                metric_states=metric_states,
            )
            for record in page_records
        ],
    }
