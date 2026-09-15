"""Prometheus 指标定义

包含三类指标:
  1. API 指标   - 由 prometheus-fastapi-instrumentator 自动采集
  2. 业务指标   - QA、文档入库、向量库、知识图谱
  3. LLM 指标   - LLM 调用次数与延迟

使用方式:
  from utils.metrics import qa_query_total, llm_api_latency
  qa_query_total.labels(intent="factoid").inc()
  with llm_api_latency.labels(model="gpt-4").time():
      response = await llm.ainvoke(...)
"""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

# ── API 指标（部分由 instrumentator 自动采集，这里补充业务相关） ──────

api_requests_total = Counter(
    name="api_requests_total",
    documentation="Total API requests",
    labelnames=["method", "endpoint", "status_code"],
)

api_request_duration = Histogram(
    name="api_request_duration_seconds",
    documentation="API request duration in seconds",
    labelnames=["method", "endpoint"],
    buckets=(0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)

api_errors_total = Counter(
    name="api_errors_total",
    documentation="Total API errors (5xx)",
    labelnames=["endpoint", "status_code"],
)

rate_limit_events_total = Counter(
    name="rate_limit_events_total",
    documentation="Sensitive-route rate limit denials and storage failures",
    labelnames=["result", "credential_kind"],
)

guardrail_decisions_total = Counter(
    name="guardrail_decisions_total",
    documentation="Content guardrail decisions with fixed safe labels",
    labelnames=["stage", "decision", "rule_version"],
)

guardrail_validation_latency_seconds = Histogram(
    name="guardrail_validation_latency_seconds",
    documentation="Optional content validator latency",
    labelnames=["stage", "validator"],
)

guardrail_validation_errors_total = Counter(
    name="guardrail_validation_errors_total",
    documentation="Optional content validator errors and timeouts",
    labelnames=["stage", "validator"],
)

security_safe_refusals_total = Counter(
    name="security_safe_refusals_total",
    documentation="Controlled security refusals by fixed reason code",
    labelnames=["reason"],
)

tool_policy_decisions_total = Counter(
    name="tool_policy_decisions_total",
    documentation="Agent tool policy decisions",
    labelnames=["tool", "decision", "policy_version"],
)

tool_policy_denials_total = Counter(
    name="tool_policy_denials_total",
    documentation="Agent tool policy denials and approval requirements",
    labelnames=["tool", "reason"],
)

security_state_operations_total = Counter(
    name="security_state_operations_total",
    documentation="Shared security-state operations with fixed backend/result labels",
    labelnames=["backend", "operation", "result"],
)

task_registry_operations_total = Counter(
    name="task_registry_operations_total",
    documentation="Durable task-registry operations with fixed operation/result labels",
    labelnames=["operation", "result"],
)

evidence_release_workflows_total = Counter(
    name="evidence_release_workflows_total",
    documentation="Company release workflow lifecycle with bounded stage and result labels",
    labelnames=["stage", "result"],
)

# ── 业务指标 ────────────────────────────────────────────────────

qa_query_total = Counter(
    name="qa_query_total",
    documentation="Total QA queries",
    labelnames=["intent"],
)

qa_confidence_score = Histogram(
    name="qa_confidence_score",
    documentation="QA confidence score distribution",
    buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)

qa_cache_hits_total = Counter(
    name="qa_cache_hits_total",
    documentation="Total QA cache hits",
)

qa_retrieval_branch_total = Counter(
    name="qa_retrieval_branch_total",
    documentation="Native QA retrieval branch outcomes with bounded labels",
    labelnames=["branch", "status"],
)

qa_cross_encoder_results_total = Counter(
    name="qa_cross_encoder_results_total",
    documentation="Optional CrossEncoder outcomes with bounded mode/status labels",
    labelnames=["mode", "status"],
)

qa_cross_encoder_latency_seconds = Histogram(
    name="qa_cross_encoder_latency_seconds",
    documentation="Optional CrossEncoder orchestration latency including fallback",
    labelnames=["mode"],
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0),
)

qa_evidence_decisions_total = Counter(
    name="qa_evidence_decisions_total",
    documentation="Evidence qualification decisions with bounded state and route labels",
    labelnames=["state", "response_status", "policy_version"],
)

qa_structured_generation_total = Counter(
    name="qa_structured_generation_total",
    documentation="Structured QA generation outcomes with bounded status and schema labels",
    labelnames=["status", "schema_version"],
)

qa_structured_generation_latency_seconds = Histogram(
    name="qa_structured_generation_latency_seconds",
    documentation="Structured QA generation latency including one controlled parse retry",
    labelnames=["status"],
    buckets=(0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0),
)

qa_grounding_results_total = Counter(
    name="qa_grounding_results_total",
    documentation="Deterministic grounding outcomes with bounded route and policy labels",
    labelnames=["passed", "response_status", "policy_version"],
)

qa_grounding_latency_seconds = Histogram(
    name="qa_grounding_latency_seconds",
    documentation="Deterministic grounding verification latency",
    labelnames=["policy_version"],
    buckets=(0.0001, 0.0005, 0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25),
)

doc_ingest_total = Counter(
    name="doc_ingest_total",
    documentation="Total documents ingested",
    labelnames=["doc_type"],
)

doc_ingest_duration = Histogram(
    name="doc_ingest_duration_seconds",
    documentation="Document ingest duration in seconds",
    labelnames=["doc_type"],
    buckets=(1.0, 5.0, 10.0, 30.0, 60.0, 300.0),
)

# ── 存储指标（Gauge） ──────────────────────────────────────────

vector_store_size = Gauge(
    name="vector_store_size",
    documentation="Number of vectors in store",
)

knowledge_graph_nodes = Gauge(
    name="knowledge_graph_nodes",
    documentation="Number of nodes in knowledge graph",
)

knowledge_graph_edges = Gauge(
    name="knowledge_graph_edges",
    documentation="Number of edges in knowledge graph",
)

# ── LLM 指标 ───────────────────────────────────────────────────

llm_api_calls_total = Counter(
    name="llm_api_calls_total",
    documentation="Total LLM API calls",
    labelnames=["model", "status"],
)

llm_api_latency = Histogram(
    name="llm_api_latency_seconds",
    documentation="LLM API call latency in seconds",
    labelnames=["model"],
    buckets=(0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0),
)

# ── 基础设施指标 ───────────────────────────────────────────────

cdc_events_total = Counter(
    name="cdc_events_total",
    documentation="Total CDC events processed",
    labelnames=["change_type", "status"],
)

circuit_breaker_state = Gauge(
    name="circuit_breaker_state",
    documentation="Circuit breaker state (0=closed, 1=open, 2=half-open)",
    labelnames=["breaker"],
)

circuit_breaker_consecutive_failures = Gauge(
    name="circuit_breaker_consecutive_failures",
    documentation="Current consecutive counted failures for a circuit breaker",
    labelnames=["breaker"],
)

circuit_breaker_state_changed_timestamp_seconds = Gauge(
    name="circuit_breaker_state_changed_timestamp_seconds",
    documentation="Unix timestamp of the most recent circuit breaker state change",
    labelnames=["breaker"],
)

circuit_breaker_rejected_calls_total = Counter(
    name="circuit_breaker_rejected_calls",
    documentation="Total calls rejected while a circuit breaker is unavailable",
    labelnames=["breaker"],
)

circuit_breaker_fallback_calls_total = Counter(
    name="circuit_breaker_fallback_calls",
    documentation="Total fallback calls invoked after protected dependency failures",
    labelnames=["breaker"],
)

task_queue_size = Gauge(
    name="task_queue_size",
    documentation="Number of pending tasks in queue",
    labelnames=["queue"],
)

# ── 外部依赖韧性指标 ──────────────────────────────────────────
# Fixed dependency labels intentionally exclude endpoint URLs, tenant data,
# payloads, prompts, credentials, and raw exception text.

dependency_timeouts_total = Counter(
    name="dependency_timeouts",
    documentation="Total external dependency executions that timed out or exhausted their budget",
    labelnames=["dependency"],
)

dependency_retries_total = Counter(
    name="dependency_retries",
    documentation="Total retries performed for external dependency executions",
    labelnames=["dependency"],
)

dependency_final_failures_total = Counter(
    name="dependency_final_failures",
    documentation="Total external dependency executions that failed after bounded handling",
    labelnames=["dependency"],
)
