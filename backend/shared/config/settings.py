"""
应用配置 — 通过环境变量或 .env 文件加载
"""

from ipaddress import ip_network

from pydantic import AliasChoices, Field, field_validator, model_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Deployment environment used by fail-closed security controls.
    """Define and validate settings configuration."""
    app_environment: str = "development"

    # Online model providers. MiMo is the OpenAI-compatible text default;
    # image messages continue to use Qwen VL. DEEPSEEK_* names are retained
    # as the deployed compatibility slot for the configured text provider.
    deepseek_api_key: str = Field(
        default="",
        validation_alias=AliasChoices("MIMO_API_KEY", "DEEPSEEK_API_KEY"),
    )
    deepseek_base_url: str = Field(
        default="https://api.xiaomimimo.com/v1",
        validation_alias=AliasChoices("MIMO_BASE_URL", "DEEPSEEK_BASE_URL"),
    )
    deepseek_model: str = Field(
        default="mimo-v2.5-pro",
        validation_alias=AliasChoices("MIMO_MODEL", "DEEPSEEK_MODEL"),
    )
    # 生成模型的思考模式：default 跟随 provider 默认（mimo 默认开启隐性思考，
    # 每次回答多 ~20s），disabled 关闭内部推理。检索问答遵循提示词规则即可，
    # 不需要推理链；RAGAS 裁判不受此配置影响，保持默认思考。
    llm_thinking_mode: str = Field(
        default="disabled",
        validation_alias=AliasChoices("LLM_THINKING_MODE", "QA_LLM_THINKING"),
    )
    dashscope_api_key: str = ""
    dashscope_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    embedding_model: str = "text-embedding-v4"
    embedding_batch_size: int = Field(default=10, gt=0, le=2048)
    vision_model: str = "qwen3-vl-flash"

    # Document chunking. Recursive is the stable default; semantic is gray mode.
    chunking_strategy: str = "recursive"
    chunk_size: int = Field(default=400, gt=0)
    chunk_overlap: int = Field(default=128, ge=0)
    semantic_chunk_percentile: float = Field(default=95.0, ge=0, le=100)
    semantic_chunk_min_size: int = Field(default=200, gt=0)
    semantic_chunk_max_size: int = Field(default=512, gt=0)

    # QA content safety. builtin never imports optional Guardrails AI.
    qa_safety_mode: str = "builtin"
    qa_guardrail_timeout_seconds: float = Field(default=1.0, gt=0, le=30)
    qa_guardrail_rule_version: str = "builtin-v1"
    tool_governance_kill_switch: bool = False
    tool_policy_version: str = "tool-policy-v1"

    # Ephemeral Redis coordination records; durable facts always use PostgreSQL.
    security_state_timeout_seconds: float = Field(default=1.0, gt=0, le=10)
    security_state_max_record_bytes: int = Field(
        default=65_536,
        ge=1024,
        le=1024 * 1024,
    )
    security_state_allow_local_bootstrap: bool = False
    task_registry_ttl_seconds: int = Field(
        default=8 * 24 * 3600,
        gt=7 * 24 * 3600,
        le=90 * 24 * 3600,
    )

    # PostgreSQL structured business facts. Schema changes are explicit CLI work.
    database_url: str = ""
    database_pool_size: int = Field(default=5, gt=0, le=100)
    database_max_overflow: int = Field(default=10, ge=0, le=100)
    database_connect_timeout_seconds: float = Field(default=3.0, gt=0, le=30)
    database_statement_timeout_ms: int = Field(default=5000, gt=0, le=120_000)
    database_required_schema_version: int = Field(default=5, gt=0)  # 要求的库结构版本；v5 新增评测数据集四个工作流字段
    webhook_secret_encryption_key: str = ""
    webhook_secret_key_reference: str = ""
    allow_demo_default_org_registration: bool = False
    qa_history_retention_days: int = Field(default=365, gt=0, le=3650)
    qa_cache_ttl_seconds: int = Field(default=3600, gt=0, le=7 * 24 * 3600)
    qa_cache_min_confidence: float = Field(default=0.75, ge=0, le=1)
    qa_semantic_cache_mode: str = "shadow"
    qa_semantic_candidate_limit: int = Field(default=3, gt=0, le=20)
    qa_semantic_confirmation_ttl_seconds: int = Field(default=300, gt=0, le=3600)
    qa_semantic_similarity_threshold: float = Field(default=0.9, ge=0, le=1)
    qa_retrieval_strategy: str = "dense_bm25_graph"
    qa_bm25_index_path: str = "./data/bm25"
    qa_bm25_schema_version: str = "native-bm25-v1"
    qa_bm25_tokenizer_version: str = "unicode-cjk-bigram-v1"
    qa_bm25_k1: float = Field(default=1.5, gt=0, le=10)
    qa_bm25_b: float = Field(default=0.75, ge=0, le=1)
    qa_bm25_top_k: int = Field(default=5, gt=0, le=100)
    qa_vector_top_k: int = Field(default=8, gt=0, le=50)
    qa_rrf_k: int = Field(default=60, gt=0, le=10_000)
    qa_rrf_vector_weight: float = Field(default=1.0, gt=0, le=10)
    qa_rrf_bm25_weight: float = Field(default=1.0, gt=0, le=10)
    qa_rrf_graph_weight: float = Field(default=1.05, gt=0, le=10)
    qa_context_limit: int = Field(default=10, gt=0, le=50)
    # 广义"范围/目录"类问题的目录区扩展：当排序第一的分块含目录时，
    # 顺序补拉同文档后续分块；0 表示关闭。
    qa_scope_expansion_chunks: int = Field(default=5, gt=0, le=10)

    # 仅限服务端、非交互式作用域：固定版本 Fixture 校验使用的财务部门 ID。
    # 留空即 fail-closed（校验必然失败）；该值绝不接受浏览器输入，也不落盘到运行记录。
    evaluation_fixture_finance_department_id: str = ""

    # Evidence qualification rollout. Off preserves the current fixed QA behavior.
    qa_evidence_gate_mode: str = "off"
    qa_evidence_policy_version: str = "evidence-composite-v2"
    qa_evidence_calibration_version: str = "qualification-boundaries-v2"
    qa_evidence_candidate_budget_version: str = "candidate-budget-v1"
    qa_evidence_candidate_limit: int = Field(default=8, gt=0, le=50)
    qa_evidence_background_threshold: float = Field(default=0.35, ge=0, le=1)
    qa_evidence_gray_zone_lower: float = Field(default=0.45, ge=0, le=1)
    qa_evidence_gray_zone_upper: float = Field(default=0.65, ge=0, le=1)
    qa_evidence_direct_threshold: float = Field(default=0.75, ge=0, le=1)
    qa_structured_answer_schema_version: str = "structured-answer-v1"
    qa_grounding_policy_version: str = "grounding-v1"

    # Optional CrossEncoder. Disabled mode must not require a model runtime.
    qa_cross_encoder_mode: str = "disabled"
    qa_cross_encoder_version: str = "disabled-v1"
    qa_cross_encoder_model: str = Field(default="qwen3-rerank", min_length=1, max_length=128)
    qa_cross_encoder_candidate_limit: int = Field(default=10, gt=0, le=50)
    qa_cross_encoder_context_limit: int = Field(default=8, gt=0, le=50)
    qa_rerank_max_per_document: int = Field(default=5, gt=0, le=50)
    qa_cross_encoder_timeout_seconds: float = Field(default=2.0, gt=0, le=30)
    qa_cross_encoder_remote_endpoint: str = ""

    @field_validator("chunking_strategy")
    @classmethod
    def _validate_chunking_strategy(cls, value: str) -> str:
        """Normalize and validate the configured document chunking strategy."""
        normalized = value.strip().lower()
        if normalized not in {"recursive", "semantic"}:
            raise ValueError("chunking_strategy must be recursive or semantic")
        return normalized


    @field_validator("qa_evidence_gate_mode")
    @classmethod
    def _validate_qa_evidence_gate_mode(cls, value: str) -> str:
        """Validate the staged evidence-gate rollout mode."""
        normalized = value.strip().lower()
        if normalized not in {"off", "shadow", "enforce"}:
            raise ValueError("qa_evidence_gate_mode must be off, shadow, or enforce")
        return normalized

    @field_validator("llm_thinking_mode")
    @classmethod
    def _validate_llm_thinking_mode(cls, value: str) -> str:
        """Normalize and validate the generation thinking mode."""
        normalized = value.strip().lower()
        if normalized not in {"default", "disabled"}:
            raise ValueError("llm_thinking_mode must be default or disabled")
        return normalized

    @field_validator("qa_cross_encoder_mode")
    @classmethod
    def _validate_qa_cross_encoder_mode(cls, value: str) -> str:
        """Validate the optional CrossEncoder execution mode."""
        normalized = value.strip().lower()
        if normalized not in {"disabled", "remote", "local"}:
            raise ValueError("qa_cross_encoder_mode must be disabled, remote, or local")
        return normalized

    @field_validator("qa_retrieval_strategy")
    @classmethod
    def _validate_qa_retrieval_strategy(cls, value: str) -> str:
        """Validate the server-selected native retrieval strategy."""
        normalized = value.strip().lower()
        allowed = {"dense", "dense_graph", "dense_bm25", "dense_bm25_graph"}
        if normalized not in allowed:
            raise ValueError(
                "qa_retrieval_strategy must be dense, dense_graph, dense_bm25, or dense_bm25_graph"
            )
        return normalized

    @property
    def qa_strategy_uses_bm25(self) -> bool:
        """Return whether the configured response strategy requires sparse retrieval."""
        return self.qa_retrieval_strategy in {"dense_bm25", "dense_bm25_graph"}

    @field_validator("qa_safety_mode")
    @classmethod
    def _validate_qa_safety_mode(cls, value: str) -> str:
        """Validate the question-answering safety mode."""
        normalized = value.strip().lower()
        if normalized not in {"builtin", "monitor", "enforce"}:
            raise ValueError("qa_safety_mode must be builtin, monitor, or enforce")
        return normalized

    @field_validator(
        "qa_guardrail_rule_version",
        "tool_policy_version",
        "qa_evidence_policy_version",
        "qa_evidence_calibration_version",
        "qa_evidence_candidate_budget_version",
        "qa_structured_answer_schema_version",
        "qa_grounding_policy_version",
        "qa_cross_encoder_version",
    )
    @classmethod
    def _validate_policy_version(cls, value: str) -> str:
        """Validate the policy version."""
        import re

        if not re.fullmatch(r"[a-z0-9][a-z0-9_.-]{0,63}", value):
            raise ValueError("policy versions must be safe low-cardinality labels")
        return value

    @field_validator("qa_semantic_cache_mode")
    @classmethod
    def _validate_semantic_cache_mode(cls, value: str) -> str:
        """Validate the staged semantic-cache rollout mode."""
        normalized = value.strip().lower()
        if normalized not in {"disabled", "shadow", "confirm"}:
            raise ValueError("qa_semantic_cache_mode must be disabled, shadow, or confirm")
        return normalized

    def validate_security_state_for_runtime(self, environment: str) -> None:
        """Reject local identity bootstrap outside explicit local use."""
        from infrastructure.security.security_state import SecurityStateConfigurationError

        normalized = (environment or "").strip().lower()
        local = normalized in {"development", "test"}
        if not local and self.security_state_allow_local_bootstrap:
            raise SecurityStateConfigurationError(
                "local identity bootstrap is forbidden"
            )

    def validate_database_for_runtime(self, environment: str) -> None:
        """Require the sole PostgreSQL fact source and Webhook protection."""
        from infrastructure.postgres.database import DatabaseConfigurationError

        normalized = (environment or "").strip().lower()
        if not self.database_url.strip():
            raise DatabaseConfigurationError("PostgreSQL facts require DATABASE_URL")
        if normalized not in {"development", "test"} and self.allow_demo_default_org_registration:
            raise DatabaseConfigurationError("demo default organization registration is local only")
        if not (
            self.webhook_secret_encryption_key.strip()
            and self.webhook_secret_key_reference.strip()
        ):
            raise DatabaseConfigurationError(
                "PostgreSQL webhooks require approved secret protection"
            )

    # Neo4j
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "password"

    # Vector Store
    vector_store_type: str = "chroma"  # chroma | pgvector
    chroma_host: str = "localhost"
    chroma_port: int = 8000
    pgvector_dsn: str = "postgresql://postgres:postgres@localhost:5432/knowledge"

    # External dependency execution budgets. Connection/read values are passed
    # to clients that expose supported transport hooks; total budgets are
    # enforced by the shared async execution policy for every dependency.
    vector_connect_timeout_seconds: float = Field(default=2.0, gt=0)
    vector_read_timeout_seconds: float = Field(default=5.0, gt=0)
    vector_total_timeout_seconds: float = Field(default=8.0, gt=0)
    vector_max_retries: int = Field(default=1, ge=0)

    graph_connect_timeout_seconds: float = Field(default=2.0, gt=0)
    graph_read_timeout_seconds: float = Field(default=5.0, gt=0)
    graph_total_timeout_seconds: float = Field(default=8.0, gt=0)
    graph_max_retries: int = Field(default=1, ge=0)

    llm_connect_timeout_seconds: float = Field(default=10.0, gt=0)
    llm_read_timeout_seconds: float = Field(default=60.0, gt=0)
    llm_total_timeout_seconds: float = Field(default=90.0, gt=0)
    llm_max_retries: int = Field(default=1, ge=0)

    webhook_connect_timeout_seconds: float = Field(default=2.0, gt=0)
    webhook_read_timeout_seconds: float = Field(default=5.0, gt=0)
    webhook_total_timeout_seconds: float = Field(default=15.0, gt=0)
    webhook_max_retries: int = Field(default=3, ge=0)
    # Comma-separated exact hostnames. Empty retains public HTTPS compatibility.
    webhook_allowed_hosts: str = ""
    webhook_allow_http_in_local_development: bool = False
    webhook_response_body_limit_bytes: int = Field(default=65536, gt=0, le=1048576)

    @property
    def webhook_allowed_host_set(self) -> frozenset[str]:
        """Handle webhook allowed host set for the settings."""
        return frozenset(host.strip() for host in self.webhook_allowed_hosts.split(",") if host.strip())

    # Kafka (CDC)
    kafka_bootstrap_servers: str = "localhost:9092"
    kafka_topic_doc_changes: str = "doc-changes"
    kafka_topic_kg_updates: str = "kg-updates"

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8080

    # Document Store
    upload_dir: str = "./uploads"
    document_catalog_path: str = ""
    upload_max_file_bytes: int = Field(default=50 * 1024 * 1024, gt=0)
    upload_max_batch_files: int = Field(default=50, gt=0, le=50)
    upload_tenant_quota_bytes: int = Field(default=5 * 1024 * 1024 * 1024, gt=0)
    upload_stream_chunk_bytes: int = Field(default=1024 * 1024, ge=4096, le=8 * 1024 * 1024)
    upload_max_archive_entries: int = Field(default=2000, gt=0, le=10000)
    upload_max_archive_uncompressed_bytes: int = Field(default=512 * 1024 * 1024, gt=0)
    upload_max_archive_compression_ratio: float = Field(default=100.0, gt=1, le=1000)
    upload_max_pdf_pages: int = Field(default=500, gt=0, le=10000)
    upload_max_image_pixels: int = Field(default=40_000_000, gt=0)
    upload_max_spreadsheet_rows: int = Field(default=100_000, gt=0)
    upload_max_text_characters: int = Field(default=5_000_000, gt=0)
    upload_quarantine_retention_hours: int = Field(default=0, ge=0, le=24 * 365)
    upload_staging_retention_hours: int = Field(default=24, gt=0, le=24 * 30)
    upload_require_external_scanner: bool = False

    # Persistent security audit store. Retention days is an operations policy
    # input; cleanup is performed outside request paths.
    audit_log_file: str = "./logs/audit.log"
    audit_log_max_bytes: int = Field(default=10 * 1024 * 1024, gt=0)
    audit_log_backup_count: int = Field(default=7, ge=0, le=365)
    audit_log_retention_days: int = Field(default=90, gt=0, le=3650)

    # Redis
    redis_url: str = "redis://localhost:6379/0"
    readiness_probe_timeout_seconds: float = Field(default=2.0, gt=0, le=10)

    # Composite sensitive-route rate limits. Production must point the limiter
    # at a shared backend such as Redis; memory:// is local development only.
    rate_limit_storage_uri: str = "memory://"
    rate_limit_trusted_proxy_cidrs: str = ""
    rate_limit_auth_public: str = "10/minute"
    rate_limit_qa_ask: str = "100/minute"
    rate_limit_doc_upload: str = "10/minute"
    rate_limit_doc_batch_upload: str = "5/minute"
    rate_limit_password: str = "5/minute"
    rate_limit_api_key_mutation: str = "20/minute"
    rate_limit_webhook_test: str = "10/minute"

    @field_validator("rate_limit_trusted_proxy_cidrs")
    @classmethod
    def _validate_trusted_proxy_cidrs(cls, value: str) -> str:
        """Validate the trusted proxy cidrs."""
        for item in value.split(","):
            candidate = item.strip()
            if candidate:
                ip_network(candidate, strict=False)
        return value

    @model_validator(mode="after")
    def _validate_llm_timeout_budget(self):
        """Ensure the shared deadline can contain one connect-and-read attempt."""
        minimum_attempt_budget = (
            self.llm_connect_timeout_seconds + self.llm_read_timeout_seconds
        )
        if self.llm_total_timeout_seconds < minimum_attempt_budget:
            raise ValueError(
                "llm_total_timeout_seconds must be >= "
                "llm_connect_timeout_seconds + llm_read_timeout_seconds"
            )
        return self

    @model_validator(mode="after")
    def _validate_qa_evidence_bounds(self):
        """Reject inconsistent evidence thresholds and candidate budgets."""
        thresholds = (
            self.qa_evidence_background_threshold,
            self.qa_evidence_gray_zone_lower,
            self.qa_evidence_gray_zone_upper,
            self.qa_evidence_direct_threshold,
        )
        if thresholds != tuple(sorted(thresholds)):
            raise ValueError(
                "evidence thresholds must satisfy background <= gray lower <= gray upper <= direct"
            )
        if self.qa_evidence_candidate_limit > self.qa_context_limit:
            raise ValueError("qa_evidence_candidate_limit must be <= qa_context_limit")
        if self.qa_cross_encoder_candidate_limit > self.qa_context_limit:
            raise ValueError("qa_cross_encoder_candidate_limit must be <= qa_context_limit")
        if self.qa_cross_encoder_context_limit > self.qa_cross_encoder_candidate_limit:
            raise ValueError(
                "qa_cross_encoder_context_limit must be <= qa_cross_encoder_candidate_limit"
            )
        if self.qa_rerank_max_per_document > self.qa_cross_encoder_candidate_limit:
            raise ValueError(
                "qa_rerank_max_per_document must be <= qa_cross_encoder_candidate_limit"
            )
        if (
            self.qa_cross_encoder_mode == "remote"
            and not self.qa_cross_encoder_remote_endpoint.strip()
        ):
            raise ValueError(
                "qa_cross_encoder_remote_endpoint is required in remote mode"
            )
        if self.qa_cross_encoder_mode == "remote" and not self.dashscope_api_key.strip():
            raise ValueError("dashscope_api_key is required in remote mode")
        return self

    @model_validator(mode="after")
    def _validate_chunking_bounds(self):
        """Reject overlapping or semantic chunk bounds that cannot be satisfied."""
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        if self.semantic_chunk_min_size > self.semantic_chunk_max_size:
            raise ValueError(
                "semantic_chunk_min_size must be <= semantic_chunk_max_size"
            )
        return self

    @model_validator(mode="after")
    def _validate_upload_bounds(self):
        """Validate the upload bounds."""
        if self.upload_tenant_quota_bytes < self.upload_max_file_bytes:
            raise ValueError("upload_tenant_quota_bytes must be >= upload_max_file_bytes")
        return self

    @property
    def rate_limit_trusted_proxy_cidr_tuple(self) -> tuple[str, ...]:
        """Handle rate limit trusted proxy cidr tuple for the settings."""
        return tuple(
            item.strip()
            for item in self.rate_limit_trusted_proxy_cidrs.split(",")
            if item.strip()
        )

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
        "populate_by_name": True,
    }


settings = Settings()
