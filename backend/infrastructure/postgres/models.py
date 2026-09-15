"""SQLAlchemy table metadata for durable, tenant-scoped business facts."""

from __future__ import annotations

from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    CheckConstraint,
    Column,
    DateTime,
    Float,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    Integer,
    LargeBinary,
    MetaData,
    String,
    Table,
    Text,
    UniqueConstraint,
)

metadata = MetaData()

schema_migrations = Table(
    "schema_migrations",
    metadata,
    Column("version", Integer, primary_key=True),
    Column("applied_at", DateTime(timezone=True), nullable=False),
)

organizations = Table(
    "organizations",
    metadata,
    Column("id", String(128), primary_key=True),
    Column("name", String(255), nullable=False),
    Column("status", String(32), nullable=False, server_default="active"),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False),
)

users = Table(
    "users",
    metadata,
    Column("id", String(128), primary_key=True),
    Column("tenant_id", String(128), ForeignKey("organizations.id"), nullable=False),
    Column("username", String(255), nullable=False, unique=True),
    Column("email", String(320), unique=True),
    Column("display_name", String(255), nullable=False, server_default=""),
    Column("password_hash", Text, nullable=False),
    Column("role", String(64), nullable=False),
    Column("department_id", String(128)),
    Column("is_department_manager", Boolean, nullable=False, server_default="0"),
    Column("is_active", Boolean, nullable=False, server_default="1"),
    Column("token_version", Integer, nullable=False, server_default="0"),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False),
    Column("last_login_at", DateTime(timezone=True)),
    Column("deleted_at", DateTime(timezone=True)),
    UniqueConstraint("tenant_id", "id", name="uq_users_tenant_id"),
    UniqueConstraint("tenant_id", "username", name="uq_users_tenant_username"),
    UniqueConstraint("tenant_id", "email", name="uq_users_tenant_email"),
    CheckConstraint("token_version >= 0", name="ck_users_token_version"),
)
Index("ix_users_username", users.c.username)
Index("ix_users_email", users.c.email)

roles = Table(
    "roles",
    metadata,
    Column("id", String(128), primary_key=True),
    Column("tenant_id", String(128), ForeignKey("organizations.id"), nullable=False),
    Column("name", String(64), nullable=False),
    Column("description", Text, nullable=False, server_default=""),
    Column("is_builtin", Boolean, nullable=False, server_default="0"),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False),
    UniqueConstraint("tenant_id", "id", name="uq_roles_tenant_id"),
    UniqueConstraint("tenant_id", "name", name="uq_roles_tenant_name"),
)

permissions = Table(
    "permissions",
    metadata,
    Column("code", String(128), primary_key=True),
    Column("description", Text, nullable=False, server_default=""),
)

role_permissions = Table(
    "role_permissions",
    metadata,
    Column("tenant_id", String(128), nullable=False),
    Column("role_id", String(128), nullable=False),
    Column("permission_code", String(128), ForeignKey("permissions.code"), nullable=False),
    ForeignKeyConstraint(
        ["tenant_id", "role_id"],
        ["roles.tenant_id", "roles.id"],
        ondelete="CASCADE",
    ),
    UniqueConstraint("tenant_id", "role_id", "permission_code", name="pk_role_permissions"),
)

user_roles = Table(
    "user_roles",
    metadata,
    Column("tenant_id", String(128), nullable=False),
    Column("user_id", String(128), nullable=False),
    Column("role_id", String(128), nullable=False),
    ForeignKeyConstraint(["tenant_id", "user_id"], ["users.tenant_id", "users.id"], ondelete="CASCADE"),
    ForeignKeyConstraint(["tenant_id", "role_id"], ["roles.tenant_id", "roles.id"]),
    UniqueConstraint("tenant_id", "user_id", name="pk_user_roles"),
)

api_keys = Table(
    "api_keys",
    metadata,
    Column("id", String(128), primary_key=True),
    Column("tenant_id", String(128), nullable=False),
    Column("user_id", String(128), nullable=False),
    Column("key_prefix", String(32), nullable=False),
    Column("key_hash", String(128), nullable=False, unique=True),
    Column("name", String(255), nullable=False),
    Column("role", String(64), nullable=False),
    Column("scopes", JSON, nullable=False),
    Column("expires_at", DateTime(timezone=True), nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("last_used_at", DateTime(timezone=True)),
    Column("is_active", Boolean, nullable=False, server_default="1"),
    Column("disabled_at", DateTime(timezone=True)),
    Column("rotated_at", DateTime(timezone=True)),
    Column("rotation_count", Integer, nullable=False, server_default="0"),
    Column("revoked_at", DateTime(timezone=True)),
    Column("revoked_by", String(128)),
    ForeignKeyConstraint(["tenant_id", "user_id"], ["users.tenant_id", "users.id"], ondelete="CASCADE"),
    UniqueConstraint("tenant_id", "id", name="uq_api_keys_tenant_id"),
)
Index("ix_api_keys_owner", api_keys.c.tenant_id, api_keys.c.user_id)

registration_invitations = Table(
    "registration_invitations",
    metadata,
    Column("id", String(128), primary_key=True),
    Column("tenant_id", String(128), ForeignKey("organizations.id"), nullable=False),
    Column("token_hash", String(128), nullable=False, unique=True),
    Column("role", String(64), nullable=False),
    Column("email", String(320)),
    Column("expires_at", DateTime(timezone=True), nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("consumed_at", DateTime(timezone=True)),
    Column("consumed_by", String(128)),
)

qa_conversations = Table(
    "qa_conversations",
    metadata,
    Column("id", String(128), primary_key=True),
    Column("tenant_id", String(128), ForeignKey("organizations.id"), nullable=False),
    Column("user_id", String(128), nullable=False),
    Column("title", String(255), nullable=False, server_default=""),
    Column("status", String(32), nullable=False, server_default="active"),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False),
    Column("deleted_at", DateTime(timezone=True)),
    ForeignKeyConstraint(["tenant_id", "user_id"], ["users.tenant_id", "users.id"]),
    UniqueConstraint("tenant_id", "id", name="uq_qa_conversations_tenant_id"),
)
Index("ix_qa_conversations_owner", qa_conversations.c.tenant_id, qa_conversations.c.user_id, qa_conversations.c.updated_at)

qa_messages = Table(
    "qa_messages",
    metadata,
    Column("id", String(128), primary_key=True),
    Column("tenant_id", String(128), nullable=False),
    Column("conversation_id", String(128), nullable=False),
    Column("sequence", Integer, nullable=False),
    Column("role", String(16), nullable=False),
    Column("content", Text, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
    ForeignKeyConstraint(
        ["tenant_id", "conversation_id"],
        ["qa_conversations.tenant_id", "qa_conversations.id"],
        ondelete="CASCADE",
    ),
    UniqueConstraint("tenant_id", "conversation_id", "sequence", name="uq_qa_message_sequence"),
)

qa_runs = Table(
    "qa_runs",
    metadata,
    Column("id", String(128), primary_key=True),
    Column("tenant_id", String(128), nullable=False),
    Column("conversation_id", String(128), nullable=False),
    Column("user_id", String(128), nullable=False),
    Column("retrieval_mode", String(32), nullable=False),
    Column("confidence", Float, nullable=False, server_default="0"),
    Column("intent", String(64), nullable=False, server_default=""),
    Column("knowledge_revision", BigInteger, nullable=False, server_default="0"),
    Column("model_version", String(128), nullable=False, server_default=""),
    Column("rule_version", String(128), nullable=False, server_default=""),
    Column("degradation_code", String(128)),
    Column("cache_hit_type", String(32), nullable=False, server_default="none"),
    Column("duration_ms", Integer, nullable=False, server_default="0"),
    Column("semantic_decision", String(32)),
    Column("created_at", DateTime(timezone=True), nullable=False),
    ForeignKeyConstraint(
        ["tenant_id", "conversation_id"],
        ["qa_conversations.tenant_id", "qa_conversations.id"],
        ondelete="CASCADE",
    ),
    UniqueConstraint("tenant_id", "id", name="uq_qa_runs_tenant_id"),
)

qa_sources = Table(
    "qa_sources",
    metadata,
    Column("id", String(128), primary_key=True),
    Column("tenant_id", String(128), nullable=False),
    Column("qa_run_id", String(128), nullable=False),
    Column("position", Integer, nullable=False),
    Column("doc_id", String(128), nullable=False),
    Column("chunk_id", String(255), nullable=False, server_default=""),
    Column("retrieval_type", String(32), nullable=False),
    Column("score", Float, nullable=False),
    Column("display_summary", Text, nullable=False),
    ForeignKeyConstraint(["tenant_id", "qa_run_id"], ["qa_runs.tenant_id", "qa_runs.id"], ondelete="CASCADE"),
    UniqueConstraint("tenant_id", "qa_run_id", "position", name="uq_qa_source_position"),
)

qa_feedback = Table(
    "qa_feedback",
    metadata,
    Column("id", String(128), primary_key=True),
    Column("tenant_id", String(128), nullable=False),
    Column("qa_run_id", String(128), nullable=False),
    Column("user_id", String(128), nullable=False),
    Column("rating", String(32), nullable=False),
    Column("note", String(1000), nullable=False, server_default=""),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False),
    ForeignKeyConstraint(["tenant_id", "qa_run_id"], ["qa_runs.tenant_id", "qa_runs.id"], ondelete="CASCADE"),
    UniqueConstraint("tenant_id", "qa_run_id", "user_id", name="uq_qa_feedback_owner"),
)

departments = Table(
    "departments",
    metadata,
    Column("tenant_id", String(128), ForeignKey("organizations.id"), primary_key=True),
    Column("department_id", String(128), primary_key=True),
    Column("company_id", String(128), nullable=False),
    Column("name", String(255), nullable=False),
    Column("normalized_key", String(255), nullable=False),
    Column("status", String(32), nullable=False, server_default="active"),
    UniqueConstraint("tenant_id", "company_id", "normalized_key", name="uq_departments_normalized_key"),
)

documents = Table(
    "documents",
    metadata,
    Column("tenant_id", String(128), primary_key=True),
    Column("doc_id", String(128), primary_key=True),
    Column("company_id", String(128), nullable=False),
    Column("department_id", String(128), nullable=False),
    Column("folder_path", Text, nullable=False),
    Column("relative_path", Text, nullable=False),
    Column("normalized_relative_path", Text, nullable=False),
    Column("uploaded_filename", String(512), nullable=False),
    Column("display_name", String(512), nullable=False),
    Column("provenance_source_filename", String(512), nullable=False),
    Column("storage_reference", Text, nullable=False),
    Column("content_sha256", String(64), nullable=False),
    Column("size_bytes", BigInteger, nullable=False),
    Column("mime_type", String(255), nullable=False),
    Column("document_type", String(64), nullable=False),
    Column("source_format", String(64), nullable=False),
    Column("ingest_status", String(64), nullable=False),
    Column("version", Integer, nullable=False),
    Column("authority", String(255), nullable=False, server_default=""),
    Column("review_status", String(64), nullable=False, server_default=""),
    Column("sensitivity", String(64), nullable=False, server_default=""),
    Column("external_source_id", String(255), nullable=False, server_default=""),
    Column("created_by", String(128), nullable=False, server_default=""),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False),
    Column("chunks_count", Integer, nullable=False, server_default="0"),
    Column("entities_count", Integer, nullable=False, server_default="0"),
    Column("relations_count", Integer, nullable=False, server_default="0"),
    Column("error_code", String(128), nullable=False, server_default=""),
    Column("legacy", Boolean, nullable=False, server_default="0"),
    Column("metadata_json", JSON, nullable=False),
    ForeignKeyConstraint(["tenant_id", "department_id"], ["departments.tenant_id", "departments.department_id"]),
    UniqueConstraint("tenant_id", "department_id", "normalized_relative_path", "version", name="uq_documents_path_version"),
    CheckConstraint("size_bytes >= 0", name="ck_documents_size"),
    CheckConstraint("version > 0", name="ck_documents_version"),
)
Index("ix_documents_tenant_company", documents.c.tenant_id, documents.c.company_id, documents.c.ingest_status)
Index("ix_documents_tenant_department", documents.c.tenant_id, documents.c.department_id, documents.c.ingest_status)

webhooks = Table(
    "webhooks",
    metadata,
    Column("id", String(128), primary_key=True),
    Column("tenant_id", String(128), ForeignKey("organizations.id"), nullable=False),
    Column("url", Text, nullable=False),
    Column("events", JSON, nullable=False),
    Column("secret_ciphertext", LargeBinary),
    Column("secret_key_ref", String(255)),
    Column("is_active", Boolean, nullable=False, server_default="1"),
    Column("failure_count", Integer, nullable=False, server_default="0"),
    Column("max_failures", Integer, nullable=False, server_default="5"),
    Column("last_triggered_at", DateTime(timezone=True)),
    Column("last_response_category", String(64)),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("metadata_json", JSON, nullable=False),
    UniqueConstraint("tenant_id", "id", name="uq_webhooks_tenant_id"),
)

webhook_delivery_attempts = Table(
    "webhook_delivery_attempts",
    metadata,
    Column("id", String(128), primary_key=True),
    Column("tenant_id", String(128), nullable=False),
    Column("webhook_id", String(128), nullable=False),
    Column("event", String(128), nullable=False),
    Column("attempt", Integer, nullable=False),
    Column("status_category", String(64), nullable=False),
    Column("final_status", String(64), nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
    ForeignKeyConstraint(["tenant_id", "webhook_id"], ["webhooks.tenant_id", "webhooks.id"], ondelete="CASCADE"),
)

task_runs = Table(
    "task_runs",
    metadata,
    Column("id", String(128), primary_key=True),
    Column("tenant_id", String(128), ForeignKey("organizations.id"), nullable=False),
    Column("user_id", String(128), nullable=False),
    Column("task_type", String(128), nullable=False),
    Column("status", String(64), nullable=False),
    Column("safe_file_reference", Text, nullable=False, server_default=""),
    Column("failure_category", String(64)),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False),
    Column("started_at", DateTime(timezone=True)),
    Column("finished_at", DateTime(timezone=True)),
    UniqueConstraint("tenant_id", "id", name="uq_task_runs_tenant_id"),
)
Index("ix_task_runs_owner", task_runs.c.tenant_id, task_runs.c.user_id, task_runs.c.created_at)

# Company-global release workflow facts.  ``company_namespace`` is an internal
# audit compatibility namespace derived from authenticated state, never client input.
evidence_release_workflows = Table(
    "evidence_release_workflows", metadata,
    Column("id", String(128), primary_key=True),
    Column("company_namespace", String(128), nullable=False),
    Column("dataset_id", String(128), nullable=False),
    Column("version", String(128), nullable=False),
    Column("manifest_sha256", String(64), nullable=False),
    Column("current_corpus_draft_id", String(36)),
    Column("current_corpus_snapshot_sha256", String(64)),
    Column("fixture_validation_run_id", String(36)),
    # current_corpus_* 字段标识被测的现役语料；以下可选字段标识单独冻结的正式基准套件。
    # 全部允许为 NULL，保证历史工作流行仍可正常读取。
    Column("evaluation_dataset_id", String(128)),  # 正式评测数据集 ID
    Column("evaluation_version", String(128)),  # 正式评测数据集版本号
    Column("evaluation_manifest_sha256", String(64)),  # 评测清单 manifest 的 SHA-256 校验值
    Column("evaluation_case_count", Integer),  # 评测用例数量
    Column("initiated_by", String(128), nullable=False),
    Column("status", String(64), nullable=False),
    Column("stage", String(64), nullable=False),
    Column("revision", Integer, nullable=False),
    Column("target_mode", String(16)),
    Column("calibration_version", String(128)),
    Column("reason_code", String(96)),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False),
    UniqueConstraint("company_namespace", "id", name="uq_evidence_release_workflows_company_id"),
)
Index("ix_evidence_release_workflows_active", evidence_release_workflows.c.company_namespace, evidence_release_workflows.c.dataset_id, evidence_release_workflows.c.version, evidence_release_workflows.c.status)

evidence_release_attempts = Table(
    "evidence_release_attempts", metadata,
    Column("id", String(128), primary_key=True),
    Column("workflow_id", String(128), ForeignKey("evidence_release_workflows.id", ondelete="CASCADE"), nullable=False),
    Column("stage", String(64), nullable=False),
    Column("attempt_number", Integer, nullable=False),
    Column("status", String(64), nullable=False),
    Column("reason_code", String(96)),
    Column("metrics", JSON, nullable=False),
    Column("started_at", DateTime(timezone=True), nullable=False),
    Column("finished_at", DateTime(timezone=True)),
    UniqueConstraint("workflow_id", "stage", "attempt_number", name="uq_evidence_release_attempts_stage"),
)

evidence_release_approvals = Table(
    "evidence_release_approvals", metadata,
    Column("id", String(128), primary_key=True),
    Column("workflow_id", String(128), ForeignKey("evidence_release_workflows.id", ondelete="CASCADE"), nullable=False),
    Column("reviewer_id", String(128), nullable=False),
    Column("decision", String(16), nullable=False),
    Column("reason", String(1000), nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
    UniqueConstraint("workflow_id", "reviewer_id", name="uq_evidence_release_approvals_reviewer"),
)
Index("ix_evidence_release_approvals_workflow", evidence_release_approvals.c.workflow_id, evidence_release_approvals.c.created_at)

evidence_gate_configuration = Table(
    "evidence_gate_configuration", metadata,
    Column("singleton", String(16), primary_key=True),
    Column("mode", String(16), nullable=False),
    Column("revision", Integer, nullable=False),
    Column("calibration_version", String(128)),
    Column("updated_by", String(128), nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False),
)

evidence_gate_configuration_history = Table(
    "evidence_gate_configuration_history", metadata,
    Column("id", String(128), primary_key=True),
    Column("action", String(32), nullable=False),
    Column("previous_mode", String(16), nullable=False),
    Column("mode", String(16), nullable=False),
    Column("configuration_revision", Integer, nullable=False),
    Column("calibration_version", String(128)),
    Column("workflow_id", String(128)),
    Column("actor_id", String(128), nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
)

__all__ = [
    "api_keys",
    "departments",
    "documents",
    "metadata",
    "organizations",
    "permissions",
    "qa_conversations",
    "qa_feedback",
    "qa_messages",
    "qa_runs",
    "qa_sources",
    "registration_invitations",
    "role_permissions",
    "roles",
    "schema_migrations",
    "task_runs",
    "user_roles",
    "users",
    "webhook_delivery_attempts",
    "webhooks",
]
