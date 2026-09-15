"""Explicit, idempotent schema migration command for PostgreSQL business facts."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from typing import Sequence

from sqlalchemy import Engine, create_engine, inspect, insert, select, text

from infrastructure.postgres.models import metadata, schema_migrations
from shared.config import settings

# 当前代码要求的 PostgreSQL 业务表结构版本；本版本新增正式评测数据集的四个工作流字段。
REQUIRED_SCHEMA_VERSION = 5


def upgrade(engine: Engine) -> int:
    """按序应用幂等的数据库结构迁移，并返回目标结构版本。

    Args:
        engine: SQLAlchemy 数据库引擎，迁移在该引擎的事务连接中执行。
    """
    metadata.create_all(engine)
    with engine.begin() as connection:
        # ① users 表补充部门归属与部门管理员标记（存在性检查保证幂等）。
        user_columns = {column["name"] for column in inspect(connection).get_columns("users")}
        if "department_id" not in user_columns:
            connection.execute(text("ALTER TABLE users ADD COLUMN department_id VARCHAR(128)"))
        if "is_department_manager" not in user_columns:
            connection.execute(
                text("ALTER TABLE users ADD COLUMN is_department_manager BOOLEAN NOT NULL DEFAULT FALSE")
            )
        # ② evidence_release_workflows 表补充现役语料草稿与 Fixture 校验关联字段。
        workflow_columns = {
            column["name"]
            for column in inspect(connection).get_columns("evidence_release_workflows")
        }
        if "current_corpus_draft_id" not in workflow_columns:
            connection.execute(
                text("ALTER TABLE evidence_release_workflows ADD COLUMN current_corpus_draft_id VARCHAR(36)")
            )
        if "current_corpus_snapshot_sha256" not in workflow_columns:
            connection.execute(
                text("ALTER TABLE evidence_release_workflows ADD COLUMN current_corpus_snapshot_sha256 VARCHAR(64)")
            )
        if "fixture_validation_run_id" not in workflow_columns:
            connection.execute(
                text("ALTER TABLE evidence_release_workflows ADD COLUMN fixture_validation_run_id VARCHAR(36)")
            )
        # ③ 正式评测数据集定位字段：标识被测语料之外单独冻结的基准套件，
        #    全部可空以保持历史工作流行仍可读取。
        if "evaluation_dataset_id" not in workflow_columns:
            connection.execute(
                text("ALTER TABLE evidence_release_workflows ADD COLUMN evaluation_dataset_id VARCHAR(128)")
            )
        if "evaluation_version" not in workflow_columns:
            connection.execute(
                text("ALTER TABLE evidence_release_workflows ADD COLUMN evaluation_version VARCHAR(128)")
            )
        if "evaluation_manifest_sha256" not in workflow_columns:
            connection.execute(
                text("ALTER TABLE evidence_release_workflows ADD COLUMN evaluation_manifest_sha256 VARCHAR(64)")
            )
        if "evaluation_case_count" not in workflow_columns:
            connection.execute(
                text("ALTER TABLE evidence_release_workflows ADD COLUMN evaluation_case_count INTEGER")
            )
        # ④ 版本登记：仅当库内版本低于要求版本时写入 schema_migrations。
        current = connection.execute(
            select(schema_migrations.c.version).order_by(schema_migrations.c.version.desc()).limit(1)
        ).scalar_one_or_none()
        if int(current or 0) < REQUIRED_SCHEMA_VERSION:
            connection.execute(
                insert(schema_migrations).values(
                    version=REQUIRED_SCHEMA_VERSION,
                    applied_at=datetime.now(timezone.utc),
                )
            )
    return REQUIRED_SCHEMA_VERSION


def current_version(engine: Engine) -> int:
    """Read the current version without creating missing schema objects."""
    try:
        with engine.connect() as connection:
            value = connection.execute(
                select(schema_migrations.c.version).order_by(schema_migrations.c.version.desc()).limit(1)
            ).scalar_one_or_none()
    except Exception:
        return 0
    return int(value or 0)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the explicit migration or status command using configured DATABASE_URL."""
    parser = argparse.ArgumentParser(description="Manage KnowledgeForge PostgreSQL schema")
    parser.add_argument("command", choices=("upgrade", "status"))
    args = parser.parse_args(argv)
    if not settings.database_url.strip():
        parser.error("DATABASE_URL is required")
    engine = create_engine(settings.database_url, pool_pre_ping=True)
    try:
        version = upgrade(engine) if args.command == "upgrade" else current_version(engine)
        print(f"schema_version={version}")
        return 0 if args.command == "upgrade" or version >= REQUIRED_SCHEMA_VERSION else 1
    finally:
        engine.dispose()


if __name__ == "__main__":
    raise SystemExit(main())
