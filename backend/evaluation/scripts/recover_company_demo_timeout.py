"""Recover one verified, timed-out company-demo parser batch through lifecycle adapters."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

BACKEND_ROOT = Path(__file__).resolve().parents[2]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from domain.documents import DocumentIngestStatus
from infrastructure.celery_app import celery_app
from infrastructure.tasks.celery_ingest_tasks import _worker_lifecycle
from infrastructure.documents.catalog import PostgreSQLDocumentCatalogRepository
from infrastructure.postgres.database import get_database_service
from infrastructure.tasks.task_registry import get_task_registry
from services.documents.lifecycle import document_input_from_record


class RecoveryError(RuntimeError):
    """Fail closed when a batch is not conclusively safe to recover."""


def _manifest_document_ids(root: Path) -> set[str]:
    try:
        payload = json.loads((root / "corpus_manifest.json").read_text(encoding="utf-8"))
        documents = payload["documents"]
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as error:
        raise RecoveryError("company-demo manifest is unavailable or invalid") from error
    if not isinstance(documents, list) or len(documents) != 41:
        raise RecoveryError("company-demo manifest must contain exactly 41 documents")
    document_ids = {
        str(item.get("source_document_id") or "").strip()
        for item in documents
        if isinstance(item, dict)
    }
    if len(document_ids) != 41 or "" in document_ids:
        raise RecoveryError("company-demo manifest has invalid document identities")
    return document_ids


def _parser_is_idle(task_id: str) -> bool:
    inspector = celery_app.control.inspect()
    active = inspector.active()
    reserved = inspector.reserved()
    if active is None or reserved is None:
        raise RecoveryError("parser worker state is unavailable")
    for tasks in [*active.values(), *reserved.values()]:
        for task in tasks or []:
            if str(task.get("id") or "") == task_id:
                return False
    return True


async def _recover(
    *,
    root: Path,
    tenant_id: str,
    actor_id: str,
    task_id: str,
    upload_id: str,
    apply: bool,
) -> dict[str, Any]:
    manifest_ids = _manifest_document_ids(root)
    task_state = str(celery_app.AsyncResult(task_id).state or "")
    if task_state != "FAILURE":
        raise RecoveryError("parser task is not terminally failed")
    if not _parser_is_idle(task_id):
        raise RecoveryError("parser task is still active or reserved")

    registry = get_task_registry()
    task_record = registry.get_owned(task_id, tenant_id)
    if (
        task_record is None
        or task_record.actor_id != actor_id
        or task_record.kind != "folder_ingest"
    ):
        raise RecoveryError("parser task ownership does not match the requested recovery")

    catalog = PostgreSQLDocumentCatalogRepository(get_database_service())
    candidates = [
        record
        for record in catalog.list_documents(tenant_id=tenant_id, include_deleted=False)
        if record.doc_id in manifest_ids
        and record.external_source_id == record.doc_id
        and record.ingest_status == DocumentIngestStatus.PROCESSING.value
        and str(record.metadata.get("_upload_batch_id") or "") == upload_id
    ]
    if not candidates:
        raise RecoveryError("no matching processing company-demo records remain")
    unexpected = [
        record
        for record in catalog.list_documents(tenant_id=tenant_id, include_deleted=False)
        if str(record.metadata.get("_upload_batch_id") or "") == upload_id
        and record.ingest_status == DocumentIngestStatus.PROCESSING.value
        and record.doc_id not in manifest_ids
    ]
    if unexpected:
        raise RecoveryError("upload batch includes a non-manifest processing record")

    report = {
        "task_id": task_id,
        "task_state": task_state,
        "selected_count": len(candidates),
        "status": "validated",
    }
    if not apply:
        return report

    lifecycle = _worker_lifecycle(catalog=catalog)
    recovered_count = 0
    for record in candidates:
        recovered = await lifecycle.recover_interrupted_document(
            document_input_from_record(record),
            cleanup_partial_artifacts=True,
        )
        if not recovered:
            raise RecoveryError("a selected processing generation changed during recovery")
        recovered_count += 1
    registry.update_state(task_id, "failed")
    return {**report, "status": "recovered", "recovered_count": recovered_count}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--tenant", required=True)
    parser.add_argument("--actor-id", required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--upload-id", required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    try:
        report = asyncio.run(
            _recover(
                root=args.root.resolve(),
                tenant_id=args.tenant,
                actor_id=args.actor_id,
                task_id=args.task_id,
                upload_id=args.upload_id,
                apply=args.apply,
            )
        )
    except RecoveryError as error:
        print(json.dumps({"status": "blocked", "reason": str(error)}, ensure_ascii=False))
        return 2
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
