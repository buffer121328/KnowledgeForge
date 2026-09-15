"""ATDD coverage for graph, Webhook, and audit adapter boundaries."""

from __future__ import annotations

import ast
import hashlib
import hmac
import json
from pathlib import Path

from infrastructure.audit.log import AuditLog, AuditService, get_audit_service
from infrastructure.audit.stores import FileAuditStore, MemoryAuditStore
from infrastructure.graph.neo4j_graph import KnowledgeGraphService
from infrastructure.graph.neo4j_queries import build_entity_lookup_query, build_source_delete_query
from infrastructure.graph.neo4j_subgraph import empty_subgraph
from infrastructure.webhooks.delivery import build_delivery_request
from infrastructure.webhooks.stores import MemoryWebhookStore
from infrastructure.webhooks.service import Webhook, WebhookEvent, WebhookService, get_webhook_service


BACKEND_ROOT = Path(__file__).resolve().parents[3]


def _imported_modules(path: Path) -> set[str]:
    """Return every direct import target declared by one Python module."""

    tree = ast.parse(path.read_text(encoding="utf-8"))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


def test_graph_facade_and_query_helpers_preserve_tenant_parameterization():
    cypher, params = build_entity_lookup_query("Customer", tenant_id="tenant_a")
    delete_cypher, delete_params = build_source_delete_query("doc.pdf", tenant_id="tenant_a")

    assert KnowledgeGraphService.__module__ == "infrastructure.graph.neo4j_graph"
    assert "$tenant_id" in cypher
    assert params == {"name": "Customer", "tenant_id": "tenant_a"}
    assert "$tenant_id" in delete_cypher
    assert delete_params == {"source": "doc.pdf", "tenant_id": "tenant_a"}
    assert empty_subgraph("disconnected") == {"nodes": [], "edges": [], "status": "disconnected"}


def test_webhook_store_and_delivery_helpers_preserve_wire_contract():
    webhook = Webhook(
        id="wh_boundary",
        url="https://example.invalid/hook",
        events=[WebhookEvent.DOC_INGESTED],
        secret="boundary-secret",
    )
    store = MemoryWebhookStore()
    store.save(webhook)

    body, headers = build_delivery_request(webhook, WebhookEvent.DOC_INGESTED, {"doc_id": "doc_1"}, "2026-07-25T00:00:00+00:00")
    expected_signature = hmac.new(
        b"boundary-secret", body.encode("utf-8"), hashlib.sha256
    ).hexdigest()

    assert store.list_by_event(WebhookEvent.DOC_INGESTED) == [webhook]
    assert json.loads(body) == {
        "event": "doc.ingested",
        "payload": {"doc_id": "doc_1"},
        "timestamp": "2026-07-25T00:00:00+00:00",
    }
    assert headers["X-Webhook-Signature"] == f"sha256={expected_signature}"
    assert WebhookService.__module__ == "infrastructure.webhooks.service"
    assert get_webhook_service().__class__ is WebhookService


def test_audit_store_preserves_hash_chain_and_reverse_bounded_query(tmp_path):
    path = tmp_path / "audit.log"
    store = FileAuditStore(str(path))
    first = AuditLog(
        audit_id="aud_1",
        timestamp="2026-07-25T00:00:00+00:00",
        user_id="user_1",
        action="auth.login",
    )
    second = AuditLog(
        audit_id="aud_2",
        timestamp="2026-07-25T00:01:00+00:00",
        user_id="user_2",
        action="qa.query",
    )

    store.append(first)
    store.append(second)

    assert first.prev_hash == ""
    assert first.hash
    assert second.prev_hash == first.hash
    assert second.hash
    assert store.query(limit=1) == [second]
    assert store.query(user_id="user_1") == [first]
    assert isinstance(MemoryAuditStore(), MemoryAuditStore)
    assert isinstance(get_audit_service(), AuditService)


def test_audit_contract_breaks_store_to_service_reverse_dependency():
    """Audit contracts stay neutral and stores never import service policy."""

    contract_imports = _imported_modules(BACKEND_ROOT / "domain" / "audit.py")
    store_imports = _imported_modules(
        BACKEND_ROOT / "infrastructure" / "audit" / "stores.py"
    )

    assert not any(
        module == "infrastructure" or module.startswith("infrastructure.")
        for module in contract_imports
    )
    assert "infrastructure.audit.log" not in store_imports


def test_audit_facade_and_store_reexport_neutral_contract_symbols():
    """Established imports resolve directly to the neutral contract objects."""

    from domain.audit import (
        AuditIntegrityError as ContractIntegrityError,
        AuditLog as ContractAuditLog,
        AuditStore as ContractAuditStore,
        AuditStoreError as ContractStoreError,
        AuditWriteError as ContractWriteError,
    )
    from infrastructure.audit import log as audit_log, stores as audit_stores

    for module in (audit_log, audit_stores):
        assert module.AuditLog is ContractAuditLog
        assert module.AuditStore is ContractAuditStore
        assert module.AuditStoreError is ContractStoreError
        assert module.AuditWriteError is ContractWriteError
        assert module.AuditIntegrityError is ContractIntegrityError
