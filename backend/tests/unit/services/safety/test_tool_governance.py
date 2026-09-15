"""ATDD acceptance tests for deny-by-default agent tool governance."""

from __future__ import annotations

from dataclasses import dataclass

from services.safety.tool_policy import (
    LocalToolPolicyEnforcer,
    ToolDecision,
    ToolRequest,
)
from infrastructure.security.agt_policy_adapter import AGTPolicyAdapter
from infrastructure.audit.log import AuditService
from infrastructure.audit.stores import MemoryAuditStore


def _request(
    tool_name: str,
    *,
    arguments: dict | None = None,
    scopes: tuple[str, ...] = ("qa:query",),
) -> ToolRequest:
    return ToolRequest(
        actor_id="user-1",
        org_id="org-1",
        scopes=scopes,
        tool_name=tool_name,
        arguments=arguments or {},
        request_id="req-1",
    )


def _enforcer(**kwargs) -> tuple[LocalToolPolicyEnforcer, MemoryAuditStore]:
    store = MemoryAuditStore()
    service = AuditService(store)
    return LocalToolPolicyEnforcer(audit_service=service, **kwargs), store


def test_current_tenant_read_only_retrieval_can_be_allowed():
    enforcer, store = _enforcer()

    decision = enforcer.evaluate(
        _request(
            "vector_store.search",
            arguments={"query_fingerprint": "sha256:abc", "top_k": 8},
        )
    )

    assert decision.decision == ToolDecision.ALLOW
    assert decision.reason_code == "registered_read_only"
    assert decision.policy_version
    record = store.query(org_id="org-1")[0]
    assert record.resource == "tool/vector_store.search"
    assert record.metadata["argument_fields"] == ["query_fingerprint", "top_k"]


def test_missing_scope_and_identity_override_are_denied_without_values_in_audit():
    enforcer, store = _enforcer()
    secret = "other-org-secret"

    missing_scope = enforcer.evaluate(
        _request("vector_store.search", scopes=("doc:read",))
    )
    override = enforcer.evaluate(
        _request(
            "vector_store.search",
            arguments={"tenant_id": secret, "query_fingerprint": "sha256:abc"},
        )
    )

    assert missing_scope.decision == ToolDecision.DENY
    assert missing_scope.reason_code == "missing_scope"
    assert override.decision == ToolDecision.DENY
    assert override.reason_code == "host_context_override"
    assert secret not in repr(store.query(org_id="org-1"))


def test_unknown_destructive_and_external_tools_never_return_allow():
    enforcer, _ = _enforcer()

    decisions = {
        name: enforcer.evaluate(_request(name)).decision
        for name in (
            "unknown.tool",
            "documents.delete",
            "database.raw_query",
            "shell.execute",
            "external.http",
            "email.send",
        )
    }

    assert decisions["unknown.tool"] == ToolDecision.DENY
    assert decisions["documents.delete"] == ToolDecision.DENY
    assert decisions["database.raw_query"] == ToolDecision.DENY
    assert decisions["shell.execute"] == ToolDecision.DENY
    assert decisions["external.http"] == ToolDecision.REQUIRE_APPROVAL
    assert decisions["email.send"] == ToolDecision.REQUIRE_APPROVAL


def test_kill_switch_and_backend_failure_are_fail_closed():
    killed, _ = _enforcer(kill_switch=True)
    assert (
        killed.evaluate(_request("vector_store.search")).reason_code
        == "governance_kill_switch"
    )

    class _BrokenBackend:
        def evaluate(self, request: ToolRequest):
            raise RuntimeError("vendor policy leaked secret")

    failed, store = _enforcer(backend=_BrokenBackend())
    decision = failed.evaluate(_request("vector_store.search"))
    assert decision.decision == ToolDecision.DENY
    assert decision.reason_code == "policy_backend_unavailable"
    assert "vendor policy" not in repr(store.query(org_id="org-1"))


@dataclass
class _AGTDecision:
    allowed: bool
    action: str
    matched_rule: str | None = None
    reason: str = "vendor reason that must not escape"


class _Evaluator:
    def __init__(self, value) -> None:
        self.value = value
        self.contexts: list[dict] = []

    def evaluate(self, context: dict):
        self.contexts.append(context)
        if isinstance(self.value, Exception):
            raise self.value
        return self.value


def test_agt_adapter_normalizes_supported_decisions_without_argument_values():
    evaluator = _Evaluator(
        _AGTDecision(allowed=False, action="deny", matched_rule="block-dangerous")
    )
    adapter = AGTPolicyAdapter(evaluator=evaluator, policy_version="agt-poc-v1")
    request = _request(
        "vector_store.search",
        arguments={"query_fingerprint": "sha256:secret-value", "top_k": 8},
    )

    decision = adapter.evaluate(request)

    assert decision.decision == ToolDecision.DENY
    assert decision.reason_code == "agt_rule:block-dangerous"
    assert "sha256:secret-value" not in repr(evaluator.contexts)
    assert evaluator.contexts[0]["argument_fields"] == [
        "query_fingerprint",
        "top_k",
    ]


def test_agt_adapter_fails_closed_on_exception_or_unknown_shape():
    raised = AGTPolicyAdapter(
        evaluator=_Evaluator(RuntimeError("vendor secret")),
        policy_version="agt-poc-v1",
    ).evaluate(_request("vector_store.search"))
    invalid = AGTPolicyAdapter(
        evaluator=_Evaluator(object()),
        policy_version="agt-poc-v1",
    ).evaluate(_request("vector_store.search"))

    assert raised.decision == ToolDecision.DENY
    assert raised.reason_code == "agt_backend_unavailable"
    assert invalid.decision == ToolDecision.DENY
    assert invalid.reason_code == "agt_invalid_decision"
