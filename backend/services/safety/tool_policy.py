"""Repository-owned, deny-by-default policy boundary for future agent tools."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol

from infrastructure.audit.log import (
    AuditAction,
    AuditResult,
    AuditService,
    AuditWriteError,
    get_audit_service,
)
from shared.utils.metrics import (
    tool_policy_decisions_total,
    tool_policy_denials_total,
)


class ToolDecision(str, Enum):
    """Represent tool decision."""
    ALLOW = "allow"
    DENY = "deny"
    REQUIRE_APPROVAL = "require_approval"


@dataclass(frozen=True)
class ToolRequest:
    """Represent a tool request."""
    actor_id: str
    org_id: str
    scopes: tuple[str, ...]
    tool_name: str
    arguments: dict[str, Any]
    request_id: str
    approval_reference: str | None = None


@dataclass(frozen=True)
class ToolPolicyResult:
    """Represent the result of tool policy processing."""
    decision: ToolDecision
    reason_code: str
    policy_version: str


class ToolPolicyBackend(Protocol):
    """Define the contract for tool policy backend operations."""
    def evaluate(self, request: ToolRequest) -> ToolPolicyResult:
        """Evaluate a request with the tool policy backend."""
        ...


@dataclass(frozen=True)
class _ToolRule:
    """Represent tool rule."""
    decision: ToolDecision
    required_scopes: frozenset[str] = frozenset()
    allowed_arguments: frozenset[str] = frozenset()


_POLICY_VERSION = "tool-policy-v1"
_FORBIDDEN_OVERRIDE_FIELDS = frozenset(
    {
        "actor",
        "actor_id",
        "approval",
        "approval_reference",
        "approved",
        "org_id",
        "policy_version",
        "scope",
        "scopes",
        "tenant",
        "tenant_id",
    }
)
_RULES: dict[str, _ToolRule] = {
    "vector_store.search": _ToolRule(
        ToolDecision.ALLOW,
        required_scopes=frozenset({"qa:query"}),
        allowed_arguments=frozenset(
            {"query_fingerprint", "top_k", "filters_fingerprint"}
        ),
    ),
    "knowledge_graph.search_entities": _ToolRule(
        ToolDecision.ALLOW,
        required_scopes=frozenset({"qa:query"}),
        allowed_arguments=frozenset({"query_fingerprint", "top_k"}),
    ),
    "knowledge_graph.get_neighbors": _ToolRule(
        ToolDecision.ALLOW,
        required_scopes=frozenset({"qa:query"}),
        allowed_arguments=frozenset({"entity_ids", "depth"}),
    ),
    "external.http": _ToolRule(ToolDecision.REQUIRE_APPROVAL),
    "browser.open": _ToolRule(ToolDecision.REQUIRE_APPROVAL),
    "email.send": _ToolRule(ToolDecision.REQUIRE_APPROVAL),
    "message.send": _ToolRule(ToolDecision.REQUIRE_APPROVAL),
    "webhook.test": _ToolRule(ToolDecision.REQUIRE_APPROVAL),
    "documents.delete": _ToolRule(ToolDecision.DENY),
    "identity.mutate": _ToolRule(ToolDecision.DENY),
    "database.raw_query": _ToolRule(ToolDecision.DENY),
    "knowledge_graph.raw_cypher": _ToolRule(ToolDecision.DENY),
    "shell.execute": _ToolRule(ToolDecision.DENY),
    "code.execute": _ToolRule(ToolDecision.DENY),
}
_SAFE_CODE = re.compile(r"^[a-z0-9][a-z0-9_.:-]{0,95}$")


class LocalToolPolicyEnforcer:
    """Evaluate policy only; this class never invokes a tool implementation."""

    def __init__(
        self,
        *,
        audit_service: AuditService | None = None,
        backend: ToolPolicyBackend | None = None,
        kill_switch: bool = False,
        policy_version: str = _POLICY_VERSION,
    ) -> None:
        """Initialize the local tool policy enforcer."""
        self.audit_service = audit_service or get_audit_service()
        self.backend = backend
        self.kill_switch = kill_switch
        self.policy_version = (
            policy_version
            if _SAFE_CODE.fullmatch(policy_version)
            else _POLICY_VERSION
        )

    def evaluate(self, request: ToolRequest) -> ToolPolicyResult:
        """Evaluate a request with the local tool policy enforcer."""
        result, argument_fields = self._evaluate_local(request)
        if result.decision == ToolDecision.ALLOW and self.backend is not None:
            try:
                backend_result = self.backend.evaluate(request)
                if not isinstance(backend_result, ToolPolicyResult):
                    raise TypeError("invalid backend result")
                result = backend_result
            except Exception:
                result = ToolPolicyResult(
                    ToolDecision.DENY,
                    "policy_backend_unavailable",
                    self.policy_version,
                )

        result = self._record(request, result, argument_fields)
        return result

    def _evaluate_local(
        self,
        request: ToolRequest,
    ) -> tuple[ToolPolicyResult, list[str]]:
        """Evaluate the local."""
        if self.kill_switch:
            return self._result(ToolDecision.DENY, "governance_kill_switch"), []
        if not request.actor_id or not request.org_id or not request.request_id:
            return self._result(ToolDecision.DENY, "missing_host_identity"), []
        if set(request.arguments) & _FORBIDDEN_OVERRIDE_FIELDS:
            return self._result(ToolDecision.DENY, "host_context_override"), []

        rule = _RULES.get(request.tool_name)
        if rule is None:
            return self._result(ToolDecision.DENY, "unknown_tool"), []
        if rule.decision == ToolDecision.DENY:
            return self._result(ToolDecision.DENY, "tool_denied"), []
        if rule.decision == ToolDecision.REQUIRE_APPROVAL:
            return (
                self._result(
                    ToolDecision.REQUIRE_APPROVAL,
                    "explicit_approval_required",
                ),
                [],
            )
        if not rule.required_scopes.issubset(set(request.scopes)):
            return self._result(ToolDecision.DENY, "missing_scope"), []
        unknown_arguments = set(request.arguments) - rule.allowed_arguments
        if unknown_arguments:
            return self._result(ToolDecision.DENY, "invalid_argument_schema"), []
        return (
            self._result(ToolDecision.ALLOW, "registered_read_only"),
            sorted(request.arguments),
        )

    def _record(
        self,
        request: ToolRequest,
        result: ToolPolicyResult,
        argument_fields: list[str],
    ) -> ToolPolicyResult:
        """Record the local tool policy enforcer."""
        tool_label = request.tool_name if request.tool_name in _RULES else "unknown"
        tool_policy_decisions_total.labels(
            tool=tool_label,
            decision=result.decision.value,
            policy_version=result.policy_version,
        ).inc()
        if result.decision != ToolDecision.ALLOW:
            tool_policy_denials_total.labels(
                tool=tool_label,
                reason=result.reason_code,
            ).inc()
        try:
            self.audit_service.log(
                user_id=request.actor_id,
                org_id=request.org_id,
                action=AuditAction.TOOL_POLICY_DECISION,
                resource=f"tool/{tool_label}",
                result=(
                    AuditResult.SUCCESS
                    if result.decision == ToolDecision.ALLOW
                    else AuditResult.DENIED
                ),
                metadata={
                    "request_id": request.request_id,
                    "tool_name": tool_label,
                    "decision": result.decision.value,
                    "reason_code": result.reason_code,
                    "policy_version": result.policy_version,
                    "argument_fields": argument_fields,
                },
                required=result.decision == ToolDecision.ALLOW,
            )
        except AuditWriteError:
            return self._result(ToolDecision.DENY, "audit_unavailable")
        return result

    def _result(
        self,
        decision: ToolDecision,
        reason_code: str,
    ) -> ToolPolicyResult:
        """Build a normalized tool policy evaluation result."""
        return ToolPolicyResult(decision, reason_code, self.policy_version)


__all__ = [
    "LocalToolPolicyEnforcer",
    "ToolDecision",
    "ToolPolicyBackend",
    "ToolPolicyResult",
    "ToolRequest",
]
