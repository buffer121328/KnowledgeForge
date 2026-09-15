"""Replaceable adapter for the evolving AGT public evaluator decision shape."""

from __future__ import annotations

import re
from typing import Any

from services.safety.tool_policy import (
    ToolDecision,
    ToolPolicyResult,
    ToolRequest,
)

_SAFE_RULE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_.:-]{0,63}$")


class AGTPolicyAdapter:
    """Normalize an injected AGT evaluator without importing Preview SDK types."""

    def __init__(self, *, evaluator: Any, policy_version: str) -> None:
        """Initialize the AGT policy adapter."""
        self.evaluator = evaluator
        self.policy_version = policy_version

    def evaluate(self, request: ToolRequest) -> ToolPolicyResult:
        """Evaluate a request with the AGT policy adapter."""
        context = {
            "tool_name": request.tool_name,
            "agent_id": request.actor_id,
            "org_id": request.org_id,
            "scopes": sorted(request.scopes),
            "request_id": request.request_id,
            "argument_fields": sorted(request.arguments),
        }
        try:
            vendor = self.evaluator.evaluate(context)
        except Exception:
            return self._deny("agt_backend_unavailable")

        allowed = getattr(vendor, "allowed", None)
        action = str(getattr(vendor, "action", "") or "").lower()
        if action in {"require_approval", "escalate"}:
            return ToolPolicyResult(
                ToolDecision.REQUIRE_APPROVAL,
                self._rule_code(vendor),
                self.policy_version,
            )
        if isinstance(allowed, bool) and action in {"allow", "deny", "block", ""}:
            decision = (
                ToolDecision.ALLOW
                if allowed and action not in {"deny", "block"}
                else ToolDecision.DENY
            )
            return ToolPolicyResult(
                decision,
                self._rule_code(vendor),
                self.policy_version,
            )
        return self._deny("agt_invalid_decision")

    def _rule_code(self, vendor: Any) -> str:
        """Derive a stable code from a policy rule identifier."""
        matched_rule = str(getattr(vendor, "matched_rule", "") or "")
        safe_rule = matched_rule if _SAFE_RULE.fullmatch(matched_rule) else "default"
        return f"agt_rule:{safe_rule}"

    def _deny(self, reason: str) -> ToolPolicyResult:
        """Deny the agt policy adapter."""
        return ToolPolicyResult(
            ToolDecision.DENY,
            reason,
            self.policy_version,
        )


__all__ = ["AGTPolicyAdapter"]
