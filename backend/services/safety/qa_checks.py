"""Deterministic safety boundaries for the fixed QA retrieval/generation flow."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from domain.knowledge import RetrievedContext

_GENERIC_SOURCES = {"", "knowledge_graph", "vector_store", "unknown", "none"}

_PROMPT_INJECTION_PATTERNS = (
    re.compile(r"\bignore\s+(?:all\s+)?(?:previous|prior|above)\s+(?:instructions?|rules?)", re.IGNORECASE),
    re.compile(r"\b(?:reveal|show|print|output)\s+(?:the\s+)?system\s+prompt", re.IGNORECASE),
    re.compile(r"忽略(?:此前|之前|以上|所有)?(?:的)?(?:指令|规则)"),
    re.compile(r"(?:输出|泄露|展示)(?:系统提示词?|system\s*prompt)" , re.IGNORECASE),
    re.compile(r"(?:可疑|恶意|不可信).{0,48}(?:忽略|跳过|绕过|无视).{0,24}(?:权限|授权|访问控制|安全规则)"),
    re.compile(r"\b(?:suspicious|malicious|untrusted)\s+(?:instruction|command).{0,48}\b(?:ignore|skip|bypass|disregard)\b.{0,24}\b(?:permission|authorization|access control|security rules?)\b", re.IGNORECASE),
)
_SENSITIVE_REQUEST_PATTERNS = (
    re.compile(
        r"\b(?:give|show|reveal|provide|output|print)\b.{0,60}\b(?:api[ _-]?key|access[ _-]?token|bearer\s+token|connection\s+string)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:给我|提供|展示|输出|泄露).{0,40}(?:API\s*(?:密钥|key|令牌)|访问令牌|认证令牌|连接串|数据库密码|数据库连接)",
        re.IGNORECASE,
    ),
)
_CROSS_TENANT_PATTERNS = (
    re.compile(r"\b(?:another|other)\s+tenant(?:'s)?\b", re.IGNORECASE),
    re.compile(r"(?:另一个|其他)(?:租户|组织)(?:的)?(?:文档|数据|内容)"),
)
_LINK_INDUCEMENT_PATTERNS = (
    re.compile(r"\b(?:open|click|visit|follow)\s+https?://", re.IGNORECASE),
    re.compile(r"(?:打开|点击|访问|跟随).{0,12}https?://", re.IGNORECASE),
)
_PROMPT_DISCLOSURE_OUTPUT = re.compile(
    r"^\s*(?:(?:here(?:'s| is)|the)\s+(?:system\s+prompt|system\s+instructions?)|(?:以下是(?:系统)?提示词?|系统提示词?)\s*[:：])",
    re.IGNORECASE,
)
_SECRET_OUTPUT_PATTERNS = (
    re.compile(r"\b(?:sk|rk|pk)-[A-Za-z0-9_-]{16,}\b"),
    re.compile(r"\bBearer\s+[A-Za-z0-9._-]{16,}\b", re.IGNORECASE),
    re.compile(r"\b(?:postgres(?:ql)?|redis|mysql|mongodb(?:\+srv)?)://[^\s'\"]+", re.IGNORECASE),
    re.compile(
        r"(?i)\bhttps?://(?:localhost|(?:[a-z0-9-]+\.)+internal|127(?:\.\d{1,3}){3}|10(?:\.\d{1,3}){3}|192\.168(?:\.\d{1,3}){2}|172\.(?:1[6-9]|2\d|3[01])(?:\.\d{1,3}){2})(?::\d+)?(?:/[^\s'\"]*)?"
    ),
)


class QASafetyRefusalError(RuntimeError):
    """Opaque signal for a direct request rejected by the QA safety boundary."""

    def __init__(self, code: str) -> None:
        """Initialize the question-answering safety refusal error."""
        super().__init__("QA request was rejected by the safety policy")
        self.code = code


@dataclass(frozen=True)
class ContextSafetyResult:
    """Represent the result of context safety processing."""
    contexts: list[RetrievedContext]
    actions: list[str]


@dataclass(frozen=True)
class SanitizedAnswer:
    """Represent sanitized answer."""
    answer: str
    actions: list[str]


def question_fingerprint(question: str) -> str:
    """Return a stable, payload-free identifier suitable for audit resources."""

    return f"sha256:{hashlib.sha256(question.encode('utf-8')).hexdigest()[:16]}"


def validate_question(question: str) -> None:
    """Deny only recognised direct attacks before cache/retrieval work begins."""

    if _matches_any(question, _PROMPT_INJECTION_PATTERNS):
        raise QASafetyRefusalError("prompt_injection")
    if _matches_any(question, _SENSITIVE_REQUEST_PATTERNS):
        raise QASafetyRefusalError("sensitive_data_request")
    if _matches_any(question, _CROSS_TENANT_PATTERNS):
        raise QASafetyRefusalError("cross_tenant_data_request")


def filter_contexts(contexts: list[RetrievedContext], tenant_id: str | None) -> ContextSafetyResult:
    """Return only evidence that can safely be treated as data, not instructions."""

    accepted: list[RetrievedContext] = []
    actions: list[str] = []
    for context in contexts:
        action = _context_rejection_action(context, tenant_id)
        if action is None:
            accepted.append(context)
        elif action not in actions:
            actions.append(action)
    return ContextSafetyResult(contexts=accepted, actions=actions)


def sanitize_answer(answer: str) -> SanitizedAnswer:
    """Remove high-risk disclosure formats before caching or responding."""

    if _PROMPT_DISCLOSURE_OUTPUT.search(answer):
        return SanitizedAnswer(
            answer="无法提供系统指令或内部安全配置。",
            actions=["output_system_prompt_refusal"],
        )

    sanitized = answer
    for pattern in _SECRET_OUTPUT_PATTERNS:
        sanitized = pattern.sub("[已脱敏]", sanitized)
    return SanitizedAnswer(
        answer=sanitized,
        actions=["output_redacted"] if sanitized != answer else [],
    )


def _context_rejection_action(context: RetrievedContext, tenant_id: str | None) -> str | None:
    """Return the context rejection action."""
    source = str(context.source or "").strip()
    metadata = context.metadata or {}
    if source.casefold() in _GENERIC_SOURCES:
        return "context_missing_provenance"

    context_tenant = str(metadata.get("tenant_id") or "").strip()
    if context.retrieval_type == "graph":
        mapped_source = str(metadata.get("source") or "").strip()
        if mapped_source.casefold() in _GENERIC_SOURCES:
            return "context_missing_provenance"
        if tenant_id and not context_tenant:
            return "context_missing_graph_evidence"
        if not str(metadata.get("doc_id") or "").strip():
            return "context_missing_graph_evidence"
        if not str(metadata.get("chunk_id") or "").strip():
            return "context_missing_graph_evidence"
        if not str(context.content or "").strip():
            return "context_missing_graph_evidence"

    if tenant_id and context_tenant and context_tenant != tenant_id:
        return "context_tenant_mismatch"

    if _matches_any(context.content, _PROMPT_INJECTION_PATTERNS):
        return "context_prompt_injection"
    if _matches_any(context.content, _LINK_INDUCEMENT_PATTERNS):
        return "context_link_inducement"
    return None


def _matches_any(value: str, patterns: tuple[re.Pattern[str], ...]) -> bool:
    """Report whether text matches any configured safety pattern."""
    return any(pattern.search(value) is not None for pattern in patterns)
