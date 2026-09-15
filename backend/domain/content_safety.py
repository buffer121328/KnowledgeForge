"""Neutral content-validation contracts shared across runtime layers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

ContentDecision = Literal["pass", "deny", "sanitize"]


@dataclass(frozen=True)
class ContentValidationOutcome:
    """Represent a normalized content-validation outcome."""

    decision: ContentDecision
    value: str | None = None

    @classmethod
    def pass_(cls, value: str) -> ContentValidationOutcome:
        """Return a passing outcome that preserves the supplied value."""
        return cls(decision="pass", value=value)

    @classmethod
    def deny(cls, _unsafe_reason: str | None = None) -> ContentValidationOutcome:
        """Return a denial without exposing an unsafe reason."""
        return cls(decision="deny")

    @classmethod
    def sanitize(
        cls,
        value: str,
        _unsafe_reason: str | None = None,
    ) -> ContentValidationOutcome:
        """Return a sanitized replacement value."""
        return cls(decision="sanitize", value=value)


class ContentValidator(Protocol):
    """Define the asynchronous external content-validator contract."""

    name: str

    async def validate(
        self,
        stage: str,
        value: str,
        metadata: dict,
    ) -> ContentValidationOutcome:
        """Validate one content value at a named safety stage."""
        ...


__all__ = [
    "ContentDecision",
    "ContentValidationOutcome",
    "ContentValidator",
]
