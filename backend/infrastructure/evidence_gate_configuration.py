"""Read the effective company-global evidence Gate with a safe last-known-good fallback."""
from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from typing import Literal

from shared.config import settings

GateMode = Literal["off", "shadow", "enforce"]


@dataclass(frozen=True, slots=True)
class EffectiveEvidenceGateConfiguration:
    mode: GateMode
    revision: int
    calibration_version: str
    source: Literal["persistent", "last_known_good", "environment_default"]

    @property
    def cache_identity(self) -> str:
        return f"gate={self.mode}:r{self.revision}:cal={self.calibration_version}"


class EvidenceGateConfigurationProvider:
    """Read durable singleton config without altering process settings or deployment state."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._last_good: EffectiveEvidenceGateConfiguration | None = None

    def get(self) -> EffectiveEvidenceGateConfiguration:
        fallback = EffectiveEvidenceGateConfiguration(
            mode=settings.qa_evidence_gate_mode,
            revision=0,
            calibration_version=settings.qa_evidence_calibration_version,
            source="environment_default",
        )
        try:
            from evaluation.release.workflow import PostgreSQLReleaseWorkflowRepository
            from infrastructure.postgres.database import get_database_service
            record = PostgreSQLReleaseWorkflowRepository(get_database_service()).gate()
            value = EffectiveEvidenceGateConfiguration(
                mode=record["mode"], revision=int(record["revision"]),
                calibration_version=str(record.get("calibration_version") or settings.qa_evidence_calibration_version),
                source="persistent" if record.get("source") == "persistent" else "environment_default",
            )
            with self._lock:
                if value.source == "persistent": self._last_good = value
            return value
        except Exception:
            with self._lock:
                if self._last_good is not None:
                    return EffectiveEvidenceGateConfiguration(
                        mode=self._last_good.mode, revision=self._last_good.revision,
                        calibration_version=self._last_good.calibration_version, source="last_known_good",
                    )
            return fallback


_provider = EvidenceGateConfigurationProvider()


def get_effective_evidence_gate_configuration() -> EffectiveEvidenceGateConfiguration:
    return _provider.get()
