"""Company-global Gate provider uses durable records or a safe last-known-good fallback."""
from __future__ import annotations

from infrastructure.evidence_gate_configuration import EvidenceGateConfigurationProvider


def test_provider_reads_persistent_config_and_keeps_last_known_good(monkeypatch):
    class Repository:
        def __init__(self, _database): pass
        def gate(self): return {"mode": "shadow", "revision": 7, "calibration_version": "cal-7", "source": "persistent"}
    monkeypatch.setattr("evaluation.release.workflow.PostgreSQLReleaseWorkflowRepository", Repository)
    monkeypatch.setattr("infrastructure.postgres.database.get_database_service", lambda: object())
    provider = EvidenceGateConfigurationProvider()
    durable = provider.get()
    assert durable.mode == "shadow"
    assert durable.revision == 7
    assert durable.cache_identity == "gate=shadow:r7:cal=cal-7"

    monkeypatch.setattr("infrastructure.postgres.database.get_database_service", lambda: (_ for _ in ()).throw(RuntimeError("unavailable")))
    recovered = provider.get()
    assert recovered.mode == "shadow"
    assert recovered.revision == 7
    assert recovered.source == "last_known_good"


def test_provider_uses_environment_default_when_no_durable_value(monkeypatch):
    class Repository:
        def __init__(self, _database): raise RuntimeError("unavailable")
    monkeypatch.setattr("evaluation.release.workflow.PostgreSQLReleaseWorkflowRepository", Repository)
    provider = EvidenceGateConfigurationProvider()
    value = provider.get()
    assert value.source == "environment_default"
    assert value.revision == 0
    assert value.mode in {"off", "shadow", "enforce"}
