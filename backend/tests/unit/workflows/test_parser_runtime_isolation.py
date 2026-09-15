"""ATDD for the dedicated parser queue and Kubernetes sandbox baseline."""

from __future__ import annotations

from pathlib import Path

import yaml

from infrastructure.celery_app import celery_app


ROOT = Path(__file__).resolve().parents[4]


def _documents(path: Path) -> list[dict]:
    return [
        document
        for document in yaml.safe_load_all(path.read_text(encoding="utf-8"))
        if document
    ]


def test_ingestion_tasks_route_only_to_parser_queue() -> None:
    routes = celery_app.conf.task_routes

    assert routes["infrastructure.celery_tasks.ingest_document_task"]["queue"] == "parser"
    assert routes["infrastructure.celery_tasks.batch_ingest_task"]["queue"] == "parser"


def test_parser_worker_has_required_pod_and_container_boundaries() -> None:
    documents = _documents(ROOT / "deploy" / "k8s" / "parser-worker.yaml")
    deployment = next(doc for doc in documents if doc["kind"] == "Deployment")
    pod = deployment["spec"]["template"]["spec"]
    container = pod["containers"][0]

    assert pod["automountServiceAccountToken"] is False
    assert pod["securityContext"]["seccompProfile"]["type"] == "RuntimeDefault"
    assert pod["securityContext"]["runAsNonRoot"] is True
    assert container["securityContext"]["readOnlyRootFilesystem"] is True
    assert container["securityContext"]["allowPrivilegeEscalation"] is False
    assert container["securityContext"]["capabilities"]["drop"] == ["ALL"]
    assert container["resources"]["limits"]["cpu"]
    assert container["resources"]["limits"]["memory"]
    assert container["command"][-2:] == ["-Q", "parser"]
    assert next(volume for volume in pod["volumes"] if volume["name"] == "tmp")[
        "emptyDir"
    ]["sizeLimit"]


def test_parser_network_policy_is_default_deny_with_only_internal_allows() -> None:
    documents = _documents(ROOT / "deploy" / "k8s" / "parser-network-policy.yaml")
    policies = {doc["metadata"]["name"]: doc for doc in documents}

    deny = policies["agenthub-parser-default-deny-egress"]
    allow = policies["agenthub-parser-allow-required-egress"]
    assert deny["spec"]["policyTypes"] == ["Egress"]
    assert deny["spec"]["egress"] == []

    ports = {
        port["port"]
        for rule in allow["spec"]["egress"]
        for port in rule.get("ports", [])
    }
    assert {53, 6379, 7687, 8000} <= ports
    assert all(
        "ipBlock" not in destination
        for rule in allow["spec"]["egress"]
        for destination in rule.get("to", [])
    )


def test_general_worker_explicitly_excludes_parser_queue() -> None:
    text = (ROOT / "deploy" / "k8s" / "worker.yaml").read_text(encoding="utf-8")
    assert '"-Q", "default,maintenance,update"' in text
    assert "parser" not in text.split("command:", 1)[1].split("envFrom:", 1)[0]
