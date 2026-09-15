"""Static, fail-closed checks for parser deployment manifests and auth config."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import yaml


_PARSER_LABEL = {"app": "agenthub-parser"}
_ALLOWED_PORTS = {
    ("dns", "UDP", 53),
    ("dns", "TCP", 53),
    ("redis-agenthub", "TCP", 6379),
    ("neo4j-agenthub", "TCP", 7687),
    ("chroma-agenthub", "TCP", 8000),
}


def _documents(source: str | Path) -> list[dict[str, Any]]:
    """Return the documents."""
    text = Path(source).read_text(encoding="utf-8") if isinstance(source, Path) else source
    return [document for document in yaml.safe_load_all(text) if isinstance(document, dict)]


def _labels_match(left: dict[str, Any], right: dict[str, Any]) -> bool:
    """Return the labels match."""
    return all(left.get(key) == value for key, value in right.items())


def _egress_keys(rule: dict[str, Any], port: dict[str, Any]) -> set[tuple[str, str, int]] | None:
    """Return the egress keys."""
    protocol = str(port.get("protocol", "TCP")).upper()
    try:
        number = int(port["port"])
    except (KeyError, TypeError, ValueError):
        return None
    destinations = rule.get("to") or []
    if not destinations:
        return None
    keys: set[tuple[str, str, int]] = set()
    for destination in destinations:
        pod = destination.get("podSelector") or {}
        pod_labels = pod.get("matchLabels") or {}
        namespace = destination.get("namespaceSelector") or {}
        namespace_labels = namespace.get("matchLabels") or {}
        if namespace_labels.get("kubernetes.io/metadata.name") == "kube-system" and pod_labels.get("k8s-app") == "kube-dns":
            keys.add(("dns", protocol, number))
            continue
        app = pod_labels.get("app")
        if app:
            keys.add((str(app), protocol, number))
            continue
        return None
    return keys or None


def run_deployment_preflight(
    deployment: str | Path,
    network_policy: str | Path,
    auth_config: dict[str, Any],
) -> dict[str, Any]:
    """Return a JSON-safe static verdict; never contacts Kubernetes."""
    failures: list[str] = []
    checks: list[str] = []
    deployments = [document for document in _documents(deployment) if document.get("kind") == "Deployment"]
    parser_deployments = [
        document
        for document in deployments
        if ((document.get("metadata") or {}).get("name") == "agenthub-parser")
        or _labels_match(((document.get("spec") or {}).get("selector") or {}).get("matchLabels") or {}, _PARSER_LABEL)
    ]
    if len(parser_deployments) != 1:
        failures.append("parser_deployment_missing_or_ambiguous")
    else:
        parser = parser_deployments[0]
        selector = ((parser.get("spec") or {}).get("selector") or {}).get("matchLabels") or {}
        labels = ((((parser.get("spec") or {}).get("template") or {}).get("metadata") or {}).get("labels") or {})
        checks.append("parser_selector_present")
        if not _labels_match(labels, selector) or not _labels_match(selector, _PARSER_LABEL):
            failures.append("parser_selector_mismatch")
        else:
            checks.append("parser_selector_aligned")

    policies = [document for document in _documents(network_policy) if document.get("kind") == "NetworkPolicy"]
    parser_policies = [
        document
        for document in policies
        if _labels_match((((document.get("spec") or {}).get("podSelector") or {}).get("matchLabels") or {}), _PARSER_LABEL)
    ]
    deny_policies = [document for document in parser_policies if (document.get("spec") or {}).get("egress") == []]
    if not deny_policies:
        failures.append("parser_default_deny_missing")
    else:
        checks.append("parser_default_deny_present")
    observed: set[tuple[str, str, int]] = set()
    for policy in parser_policies:
        for rule in (policy.get("spec") or {}).get("egress") or []:
            ports = rule.get("ports") or []
            if not rule.get("to") or not ports:
                failures.append("undeclared_egress")
                continue
            for port in ports:
                keys = _egress_keys(rule, port)
                if keys is None:
                    failures.append("egress_rule_unparseable")
                else:
                    observed.update(keys)
    unexpected = observed - _ALLOWED_PORTS
    if unexpected:
        failures.append("undeclared_egress")
    if observed:
        checks.append("parser_egress_allowlist_checked")
    if not {key for key in observed if key != ("dns", "UDP", 53) and key != ("dns", "TCP", 53)} >= {
        ("redis-agenthub", "TCP", 6379),
        ("neo4j-agenthub", "TCP", 7687),
        ("chroma-agenthub", "TCP", 8000),
    }:
        failures.append("required_internal_egress_missing")
    if not {
        ("dns", "UDP", 53),
        ("dns", "TCP", 53),
    }.issubset(observed):
        failures.append("required_dns_egress_missing")

    environment = str(auth_config.get("app_environment", "")).strip().lower()
    if environment != "production":
        failures.append("production_environment_missing")
    else:
        checks.append("production_environment")
    if bool(auth_config.get("allow_insecure_local_development", False)):
        failures.append("production_insecure_auth")
    else:
        checks.append("local_insecure_auth_disabled")

    return {
        "status": "fail" if failures else "pass",
        "evidence": "static_only",
        "failures": sorted(set(failures)),
        "checks": checks,
    }


def main(argv: Iterable[str] | None = None) -> int:
    """Run the command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deployment", type=Path, required=True)
    parser.add_argument("--network-policy", type=Path, required=True)
    parser.add_argument("--auth-config", type=Path, required=True, help="JSON object with app_environment and auth flags")
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        auth_config = json.loads(args.auth_config.read_text(encoding="utf-8"))
        if not isinstance(auth_config, dict):
            raise ValueError("auth config must be an object")
        result = run_deployment_preflight(args.deployment, args.network_policy, auth_config)
    except (OSError, TypeError, ValueError, yaml.YAMLError, json.JSONDecodeError):
        print(json.dumps({"status": "fail", "error": "invalid_preflight_input"}, ensure_ascii=False, sort_keys=True))
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["main", "run_deployment_preflight"]
