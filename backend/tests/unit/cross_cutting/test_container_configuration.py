"""Acceptance checks for the repository's Docker application stack."""

import ast
import re
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[4]


def _read(relative_path: str) -> str:
    """Read a repository file used by the deployment acceptance checks."""
    return (REPOSITORY_ROOT / relative_path).read_text(encoding="utf-8")


def _env_entries(content: str) -> list[tuple[str, str]]:
    """Return active KEY=value entries without consulting a real environment file."""
    entries: list[tuple[str, str]] = []
    for raw_line in content.splitlines():
        line = raw_line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", maxsplit=1)
            entries.append((key, value))
    return entries


def _settings_env_keys(relative_path: str, class_name: str, *, prefix: str = "") -> set[str]:
    """Extract declared Pydantic setting names without loading any local .env file."""
    tree = ast.parse(_read(relative_path))
    model = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    return {
        f"{prefix}{statement.target.id.upper()}"
        for statement in model.body
        if isinstance(statement, ast.AnnAssign)
        and isinstance(statement.target, ast.Name)
    }


def _compose_env_keys(content: str) -> set[str]:
    """Return variables expanded by Docker Compose."""
    return set(re.findall(r"\$\{([A-Z][A-Z0-9_]*)", content))


def _prepare_env_keys(content: str) -> set[str]:
    """Return keys explicitly rewritten by the remote preparation script."""
    return set(re.findall(r"^set_env ([A-Z][A-Z0-9_]*) ", content, flags=re.MULTILINE))


def test_environment_example_is_complete_unique_and_secret_free() -> None:
    """The committed template is the complete safe operator configuration inventory."""
    template = _read("config/.env.example")
    entries = _env_entries(template)
    keys = [key for key, _ in entries]
    values = dict(entries)

    assert len(keys) == len(set(keys))

    expected_keys = _settings_env_keys(
        "backend/shared/config/settings.py",
        "Settings",
    )
    expected_keys.update(
        _settings_env_keys(
            "backend/auth/config.py",
            "AuthSettings",
            prefix="AUTH_",
        )
    )
    expected_keys.update(_compose_env_keys(_read("docker-compose.yml")))
    expected_keys.update(
        {
            "CELERY_BROKER_URL",
            "CELERY_RESULT_BACKEND",
            "EVAL_OPENAI_API_KEY",
            "EVAL_OPENAI_BASE_URL",
            "EVAL_OPENAI_MODEL",
            "LOG_LEVEL",
            "SERVICE_NAME",
        }
    )

    assert expected_keys <= set(keys), sorted(expected_keys - set(keys))
    assert values["QA_EVIDENCE_GATE_MODE"] == "off"
    assert values["QA_CROSS_ENCODER_MODE"] == "disabled"
    assert values["QA_EVIDENCE_POLICY_VERSION"] == "evidence-composite-v2"
    assert values["QA_EVIDENCE_CALIBRATION_VERSION"] == "qualification-boundaries-v2"
    assert values["BUILD_APT_MIRROR"] == "https://mirrors.tencentyun.com/debian"
    assert values["BUILD_NPM_REGISTRY"] == "https://registry.npmmirror.com"

    assert values["DEEPSEEK_API_KEY"] == "your-mimo-api-key-here"
    assert values["MIMO_API_KEY"] == "your-mimo-api-key-here"
    assert values["MIMO_BASE_URL"] == "https://api.xiaomimimo.com/v1"
    assert values["MIMO_MODEL"] == "mimo-v2.5-pro"
    assert values["DASHSCOPE_API_KEY"] == "your-dashscope-api-key-here"
    assert values["NEO4J_PASSWORD"] == "change-me"
    assert values["POSTGRES_PASSWORD"] == "replace-me"
    assert "replace-me" in values["DATABASE_URL"]
    assert values["AUTH_SECRET_KEY"] == "REPLACE_VIA_SECRET_MANAGER"
    assert values["WEBHOOK_SECRET_ENCRYPTION_KEY"] == "REPLACE_WITH_URLSAFE_BASE64_32_BYTE_KEY"
    assert values["EVAL_OPENAI_API_KEY"] == "your-mimo-api-key-here"
    assert values["EVAL_OPENAI_BASE_URL"] == "https://api.xiaomimimo.com/v1"
    assert values["EVAL_OPENAI_MODEL"] == "mimo-v2.5-pro"
    assert not any(
        marker in template
        for marker in (
            "sk-proj-",
            "sk-ant-",
            "BEGIN PRIVATE KEY",
            "AKIA",
        )
    )


def test_backend_image_is_locked_multistage_and_non_root() -> None:
    """The backend runtime must be reproducible, health-aware, and non-root."""
    dockerfile = _read("backend/Dockerfile")

    assert " AS builder" in dockerfile
    assert "uv sync --locked --no-dev --no-install-project" in dockerfile
    assert "--mount=type=cache,target=/root/.cache/uv,sharing=locked" in dockerfile
    assert "--group eval" not in dockerfile
    assert "FROM python:3.12-slim AS runtime" in dockerfile
    assert "ARG APT_MIRROR=http://deb.debian.org/debian" in dockerfile
    assert "/etc/apt/sources.list.d/debian.sources" in dockerfile
    assert "${apt_mirror}-security" in dockerfile
    assert "USER app" in dockerfile
    assert "HEALTHCHECK" in dockerfile
    assert "urllib.request.urlopen" in dockerfile


def test_evaluation_worker_image_installs_isolated_ragas_dependencies() -> None:
    """Only the evaluation Worker overlay may add the RAGAS dependency group."""
    dockerfile = _read("backend/Dockerfile.evaluation-worker")

    assert "FROM agenthub-api:2.0.0 AS evaluation-worker" in dockerfile
    assert "UV_PROJECT_ENVIRONMENT=/app/.venv" in dockerfile
    assert "--mount=type=cache,target=/root/.cache/uv,sharing=locked" in dockerfile
    assert (
        "uv sync --locked --active --no-dev --no-install-project "
        "--no-default-groups --group eval"
    ) in dockerfile
    assert "COPY --chown=app:app backend /app/backend" in dockerfile
    assert "USER app" in dockerfile


def test_backend_build_context_includes_evidence_governance_seed_bundle() -> None:
    """The API image must contain governance seeds and the corpus identity manifest."""
    dockerignore = _read(".dockerignore")

    assert "backend/evaluation/data/*" in dockerignore
    assert "!backend/evaluation/data/evidence-gates/" in dockerignore
    assert "!backend/evaluation/data/evidence-gates/**" in dockerignore
    assert "!backend/evaluation/data/company-demo/" in dockerignore
    assert "!backend/evaluation/data/company-demo/corpus_manifest.json" in dockerignore
    assert "\nbackend/evaluation/data\n" not in f"\n{dockerignore}\n"


def test_frontend_image_serves_spa_and_proxies_api() -> None:
    """The frontend image must build from its lock and serve SPA/API traffic."""
    dockerfile = _read("frontend/Dockerfile")
    nginx = _read("frontend/nginx.conf")

    assert "delete value.resolved" in dockerfile
    assert "npm ci" in dockerfile
    assert "--mount=type=cache,target=/root/.npm,sharing=locked" in dockerfile
    assert "--cache=/root/.npm" in dockerfile
    assert "npm run build" in dockerfile
    assert "USER nginx" in dockerfile
    assert "HEALTHCHECK" in dockerfile
    assert "listen 8080" in nginx
    assert "try_files $uri $uri/ /index.html" in nginx
    assert "resolver 127.0.0.11 valid=10s ipv6=off;" in nginx
    assert "set $api_upstream http://api:8080;" in nginx
    assert "proxy_pass $api_upstream;" in nginx
    assert "proxy_pass http://api:8080;" not in nginx
    assert "proxy_buffering off" in nginx
    assert "location = /healthz" in nginx


def test_compose_defines_immutable_complete_application_stack() -> None:
    """Compose must run built application images with durable shared data."""
    compose = _read("docker-compose.yml")

    assert "  frontend:" in compose
    assert "dockerfile: frontend/Dockerfile" in compose
    assert '"${FRONTEND_PORT:-5173}:8080"' in compose
    assert "condition: service_healthy" in compose
    assert "./backend:/app/backend" not in compose
    assert "uploads:/app/uploads" in compose
    assert "audit_logs:/app/logs" in compose
    assert "  audit_logs:" in compose
    assert "/api/health/ready" in compose


def test_compose_builds_an_isolated_ragas_worker_overlay() -> None:
    """Only the evaluation Worker adds RAGAS; migrations and parser reuse API image."""
    compose = _read("docker-compose.yml")
    api = compose.split("  api:", maxsplit=1)[1].split("\n  worker:", maxsplit=1)[0]
    migration = compose.split("  postgres-migrate:", maxsplit=1)[1].split("\n  neo4j:", maxsplit=1)[0]
    worker = compose.split("  worker:", maxsplit=1)[1].split("\n  parser-worker:", maxsplit=1)[0]
    parser_worker = compose.split("  parser-worker:", maxsplit=1)[1].split("\n  frontend:", maxsplit=1)[0]

    assert compose.count("dockerfile: backend/Dockerfile\n") == 1
    assert "dockerfile: backend/Dockerfile" in api
    assert "image: agenthub-evaluation-worker:2.0.0" in worker
    assert "dockerfile: backend/Dockerfile.evaluation-worker" in worker
    assert "\n    build:" in worker
    for reused_service in (migration, parser_worker):
        assert "image: agenthub-api:2.0.0" in reused_service
        assert "\n    build:" not in reused_service


def test_compose_disables_api_http_healthcheck_for_celery_workers() -> None:
    """Celery services must not inherit the API image HTTP healthcheck."""
    compose = _read("docker-compose.yml")
    worker = compose.split("  worker:", maxsplit=1)[1].split("\n  parser-worker:", maxsplit=1)[0]
    parser_worker = compose.split("  parser-worker:", maxsplit=1)[1].split("\n  frontend:", maxsplit=1)[0]

    assert "healthcheck:\n      disable: true" in worker
    assert "healthcheck:\n      disable: true" in parser_worker


def test_local_worker_defaults_to_one_memory_intensive_update_at_a_time() -> None:
    """Operators can opt in to higher throughput, but the local default is safe."""
    compose = _read("docker-compose.yml")
    worker = compose.split("  worker:", maxsplit=1)[1].split("\n  parser-worker:", maxsplit=1)[0]

    assert '"--concurrency=${WORKER_CONCURRENCY:-1}"' in worker


def test_chromadb_healthcheck_and_volume_match_pinned_image() -> None:
    """Chroma must use runtime-available health tools and its configured data path."""
    compose = _read("docker-compose.yml")
    chromadb = compose.split("  chromadb:", maxsplit=1)[1].split("\n  redis:", maxsplit=1)[0]

    assert "image: chromadb/chroma:1.5.9" in chromadb
    assert "chroma_data:/data" in chromadb
    assert "bash -ec" in chromadb
    assert "/dev/tcp/127.0.0.1/8000" in chromadb
    assert "GET /api/v2/heartbeat HTTP/1.1" in chromadb
    assert "200 OK" in chromadb
    assert "wget" not in chromadb
    assert "curl" not in chromadb


def test_compose_advertises_reachable_internal_and_external_kafka_listeners() -> None:
    """Kafka must advertise network-appropriate addresses to each client."""
    compose = _read("docker-compose.yml")

    assert "INTERNAL://kafka:29092" in compose
    assert "EXTERNAL://localhost:9092" in compose
    assert "KAFKA_INTER_BROKER_LISTENER_NAME: INTERNAL" in compose
    assert "KAFKA_BOOTSTRAP_SERVERS: kafka:29092" in compose


def test_docker_build_context_excludes_local_and_sensitive_files() -> None:
    """Docker contexts must exclude secrets, caches, and generated artifacts."""
    dockerignore = _read(".dockerignore")

    for entry in (
        ".git",
        "config/.env",
        "backend/.venv",
        "backend/.uv-cache",
        "backend/tests",
        "frontend/node_modules",
        "frontend/dist",
    ):
        assert entry in dockerignore


def test_kubernetes_distinguishes_liveness_and_readiness() -> None:
    """Kubernetes must not restart a live process for a dependency-only outage."""
    deployment = _read("deploy/k8s/deployment.yaml")

    assert deployment.count("path: /api/health/live") == 2
    assert deployment.count("path: /api/health/ready") == 1


def test_stack_persists_offline_evaluation_results_for_the_non_root_backend() -> None:
    """Ragas and quality-report artifacts survive API container replacement."""
    dockerfile = _read("backend/Dockerfile")
    compose = _read("docker-compose.yml")

    assert "/app/backend/evaluation/results" in dockerfile
    assert dockerfile.index("/app/backend/evaluation/results") < dockerfile.index("USER app")
    assert "evaluation_results:/app/backend/evaluation/results" in compose
    assert "  evaluation_results:" in compose


def test_stack_persists_native_bm25_for_the_non_root_backend() -> None:
    """Native sparse snapshots use a writable volume shared by backend services."""

    dockerfile = _read("backend/Dockerfile")
    compose = _read("docker-compose.yml")

    assert "QA_BM25_INDEX_PATH: /app/data/bm25" in compose
    assert "/app/data/bm25" in dockerfile
    assert dockerfile.index("/app/data/bm25") < dockerfile.index("USER app")
    assert compose.count("bm25_index:/app/data/bm25") == 3
    assert "  bm25_index:" in compose


REMOTE_LOG_SHORTCUTS = {
    "logs-api.sh": ("api",),
    "logs-frontend.sh": ("frontend",),
    "logs-worker.sh": ("worker",),
    "logs-parser-worker.sh": ("parser-worker",),
    "logs-data.sh": ("neo4j", "chromadb", "redis", "kafka"),
    "logs-all.sh": (
        "neo4j",
        "chromadb",
        "redis",
        "kafka",
        "api",
        "worker",
        "parser-worker",
        "frontend",
    ),
}


