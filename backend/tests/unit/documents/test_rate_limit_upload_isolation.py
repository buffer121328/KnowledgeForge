from __future__ import annotations

import ast
import asyncio
import io
import json
import os
import time
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI, HTTPException, Response, UploadFile
from fastapi.testclient import TestClient
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from starlette.requests import Request
from unittest.mock import AsyncMock, Mock

from domain.identity import Permission, UserContext, UserRole
from infrastructure.documents import local_uploads
from shared.config.settings import Settings
from shared.utils import ratelimit


def _request(
    *,
    client: str = "203.0.113.10",
    path: str = "/api/qa/ask",
    forwarded_for: str | None = None,
    user: UserContext | None = None,
) -> Request:
    headers: list[tuple[bytes, bytes]] = []
    if forwarded_for is not None:
        headers.append((b"x-forwarded-for", forwarded_for.encode()))
    request = Request(
        {
            "type": "http",
            "method": "POST",
            "path": path,
            "headers": headers,
            "client": (client, 43123),
            "server": ("testserver", 80),
            "scheme": "http",
            "query_string": b"",
        }
    )
    if user is not None:
        request.state.user = user
    return request


def _user(*, user_id: str = "user-a", api_key_id: str | None = None) -> UserContext:
    return UserContext(
        user_id=user_id,
        username="must-not-enter-limit-key",
        role=UserRole.API_USER if api_key_id else UserRole.EDITOR,
        org_id="org-a",
        permissions=[Permission.QA_QUERY],
        api_key_id=api_key_id,
    )


class TestCompositeRateLimitIdentity:
    def test_untrusted_forwarded_header_is_ignored(self) -> None:
        request = _request(client="198.51.100.7", forwarded_for="10.0.0.1")

        assert ratelimit.resolve_client_ip(
            request,
            trusted_proxy_cidrs=("192.0.2.0/24",),
        ) == "198.51.100.7"

    def test_trusted_proxy_chain_uses_nearest_untrusted_address(self) -> None:
        request = _request(
            client="10.0.0.3",
            forwarded_for="198.51.100.77, 203.0.113.9, 10.0.0.2",
        )

        assert ratelimit.resolve_client_ip(
            request,
            trusted_proxy_cidrs=("10.0.0.0/8", "203.0.113.0/24"),
        ) == "198.51.100.77"

    def test_malformed_trusted_proxy_chain_falls_back_to_peer(self) -> None:
        request = _request(client="10.0.0.3", forwarded_for="not-an-ip, 10.0.0.2")

        assert ratelimit.resolve_client_ip(
            request,
            trusted_proxy_cidrs=("10.0.0.0/8",),
        ) == "10.0.0.3"

    def test_user_composite_key_is_stable_and_secret_free(self) -> None:
        request = _request(user=_user())

        first = ratelimit.authenticated_composite_key(request)
        second = ratelimit.authenticated_composite_key(request)

        assert first == second
        assert first.startswith("rl:v1:")
        assert "org-a" not in first
        assert "user-a" not in first
        assert "203.0.113.10" not in first
        assert "must-not-enter-limit-key" not in first

    def test_api_key_subject_uses_record_id_without_raw_header(self) -> None:
        request = _request(user=_user(api_key_id="key-record-a"))
        request.scope["headers"].append((b"x-api-key", b"ak_live_super-secret-material"))

        key = ratelimit.authenticated_composite_key(request)

        assert "key-record-a" not in key
        assert "ak_live" not in key
        assert key != ratelimit.authenticated_composite_key(
            _request(user=_user(api_key_id="key-record-b"))
        )

    def test_public_key_is_route_and_validated_ip_scoped(self) -> None:
        login = _request(path="/api/auth/login")
        refresh = _request(path="/api/auth/refresh")

        login_key = ratelimit.public_route_ip_key(login)

        assert login_key.startswith("rl:v1:")
        assert login_key != ratelimit.public_route_ip_key(refresh)
        assert "203.0.113.10" not in login_key

    def test_production_rejects_process_local_limiter_storage(self) -> None:
        with pytest.raises(ValueError, match="shared"):
            ratelimit.validate_rate_limit_configuration(
                environment="production",
                storage_uri="memory://",
            )

    def test_throttle_response_is_stable_and_secret_free(self) -> None:
        request = _request(user=_user(api_key_id="key-record-a"))
        request.scope["headers"].append((b"x-api-key", b"ak_live_super-secret-material"))

        response = ratelimit.rate_limit_exceeded_handler(
            request,
            SimpleNamespace(detail="1 per 1 minute"),
        )
        payload = json.loads(response.body)

        assert response.status_code == 429
        assert payload["detail"]["code"] == "rate_limit_exceeded"
        assert payload["detail"]["request_id"]
        assert response.headers["x-request-id"] == payload["detail"]["request_id"]
        assert response.headers["retry-after"] == "60"
        assert "ak_live" not in response.body.decode()
        assert "key-record-a" not in response.body.decode()

    def test_sensitive_policy_names_are_complete(self) -> None:
        expected = {
            "auth_public",
            "qa_ask",
            "doc_upload",
            "doc_batch_upload",
            "password",
            "api_key_mutation",
            "webhook_test",
        }
        assert expected <= set(ratelimit.RATE_LIMITS)

    def test_sensitive_route_decorators_use_expected_key_functions(self) -> None:
        from api.routers.apikeys import (
            create_api_key,
            delete_api_key,
            rotate_api_key,
            toggle_api_key,
            update_api_key_scopes,
        )
        from api.routers.auth import login, refresh_token, register
        from api.routers.documents_ingest import upload_batch, upload_document
        from api.routers.documents_read import retry_document
        from api.routers.qa import ask_question
        from api.routers.users import change_own_password, reset_user_password
        from api.routers.webhooks import test_webhook

        public = (login, register, refresh_token)
        protected = (
            ask_question,
            upload_document,
            upload_batch,
            retry_document,
            change_own_password,
            reset_user_password,
            create_api_key,
            rotate_api_key,
            update_api_key_scopes,
            delete_api_key,
            toggle_api_key,
            test_webhook,
        )
        for endpoint in public:
            name = f"{endpoint.__module__}.{endpoint.__name__}"
            assert ratelimit.limiter._route_limits[name][0].key_func is ratelimit.public_route_ip_key
        for endpoint in protected:
            name = f"{endpoint.__module__}.{endpoint.__name__}"
            assert ratelimit.limiter._route_limits[name][0].key_func is ratelimit.authenticated_composite_key

    def test_limited_routes_accept_annotated_request_and_response(self) -> None:
        routers_dir = Path(__file__).resolve().parents[3] / "api" / "routers"
        limited_endpoints: list[str] = []

        for source_path in routers_dir.glob("*.py"):
            tree = ast.parse(source_path.read_text())
            for node in tree.body:
                if not isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef)):
                    continue
                if not any(
                    isinstance(decorator, ast.Call)
                    and isinstance(decorator.func, ast.Attribute)
                    and decorator.func.attr == "limit"
                    for decorator in node.decorator_list
                ):
                    continue

                limited_endpoints.append(f"{source_path.name}:{node.name}")
                parameters = {
                    argument.arg: ast.unparse(argument.annotation)
                    for argument in node.args.args
                    if argument.annotation is not None
                }
                assert parameters.get("request") == "Request", limited_endpoints[-1]
                assert parameters.get("response") == "Response", limited_endpoints[-1]

        assert len(limited_endpoints) == 16

    def test_storage_failure_response_is_stable(self) -> None:
        response = ratelimit.rate_limit_storage_error_handler(
            _request(user=_user()),
            RuntimeError("redis://internal:6379 secret detail"),
        )
        payload = json.loads(response.body)

        assert response.status_code == 503
        assert payload["detail"]["code"] == "rate_limit_unavailable"
        assert payload["detail"]["request_id"]
        assert "redis://" not in response.body.decode()

    def test_exhausted_subject_is_throttled_without_spending_another_subject(self) -> None:
        from slowapi import Limiter

        local_limiter = Limiter(key_func=ratelimit.authenticated_composite_key)
        app = FastAPI()
        app.state.limiter = local_limiter
        app.add_exception_handler(RateLimitExceeded, ratelimit.rate_limit_exceeded_handler)
        app.add_middleware(SlowAPIMiddleware)

        @app.middleware("http")
        async def authenticate_for_test(request, call_next):
            user_id = request.headers.get("x-test-user", "user-a")
            request.state.user = _user(user_id=user_id)
            return await call_next(request)

        @app.get("/limited")
        @local_limiter.limit("1/minute", key_func=ratelimit.authenticated_composite_key)
        async def limited(request: Request):
            return {"ok": True}

        client = TestClient(app)
        assert client.get("/limited", headers={"x-test-user": "user-a"}).status_code == 200
        throttled = client.get("/limited", headers={"x-test-user": "user-a"})
        assert throttled.status_code == 429
        assert throttled.json()["detail"]["code"] == "rate_limit_exceeded"
        assert client.get("/limited", headers={"x-test-user": "user-b"}).status_code == 200


class TestTenantUploadStorage:
    def test_bounded_stage_and_atomic_promotion_use_tenant_namespace(self, tmp_path: Path) -> None:
        storage = local_uploads.LocalUploadStorage(str(tmp_path))

        staged = storage.stage(
            tenant_id="org-a",
            original_filename="../../report.txt",
            source=io.BytesIO(b"hello"),
            max_file_bytes=10,
            tenant_quota_bytes=20,
        )
        accepted = Path(storage.promote(staged))

        assert accepted.read_bytes() == b"hello"
        assert accepted.parent.name == "accepted"
        assert accepted.parent.parent.parent.name == "tenants"
        assert "org-a" not in str(accepted)
        assert ".." not in accepted.name
        assert not Path(staged.path).exists()
        assert os.stat(accepted).st_mode & 0o777 == 0o600

    def test_oversized_partial_file_is_removed(self, tmp_path: Path) -> None:
        storage = local_uploads.LocalUploadStorage(str(tmp_path))

        with pytest.raises(local_uploads.UploadPolicyError) as exc_info:
            storage.stage(
                tenant_id="org-a",
                original_filename="large.txt",
                source=io.BytesIO(b"123456"),
                max_file_bytes=5,
                tenant_quota_bytes=100,
            )

        assert exc_info.value.code == "upload_too_large"
        assert not list(tmp_path.rglob("*.part"))

    def test_tenant_quota_does_not_cross_tenants(self, tmp_path: Path) -> None:
        storage = local_uploads.LocalUploadStorage(str(tmp_path))
        first = storage.stage(
            tenant_id="org-a",
            original_filename="a.txt",
            source=io.BytesIO(b"12345"),
            max_file_bytes=10,
            tenant_quota_bytes=5,
        )
        storage.promote(first)

        with pytest.raises(local_uploads.UploadPolicyError) as exc_info:
            storage.stage(
                tenant_id="org-a",
                original_filename="b.txt",
                source=io.BytesIO(b"x"),
                max_file_bytes=10,
                tenant_quota_bytes=5,
            )
        other = storage.stage(
            tenant_id="org-b",
            original_filename="b.txt",
            source=io.BytesIO(b"x"),
            max_file_bytes=10,
            tenant_quota_bytes=5,
        )

        assert exc_info.value.code == "tenant_upload_quota_exceeded"
        assert Path(other.path).is_file()

    def test_delete_requires_matching_tenant_accepted_directory(self, tmp_path: Path) -> None:
        storage = local_uploads.LocalUploadStorage(str(tmp_path))
        staged = storage.stage(
            tenant_id="org-a",
            original_filename="a.txt",
            source=io.BytesIO(b"safe"),
            max_file_bytes=10,
            tenant_quota_bytes=100,
        )
        accepted = storage.promote(staged)
        legacy = tmp_path / "legacy.txt"
        legacy.write_text("legacy", encoding="utf-8")

        assert storage.delete(accepted, tenant_id="org-b") is False
        assert Path(accepted).exists()
        assert storage.delete(str(legacy), tenant_id="org-a") is False
        assert legacy.exists()
        assert storage.delete(accepted, tenant_id="org-a") is True

    def test_cleanup_only_removes_expired_staging_and_quarantine(self, tmp_path: Path) -> None:
        storage = local_uploads.LocalUploadStorage(str(tmp_path))
        staged = storage.stage(
            tenant_id="org-a",
            original_filename="a.txt",
            source=io.BytesIO(b"safe"),
            max_file_bytes=10,
            tenant_quota_bytes=100,
        )
        quarantine = storage.quarantine(staged)
        outside = tmp_path / "outside.part"
        outside.write_bytes(b"keep")
        old = time.time() - 7200
        os.utime(quarantine, (old, old))
        os.utime(outside, (old, old))

        removed = storage.cleanup("org-a", older_than_seconds=3600, now=time.time())

        assert removed == 1
        assert not Path(quarantine).exists()
        assert outside.exists()


def _write_ooxml(path: Path, entries: dict[str, bytes]) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        for name, content in entries.items():
            archive.writestr(name, content)


class TestUploadValidationPolicy:
    def _policy(self, **overrides):
        values = {
            "max_archive_entries": 20,
            "max_archive_uncompressed_bytes": 100_000,
            "max_archive_compression_ratio": 20,
            "max_pdf_pages": 3,
            "max_image_pixels": 10_000,
            "max_spreadsheet_rows": 3,
            "max_text_characters": 100,
        }
        values.update(overrides)
        return local_uploads.UploadValidationPolicy(**values)

    def test_valid_pdf_signature_is_accepted(self, tmp_path: Path) -> None:
        from pypdf import PdfWriter

        pdf = tmp_path / "ok.pdf"
        writer = PdfWriter()
        writer.add_blank_page(width=72, height=72)
        with pdf.open("wb") as target:
            writer.write(target)

        result = self._policy().validate(pdf, "ok.pdf", "application/pdf")

        assert result.parser_type == "pdf"

    def test_renamed_executable_is_rejected(self, tmp_path: Path) -> None:
        fake = tmp_path / "fake.pdf"
        fake.write_bytes(b"MZ\x90\x00arbitrary")

        with pytest.raises(local_uploads.UploadPolicyError) as exc_info:
            self._policy().validate(fake, "fake.pdf", "application/pdf")

        assert exc_info.value.code == "upload_type_mismatch"

    def test_docx_requires_word_container_entries(self, tmp_path: Path) -> None:
        fake = tmp_path / "fake.docx"
        _write_ooxml(
            fake,
            {
                "[Content_Types].xml": b"<Types/>",
                "xl/workbook.xml": b"<workbook/>",
            },
        )

        with pytest.raises(local_uploads.UploadPolicyError) as exc_info:
            self._policy().validate(
                fake,
                "fake.docx",
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            )

        assert exc_info.value.code == "upload_type_mismatch"

    def test_eicar_signature_is_rejected_without_echo(self, tmp_path: Path) -> None:
        eicar = (
            b"X5O!P%@AP[4\\PZX54(P^)7CC)7}$"
            b"EICAR-STANDARD-ANTIVIRUS-TEST-FILE!$H+H*"
        )
        sample = tmp_path / "sample.txt"
        sample.write_bytes(eicar)

        with pytest.raises(local_uploads.UploadPolicyError) as exc_info:
            self._policy().validate(sample, "sample.txt", "text/plain")

        assert exc_info.value.code == "upload_malicious"
        assert "EICAR" not in str(exc_info.value)

    def test_macro_enabled_ooxml_is_rejected(self, tmp_path: Path) -> None:
        docx = tmp_path / "macro.docx"
        _write_ooxml(
            docx,
            {
                "[Content_Types].xml": b"<Types/>",
                "word/document.xml": b"<document/>",
                "word/vbaProject.bin": b"macro",
            },
        )

        with pytest.raises(local_uploads.UploadPolicyError) as exc_info:
            self._policy().validate(
                docx,
                "macro.docx",
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            )

        assert exc_info.value.code == "upload_malicious"

    def test_eicar_inside_ooxml_member_is_rejected(self, tmp_path: Path) -> None:
        marker = (
            b"X5O!P%@AP[4\\PZX54(P^)7CC)7}$"
            b"EICAR-STANDARD-ANTIVIRUS-TEST-FILE!$H+H*"
        )
        docx = tmp_path / "member.docx"
        _write_ooxml(
            docx,
            {
                "[Content_Types].xml": b"<Types/>",
                "word/document.xml": b"<document/>",
                "word/media/note.txt": marker,
            },
        )

        with pytest.raises(local_uploads.UploadPolicyError) as exc_info:
            self._policy().validate(
                docx,
                "member.docx",
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            )

        assert exc_info.value.code == "upload_malicious"

    def test_text_work_limit_is_enforced(self, tmp_path: Path) -> None:
        text_file = tmp_path / "long.txt"
        text_file.write_text("a" * 101, encoding="utf-8")

        with pytest.raises(local_uploads.UploadPolicyError) as exc_info:
            self._policy().validate(text_file, "long.txt", "text/plain")

        assert exc_info.value.code == "upload_work_limit_exceeded"

    def test_pdf_page_work_limit_is_enforced(self, tmp_path: Path) -> None:
        from pypdf import PdfWriter

        pdf = tmp_path / "two.pdf"
        writer = PdfWriter()
        writer.add_blank_page(width=72, height=72)
        writer.add_blank_page(width=72, height=72)
        with pdf.open("wb") as target:
            writer.write(target)

        with pytest.raises(local_uploads.UploadPolicyError) as exc_info:
            self._policy(max_pdf_pages=1).validate(pdf, "two.pdf", "application/pdf")

        assert exc_info.value.code == "upload_work_limit_exceeded"

    def test_required_external_scanner_fails_closed(self, tmp_path: Path) -> None:
        text_file = tmp_path / "safe.txt"
        text_file.write_text("safe", encoding="utf-8")

        with pytest.raises(local_uploads.UploadPolicyError) as exc_info:
            self._policy(require_external_scanner=True).validate(
                text_file,
                "safe.txt",
                "text/plain",
            )

        assert exc_info.value.code == "upload_scanner_unavailable"
        assert exc_info.value.status_code == 503


class TestUploadRouteIsolation:
    @staticmethod
    def _route_request(workflow) -> Request:
        app = SimpleNamespace(
            state=SimpleNamespace(
                workflows={"ingest": workflow},
            )
        )
        request = _request(user=_user())
        request.scope["app"] = app
        request.state.tenant_id = "org-a"
        return request

    @pytest.mark.asyncio
    async def test_type_mismatch_skips_ingestion_workflow(self, tmp_path: Path, monkeypatch) -> None:
        from api.dependencies import documents as document_dependencies
        from api.routers import documents_ingest as documents

        workflow = SimpleNamespace(
            ainvoke=AsyncMock(return_value={"chunks": [], "extractions": []})
        )
        audit_service = SimpleNamespace(log=Mock())
        monkeypatch.setattr(documents.settings, "upload_dir", str(tmp_path))
        monkeypatch.setattr(document_dependencies, "get_audit_service", lambda: audit_service)
        request = self._route_request(workflow)
        submission = document_dependencies.get_document_submission(request)
        upload = UploadFile(
            filename="renamed.pdf",
            file=io.BytesIO(b"MZ\x90\x00arbitrary"),
            headers={"content-type": "application/pdf"},
        )

        with pytest.raises(HTTPException) as exc_info:
            await documents.upload_document.__wrapped__(request, Response(), upload, _user(), submission)

        assert exc_info.value.status_code == 400
        assert exc_info.value.detail["code"] == "upload_type_mismatch"
        workflow.ainvoke.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_ingestion_failure_removes_promoted_original(self, tmp_path: Path, monkeypatch) -> None:
        from api.dependencies import documents as document_dependencies
        from api.routers import documents_ingest as documents

        workflow = SimpleNamespace(ainvoke=AsyncMock())
        workflow.ainvoke.side_effect = RuntimeError("sensitive parser detail")
        audit_service = SimpleNamespace(log=Mock())
        monkeypatch.setattr(documents.settings, "upload_dir", str(tmp_path))
        monkeypatch.setattr(document_dependencies, "get_audit_service", lambda: audit_service)
        request = self._route_request(workflow)
        submission = document_dependencies.get_document_submission(request)
        upload = UploadFile(
            filename="safe.txt",
            file=io.BytesIO(b"safe text"),
            headers={"content-type": "text/plain"},
        )

        with pytest.raises(HTTPException) as exc_info:
            await documents.upload_document.__wrapped__(request, Response(), upload, _user(), submission)

        assert exc_info.value.status_code == 500
        assert exc_info.value.detail["code"] == "document_ingest_failed"
        assert "sensitive parser detail" not in str(exc_info.value.detail)
        assert not [
            path
            for path in tmp_path.rglob("*")
            if path.is_file() and path.name != ".quota.lock"
        ]

    @pytest.mark.asyncio
    async def test_successful_upload_passes_only_tenant_accepted_path(self, tmp_path: Path, monkeypatch) -> None:
        from api.dependencies import documents as document_dependencies
        from api.routers import documents_ingest as documents

        workflow = SimpleNamespace(
            ainvoke=AsyncMock(return_value={"chunks": [], "extractions": []})
        )
        webhook = SimpleNamespace(trigger=AsyncMock())
        audit_service = SimpleNamespace(log=Mock())
        monkeypatch.setattr(documents.settings, "upload_dir", str(tmp_path))
        monkeypatch.setattr(document_dependencies, "get_audit_service", lambda: audit_service)
        monkeypatch.setattr(document_dependencies, "get_webhook_service", lambda: webhook)
        request = self._route_request(workflow)
        submission = document_dependencies.get_document_submission(request)
        upload = UploadFile(
            filename="../../safe.txt",
            file=io.BytesIO(b"safe text"),
            headers={"content-type": "text/plain"},
        )

        response = await documents.upload_document.__wrapped__(
            request,
            Response(),
            upload,
            _user(),
            submission,
        )

        submitted_path = Path(workflow.ainvoke.await_args.args[0]["file_paths"][0])
        assert submitted_path.is_file()
        assert submitted_path.parent.name == "accepted"
        assert "org-a" not in str(submitted_path)
        assert response.file_name == "safe.txt"
        webhook.trigger.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_batch_with_exactly_fifty_files_is_processed_in_request_order(
        self, monkeypatch
    ) -> None:
        from api.routers import documents_ingest as documents

        uploads = [
            UploadFile(filename=f"file-{index}.txt", file=io.BytesIO(b"safe"))
            for index in range(50)
        ]
        upload_one = AsyncMock(
            side_effect=[SimpleNamespace(file_name=upload.filename) for upload in uploads]
        )
        monkeypatch.setattr(documents.settings, "upload_max_batch_files", 50)
        monkeypatch.setattr(documents, "_upload_one", upload_one)
        request = self._route_request(SimpleNamespace())

        results = await documents.upload_batch.__wrapped__(
            request,
            Response(),
            uploads,
            _user(),
        )

        assert [result.file_name for result in results] == [
            upload.filename for upload in uploads
        ]
        assert [call.args[1] for call in upload_one.await_args_list] == uploads

    @pytest.mark.asyncio
    async def test_batch_upload_starts_multiple_files_before_first_result(
        self, monkeypatch
    ) -> None:
        from api.routers import documents_ingest as documents

        uploads = [
            UploadFile(filename=f"file-{index}.txt", file=io.BytesIO(b"safe"))
            for index in range(3)
        ]
        started: list[str] = []
        release = asyncio.Event()

        async def upload_one(_request, upload: UploadFile, _user_context, _submission):
            started.append(upload.filename or "")
            if len(started) == len(uploads):
                release.set()
            await release.wait()
            return SimpleNamespace(file_name=upload.filename)

        monkeypatch.setattr(documents.settings, "upload_max_batch_files", 50)
        monkeypatch.setattr(documents, "_BATCH_UPLOAD_CONCURRENCY", 3)
        monkeypatch.setattr(documents, "_upload_one", upload_one)
        request = self._route_request(SimpleNamespace())

        results = await documents.upload_batch.__wrapped__(
            request,
            Response(),
            uploads,
            _user(),
        )

        assert started == [upload.filename for upload in uploads]
        assert [result.file_name for result in results] == [
            upload.filename for upload in uploads
        ]

    @pytest.mark.asyncio
    async def test_document_list_uses_user_org_when_request_tenant_is_missing(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        from api.routers import documents_read as documents

        storage = local_uploads.LocalUploadStorage(str(tmp_path))
        staged = storage.stage(
            tenant_id="org-a",
            original_filename="policy.txt",
            source=io.BytesIO(b"safe"),
            max_file_bytes=10,
            tenant_quota_bytes=100,
        )
        accepted = storage.promote(staged)
        vector_store = SimpleNamespace(
            list_documents=AsyncMock(
                return_value=[
                    {
                        "doc_id": "doc-a",
                        "source": accepted,
                        "file_name": "policy.txt",
                        "doc_type": "text",
                        "chunks_count": 1,
                    }
                ]
            )
        )
        request = self._route_request(SimpleNamespace())
        request.app.state.vector_store = vector_store
        request.state.tenant_id = ""
        monkeypatch.setattr(documents.settings, "upload_dir", str(tmp_path))

        result = await documents.list_documents(request, _user())

        vector_store.list_documents.assert_awaited_once_with(tenant_id="org-a")
        assert result[0].file_reference == Path(accepted).name

    def test_upload_reference_rejects_missing_tenant_without_attribute_error(
        self, tmp_path: Path
    ) -> None:
        storage = local_uploads.LocalUploadStorage(str(tmp_path))

        with pytest.raises(local_uploads.UploadPolicyError) as exc_info:
            storage.accepted_reference(tmp_path / "missing.txt", None)

        assert exc_info.value.code == "upload_tenant_required"

    @pytest.mark.asyncio
    async def test_batch_with_fifty_one_files_is_rejected_before_processing_and_audited(
        self, monkeypatch
    ) -> None:
        from api.dependencies import documents as document_dependencies
        from api.routers import documents_ingest as documents

        uploads = [
            UploadFile(filename=f"file-{index}.txt", file=io.BytesIO(b"safe"))
            for index in range(51)
        ]
        upload_one = AsyncMock()
        audit_service = SimpleNamespace(log=Mock())
        monkeypatch.setattr(documents.settings, "upload_max_batch_files", 50)
        monkeypatch.setattr(documents, "_upload_one", upload_one)
        monkeypatch.setattr(document_dependencies, "get_audit_service", lambda: audit_service)
        request = self._route_request(SimpleNamespace())
        submission = document_dependencies.get_document_submission(request)

        with pytest.raises(HTTPException) as exc_info:
            await documents.upload_batch.__wrapped__(
                request, Response(), uploads, _user(), submission
            )

        assert exc_info.value.status_code == 400
        assert exc_info.value.detail["code"] == "upload_batch_too_many_files"
        upload_one.assert_not_awaited()
        audit_service.log.assert_called_once()
        assert audit_service.log.call_args.kwargs["result"].value == "denied"
        assert audit_service.log.call_args.kwargs["metadata"] == {
            "code": "upload_batch_too_many_files",
            "size_bytes": 0,
            "file_count": 51,
        }


def test_settings_expose_bounded_upload_and_rate_limit_defaults() -> None:
    configured = Settings(
        _env_file=None,
        upload_max_file_bytes=1_000,
        upload_tenant_quota_bytes=2_000,
        rate_limit_storage_uri="memory://",
    )

    assert configured.upload_max_file_bytes == 1_000
    assert configured.upload_max_batch_files == 50
    assert configured.upload_tenant_quota_bytes == 2_000
    assert configured.upload_tenant_quota_bytes >= configured.upload_max_file_bytes

    with pytest.raises(ValueError):
        Settings(_env_file=None, upload_max_batch_files=51)


class TestParserDefenseInDepth:
    @pytest.mark.asyncio
    async def test_internal_text_parse_enforces_character_limit(self, tmp_path: Path, monkeypatch) -> None:
        from agents import document_parser
        from unittest.mock import patch

        text_file = tmp_path / "long.txt"
        text_file.write_text("abcdef", encoding="utf-8")
        monkeypatch.setattr("agents.document_format_parsers.settings.upload_max_text_characters", 5)
        with patch("agents.document_format_parsers.ChatOpenAI"):
            parser = document_parser.DocParserAgent()

        with pytest.raises(document_parser.DocumentWorkLimitError):
            await parser.parse(str(text_file), tenant_id="org-a")

    @pytest.mark.asyncio
    async def test_internal_pdf_parse_enforces_page_limit(self, tmp_path: Path, monkeypatch) -> None:
        from agents import document_parser
        from pypdf import PdfWriter
        from unittest.mock import patch

        pdf = tmp_path / "two-pages.pdf"
        writer = PdfWriter()
        writer.add_blank_page(width=72, height=72)
        writer.add_blank_page(width=72, height=72)
        with pdf.open("wb") as target:
            writer.write(target)
        monkeypatch.setattr("agents.document_format_parsers.settings.upload_max_pdf_pages", 1)
        with patch("agents.document_format_parsers.ChatOpenAI"):
            parser = document_parser.DocParserAgent()

        with pytest.raises(document_parser.DocumentWorkLimitError):
            await parser.parse(str(pdf), tenant_id="org-a")
