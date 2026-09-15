"""Filesystem acceptance tests for catalog-owned document storage."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pytest

from infrastructure.documents.local_uploads import LocalUploadStorage, UploadPolicyError


def _stage(storage: LocalUploadStorage, *, tenant: str, filename: str, content: bytes):
    """Stage one small test file beneath a tenant namespace."""

    return storage.stage(
        tenant_id=tenant,
        original_filename=filename,
        source=BytesIO(content),
        max_file_bytes=1024,
        tenant_quota_bytes=4096,
        chunk_bytes=16,
    )


def test_catalog_promotion_preserves_basename_inside_server_owned_directories(
    tmp_path: Path,
) -> None:
    """The logical path never selects physical storage while the basename survives."""

    storage = LocalUploadStorage(str(tmp_path))
    staged = _stage(storage, tenant="tenant-a", filename="招聘制度.txt", content=b"policy")

    accepted = Path(
        storage.promote(
            staged,
            department_id="human_resources",
            doc_id="doc-1",
            original_filename="招聘制度.txt",
        )
    )

    assert accepted.name == "招聘制度.txt"
    assert accepted.parts[-5:] == (
        "departments",
        "human_resources",
        "documents",
        "doc-1",
        "招聘制度.txt",
    )
    reference = storage.accepted_reference(accepted, "tenant-a")
    assert reference == "departments/human_resources/documents/doc-1/招聘制度.txt"
    assert storage.resolve_accepted(reference, "tenant-a") == str(accepted)


def test_same_basename_isolated_by_document_and_tenant(tmp_path: Path) -> None:
    """Same-name files never depend on filename rewriting for isolation."""

    storage = LocalUploadStorage(str(tmp_path))
    paths = []
    for tenant, doc_id in (("tenant-a", "doc-a"), ("tenant-a", "doc-b"), ("tenant-b", "doc-a")):
        staged = _stage(storage, tenant=tenant, filename="制度.txt", content=doc_id.encode())
        paths.append(
            Path(
                storage.promote(
                    staged,
                    department_id="finance",
                    doc_id=doc_id,
                    original_filename="制度.txt",
                )
            )
        )

    assert len({path.resolve() for path in paths}) == 3
    assert [path.name for path in paths] == ["制度.txt", "制度.txt", "制度.txt"]
    assert paths[0].read_bytes() == b"doc-a"
    assert paths[1].read_bytes() == b"doc-b"


def test_catalog_promotion_rejects_unsafe_basename_and_incomplete_context(tmp_path: Path) -> None:
    """Catalog storage rejects path-like basenames and partial server context."""

    storage = LocalUploadStorage(str(tmp_path))
    staged = _stage(storage, tenant="tenant-a", filename="safe.txt", content=b"content")
    with pytest.raises(UploadPolicyError) as unsafe:
        storage.promote(
            staged,
            department_id="finance",
            doc_id="doc-a",
            original_filename="../unsafe.txt",
        )
    assert unsafe.value.code == "upload_filename_invalid"

    staged = _stage(storage, tenant="tenant-a", filename="safe.txt", content=b"content")
    with pytest.raises(UploadPolicyError) as incomplete:
        storage.promote(staged, department_id="finance")
    assert incomplete.value.code == "upload_path_invalid"


def test_nested_reference_rejects_traversal_and_cross_tenant_access(tmp_path: Path) -> None:
    """Accepted references remain opaque to other tenants and traversal attempts."""

    storage = LocalUploadStorage(str(tmp_path))
    staged = _stage(storage, tenant="tenant-a", filename="制度.txt", content=b"content")
    accepted = storage.promote(
        staged,
        department_id="finance",
        doc_id="doc-a",
        original_filename="制度.txt",
    )
    reference = storage.accepted_reference(accepted, "tenant-a")

    with pytest.raises(UploadPolicyError):
        storage.resolve_accepted("../" + reference, "tenant-a")
    with pytest.raises(UploadPolicyError):
        storage.resolve_accepted(reference, "tenant-b")

    assert storage.delete(accepted, tenant_id="tenant-a") is True
    assert not Path(accepted).exists()
