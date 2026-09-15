"""ATDD coverage for folder manifest validation and company-demo mapping."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest
from api.dependencies import DocumentReadProvider
from api.routers.documents_ingest import _ensure_manifest_departments, _validate_folder_manifest
from api.schemas import FolderManifestFileRequest, FolderUploadManifestRequest
from domain.documents import DepartmentRecord
from fastapi import HTTPException
from infrastructure.documents.catalog import SQLiteDocumentCatalogRepository

COMPANY_DEMO_NORMALIZED = (
    Path(__file__).resolve().parents[3] / "evaluation" / "data" / "company-demo" / "normalized"
)
DEPARTMENT_MAPPINGS = {
    "human_resources": "human_resources",
    "finance": "finance",
    "procurement_warehouse": "procurement_warehouse",
    "administration": "administration",
}


def _manifest(paths: list[str], *, company_id: str = "tenant-a") -> FolderUploadManifestRequest:
    """Build a valid ordered folder manifest for the supplied relative paths."""

    return FolderUploadManifestRequest(
        root_folder_name="normalized",
        company_id=company_id,
        department_mappings=DEPARTMENT_MAPPINGS,
        files=[
            FolderManifestFileRequest(
                client_file_id=str(index),
                relative_path=relative_path,
                department_id=DEPARTMENT_MAPPINGS[relative_path.split("/", 1)[0]],
            )
            for index, relative_path in enumerate(paths)
        ],
    )


def _files(paths: list[str]):
    """Build lightweight upload objects exposing browser basename semantics."""

    return [SimpleNamespace(filename=Path(relative_path).name) for relative_path in paths]


def _code(error: HTTPException) -> str:
    """Extract one stable structured API error code."""

    assert isinstance(error.detail, dict)
    return str(error.detail["code"])


def test_company_demo_normalized_folder_maps_41_files_to_four_departments() -> None:
    """The real demo normalized directory yields the deterministic 21/5/5/10 preview."""

    paths = sorted(
        path.relative_to(COMPANY_DEMO_NORMALIZED).as_posix()
        for path in COMPANY_DEMO_NORMALIZED.rglob("*")
        if path.is_file()
    )
    manifest = _manifest(paths)

    validated = _validate_folder_manifest(manifest, _files(paths), tenant_id="tenant-a")

    counts = Counter(entry.department_id for _, entry, _ in validated)
    assert len(validated) == 41
    assert counts == {
        "human_resources": 21,
        "finance": 5,
        "procurement_warehouse": 5,
        "administration": 10,
    }


def test_manifest_creates_known_departments_with_canonical_names(tmp_path: Path) -> None:
    catalog = SQLiteDocumentCatalogRepository(tmp_path / "catalog.sqlite3")

    provider = DocumentReadProvider(
        catalog=catalog,
        vector_store=None,
        storage=SimpleNamespace(),
    )

    _ensure_manifest_departments(
        provider,
        _manifest(["administration/a.txt"]),
        tenant_id="tenant-a",
    )

    department = catalog.get_department("administration", tenant_id="tenant-a")
    assert department is not None
    assert department.name == "行政管理部"


def test_manifest_count_mismatch_rejects_before_staging() -> None:
    """Multipart files and manifest entries must have one-to-one cardinality."""

    manifest = _manifest(["finance/a.txt"])
    with pytest.raises(HTTPException) as error:
        _validate_folder_manifest(manifest, [], tenant_id="tenant-a")
    assert _code(error.value) == "upload_manifest_mismatch"


@pytest.mark.parametrize(
    ("path", "expected_code"),
    [
        ("finance/../secret.txt", "upload_relative_path_invalid"),
        ("/absolute.txt", "upload_relative_path_invalid"),
    ],
)
def test_manifest_rejects_unsafe_paths(path: str, expected_code: str) -> None:
    """Unsafe logical paths fail before any storage operation."""

    manifest = FolderUploadManifestRequest(
        root_folder_name="normalized",
        company_id="tenant-a",
        department_mappings={"finance": "finance"},
        files=[
            FolderManifestFileRequest(
                client_file_id="0",
                relative_path=path,
                department_id="finance",
            )
        ],
    )
    with pytest.raises(HTTPException) as error:
        _validate_folder_manifest(manifest, _files([path]), tenant_id="tenant-a")
    assert _code(error.value) == expected_code


def test_manifest_rejects_duplicate_normalized_paths() -> None:
    """Two multipart members cannot target one logical catalog path."""

    paths = ["finance/a.txt", "finance/a.txt"]
    manifest = _manifest(paths)
    with pytest.raises(HTTPException) as error:
        _validate_folder_manifest(manifest, _files(paths), tenant_id="tenant-a")
    assert _code(error.value) == "upload_duplicate_path"


@pytest.mark.parametrize(
    "multipart_filename",
    [
        "finance/资金管理制度.docx",
        "company-root/finance/资金管理制度.docx",
    ],
)
def test_manifest_accepts_browser_folder_relative_multipart_filenames(
    multipart_filename: str,
) -> None:
    """Chromium folder uploads may send the selected relative path as filename."""

    manifest = _manifest(["finance/资金管理制度.docx"])
    manifest.root_folder_name = (
        "finance" if multipart_filename.startswith("finance/") else "company-root"
    )

    validated = _validate_folder_manifest(
        manifest,
        [SimpleNamespace(filename=multipart_filename)],
        tenant_id="tenant-a",
    )

    assert validated[0][2] == "finance/资金管理制度.docx"


def test_manifest_reports_invalid_multipart_filename_separately() -> None:
    """A malformed multipart filename is not mislabeled as a logical-path error."""

    manifest = _manifest(["finance/资金管理制度.docx"])
    files = [SimpleNamespace(filename="../资金管理制度.docx")]

    with pytest.raises(HTTPException) as error:
        _validate_folder_manifest(manifest, files, tenant_id="tenant-a")

    assert _code(error.value) == "upload_filename_invalid"


def test_manifest_rejects_multipart_path_that_disagrees_with_manifest() -> None:
    """A safe-looking browser path cannot be rebound to another manifest member."""

    manifest = _manifest(["finance/资金管理制度.docx"])

    with pytest.raises(HTTPException) as error:
        _validate_folder_manifest(
            manifest,
            [SimpleNamespace(filename="finance/其他制度.docx")],
            tenant_id="tenant-a",
        )

    assert _code(error.value) == "upload_filename_invalid"


def test_manifest_rejects_company_and_department_outside_scope(tmp_path: Path) -> None:
    """Tenant and company ownership are checked before any file is staged."""

    with pytest.raises(HTTPException) as company_error:
        _validate_folder_manifest(
            _manifest(["finance/a.txt"], company_id="tenant-b"),
            _files(["finance/a.txt"]),
            tenant_id="tenant-a",
        )
    assert _code(company_error.value) == "upload_company_forbidden"

    catalog = SQLiteDocumentCatalogRepository(tmp_path / "catalog.sqlite3")
    catalog.upsert_department(
        DepartmentRecord(
            department_id="finance",
            tenant_id="tenant-a",
            company_id="other-company",
            name="财务部",
            normalized_key="finance",
        )
    )
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(document_catalog=catalog)))
    with pytest.raises(HTTPException) as department_error:
        _ensure_manifest_departments(
            request,
            _manifest(["finance/a.txt"]),
            tenant_id="tenant-a",
        )
    assert _code(department_error.value) == "upload_department_forbidden"


@pytest.mark.asyncio
async def test_folder_member_success_persists_catalog_context_and_original_names(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A valid member uses a stable ID and writes complete catalog metadata."""

    from io import BytesIO

    from api.routers import documents_ingest as documents_ingest_mod
    from domain.identity import Permission, UserContext, UserRole
    from starlette.datastructures import Headers, UploadFile

    catalog = SQLiteDocumentCatalogRepository(tmp_path / "catalog.sqlite3")
    catalog.upsert_department(
        DepartmentRecord(
            department_id="finance",
            tenant_id="tenant-a",
            company_id="tenant-a",
            name="财务部",
            normalized_key="finance",
        )
    )

    class Workflow:
        """Capture one typed workflow invocation and return deterministic counts."""

        invocation = None

        async def ainvoke(self, payload):
            self.invocation = payload
            item = payload["document_inputs"][0]
            chunk = SimpleNamespace(doc_id=item.doc_id, doc_type=SimpleNamespace(value="text"))
            return {"chunks": [chunk], "entities_stored": 2, "relations_stored": 1}

    class Audit:
        """Accept audit writes for the isolated route test."""

        def log(self, **_kwargs) -> None:
            return None

    class Webhooks:
        """Accept webhook dispatch for the isolated route test."""

        async def trigger(self, *_args, **_kwargs) -> None:
            return None

    workflow = Workflow()
    request = SimpleNamespace(
        state=SimpleNamespace(tenant_id="tenant-a"),
        app=SimpleNamespace(
            state=SimpleNamespace(
                document_catalog=catalog,
                workflows={"ingest": workflow},
            )
        ),
    )
    monkeypatch.setattr(documents_ingest_mod.settings, "upload_dir", str(tmp_path / "uploads"))
    user = UserContext(
        user_id="user-a",
        username="alice",
        role=UserRole.EDITOR,
        org_id="tenant-a",
        permissions=[Permission.DOC_WRITE],
    )
    manifest = FolderUploadManifestRequest(
        root_folder_name="normalized",
        company_id="tenant-a",
        department_mappings={"finance": "finance"},
        files=[],
    )
    entry = FolderManifestFileRequest(
        client_file_id="0",
        relative_path="finance/资金管理制度.txt",
        department_id="finance",
        display_name="资金管理制度",
        provenance_source_filename="资金管理制度.docx",
        metadata={
            "authority": "formal_candidate",
            "status": "needs_review",
            "sensitivity": "internal_demo",
            "source_format": "txt",
        },
    )
    upload = UploadFile(
        file=BytesIO("资金制度".encode()),
        filename="资金管理制度.txt",
        headers=Headers({"content-type": "text/plain"}),
    )

    from api.dependencies import get_document_submission

    submission = get_document_submission(request)
    prepared = await documents_ingest_mod._prepare_folder_member(
        request,
        upload,
        entry,
        entry.relative_path,
        manifest,
        user,
        submission,
    )

    assert isinstance(prepared, documents_ingest_mod.PreparedFolderMember)
    assert prepared.response.status == "accepted"
    assert prepared.response.chunks_count == 0
    assert prepared.response.entities_count == 0
    assert prepared.response.relations_count == 0
    record = catalog.get_document(prepared.record.doc_id, tenant_id="tenant-a")
    assert record is not None
    assert record.ingest_status == "accepted"
    assert record.uploaded_filename == "资金管理制度.txt"
    assert record.provenance_source_filename == "资金管理制度.docx"
    assert record.relative_path == "finance/资金管理制度.txt"
    assert record.department_id == "finance"
    assert Path(record.storage_reference).is_file()
    assert prepared.document_input.doc_id == record.doc_id
    assert workflow.invocation is None
