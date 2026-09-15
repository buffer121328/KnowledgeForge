"""Tenant-owned local upload storage, validation, and lifecycle policy."""

from __future__ import annotations

import hashlib
import os
import shutil
import threading
import time
import uuid
import zipfile
import re
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import BinaryIO, Callable

from domain.documents import normalize_uploaded_filename

try:
    import fcntl
except ImportError:  # pragma: no cover - production and supported dev hosts are POSIX
    fcntl = None

_FILE_MODE = 0o600
_DIR_MODE = 0o700
_EICAR_MARKER = (
    b"X5O!P%@AP[4\\PZX54(P^)7CC)7}$"
    b"EICAR-STANDARD-ANTIVIRUS-TEST-FILE!$H+H*"
)
_OLE_SIGNATURE = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"
_ACTIVE_ARCHIVE_SUFFIXES = (
    "vbaproject.bin",
    ".exe",
    ".dll",
    ".com",
    ".js",
    ".vbs",
    ".ps1",
    ".sh",
)


class UploadPolicyError(ValueError):
    """Stable, content-free upload rejection."""

    def __init__(self, code: str, message: str, *, status_code: int = 400) -> None:
        """Initialize the upload policy error."""
        super().__init__(message)
        self.code = code
        self.status_code = status_code
        self.message = message


@dataclass(frozen=True)
class StagedUpload:
    """Represent staged upload."""
    path: str
    tenant_id: str
    tenant_key: str
    extension: str
    size_bytes: int


@dataclass(frozen=True)
class UploadInspection:
    """Represent upload inspection."""
    parser_type: str
    extension: str
    size_bytes: int


class LocalUploadStorage:
    """Stores uploads beneath server-derived tenant lifecycle directories."""

    _lock = threading.RLock()

    def __init__(self, root: str) -> None:
        """Initialize the local upload storage."""
        self.root = Path(root).resolve()

    @staticmethod
    def _tenant_key(tenant_id: str | None) -> str:
        """Build a filesystem-safe storage key for a non-empty tenant."""
        normalized = (tenant_id or "").strip()
        if not normalized:
            raise UploadPolicyError("upload_tenant_required", "上传需要有效租户", status_code=400)
        return hashlib.sha256(normalized.encode()).hexdigest()[:32]

    def _tenant_root(self, tenant_id: str) -> Path:
        """Return the root upload directory for a tenant."""
        return self.root / "tenants" / self._tenant_key(tenant_id)

    @contextmanager
    def _quota_guard(self, tenant_id: str):
        """Serialize tenant quota checks with a filesystem lock."""
        tenant_root = self._tenant_root(tenant_id)
        tenant_root.mkdir(parents=True, exist_ok=True, mode=_DIR_MODE)
        lock_path = tenant_root / ".quota.lock"
        descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT, _FILE_MODE)
        try:
            if fcntl is not None:
                fcntl.flock(descriptor, fcntl.LOCK_EX)
            yield
        finally:
            if fcntl is not None:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)

    def _directory(self, tenant_id: str, state: str) -> Path:
        """Create and return a tenant upload lifecycle directory."""
        if state not in {"staging", "accepted", "quarantine"}:
            raise ValueError("unknown upload lifecycle state")
        directory = self._tenant_root(tenant_id) / state
        directory.mkdir(parents=True, exist_ok=True, mode=_DIR_MODE)
        for protected_directory in (
            self.root / "tenants",
            directory.parent,
            directory,
        ):
            try:
                os.chmod(protected_directory, _DIR_MODE)
            except OSError:
                pass
        return directory.resolve()

    @staticmethod
    def _extension(filename: str) -> str:
        """Return a normalized, filesystem-safe filename extension."""
        extension = Path(Path(filename).name).suffix.lower()
        if len(extension) > 12 or any(character not in ".abcdefghijklmnopqrstuvwxyz0123456789" for character in extension):
            return ""
        return extension

    @staticmethod
    def _contains(parent: Path, candidate: Path) -> bool:
        """Report whether a candidate path is contained by its parent."""
        resolved_parent = parent.resolve()
        resolved_candidate = candidate.resolve()
        return resolved_candidate != resolved_parent and resolved_parent in resolved_candidate.parents

    @staticmethod
    def _server_segment(value: str, *, field: str) -> str:
        """Return a stable filesystem segment for a server-owned identifier."""

        normalized = (value or "").strip()
        if not normalized:
            raise UploadPolicyError("upload_path_invalid", f"{field} 不能为空")
        if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", normalized):
            return normalized
        return hashlib.sha256(normalized.encode()).hexdigest()[:32]

    def _usage(self, tenant_id: str) -> int:
        """Calculate the tenant storage used by active uploads, including nested files."""
        total = 0
        for state in ("staging", "accepted"):
            directory = self._directory(tenant_id, state)
            for candidate in directory.rglob("*"):
                if candidate.is_file() and not candidate.is_symlink():
                    try:
                        total += candidate.stat().st_size
                    except OSError:
                        continue
        return total

    def stage(
        self,
        *,
        tenant_id: str,
        original_filename: str,
        source: BinaryIO,
        max_file_bytes: int,
        tenant_quota_bytes: int,
        chunk_bytes: int = 1024 * 1024,
    ) -> StagedUpload:
        """Stage the local upload storage."""
        if max_file_bytes <= 0 or tenant_quota_bytes <= 0 or chunk_bytes <= 0:
            raise ValueError("upload byte limits must be positive")
        extension = self._extension(original_filename)
        tenant_key = self._tenant_key(tenant_id)
        staging = self._directory(tenant_id, "staging")
        path = staging / f"{uuid.uuid4().hex}{extension}.part"
        size = 0

        with self._lock, self._quota_guard(tenant_id):
            baseline = self._usage(tenant_id)
            try:
                descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, _FILE_MODE)
                with os.fdopen(descriptor, "wb") as target:
                    while True:
                        chunk = source.read(chunk_bytes)
                        if not chunk:
                            break
                        size += len(chunk)
                        if size > max_file_bytes:
                            raise UploadPolicyError(
                                "upload_too_large",
                                "上传文件超过大小上限",
                                status_code=413,
                            )
                        if baseline + size > tenant_quota_bytes:
                            raise UploadPolicyError(
                                "tenant_upload_quota_exceeded",
                                "租户上传空间不足",
                                status_code=413,
                            )
                        target.write(chunk)
                    target.flush()
                    os.fsync(target.fileno())
            except Exception:
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    pass
                raise

        return StagedUpload(
            path=str(path),
            tenant_id=tenant_id,
            tenant_key=tenant_key,
            extension=extension,
            size_bytes=size,
        )

    def promote(
        self,
        staged: StagedUpload,
        *,
        department_id: str | None = None,
        doc_id: str | None = None,
        original_filename: str | None = None,
    ) -> str:
        """Promote a staged file to legacy-flat or catalog-owned accepted storage."""

        staging = self._directory(staged.tenant_id, "staging")
        source = Path(staged.path).resolve()
        if staged.tenant_key != self._tenant_key(staged.tenant_id) or not self._contains(staging, source):
            raise UploadPolicyError("upload_path_invalid", "上传暂存路径无效")
        if not source.is_file():
            raise UploadPolicyError("upload_path_invalid", "上传暂存文件不存在")
        accepted = self._directory(staged.tenant_id, "accepted")
        catalog_values = (department_id, doc_id, original_filename)
        if any(value is not None for value in catalog_values):
            if not all(value is not None for value in catalog_values):
                raise UploadPolicyError("upload_path_invalid", "目录化存储上下文不完整")
            try:
                basename = normalize_uploaded_filename(original_filename or "")
            except ValueError as error:
                raise UploadPolicyError("upload_filename_invalid", "上传文件名无效") from error
            department_segment = self._server_segment(department_id or "", field="department_id")
            document_segment = self._server_segment(doc_id or "", field="doc_id")
            destination_directory = (
                accepted
                / "departments"
                / department_segment
                / "documents"
                / document_segment
            )
            destination_directory.mkdir(parents=True, exist_ok=True, mode=_DIR_MODE)
            destination = destination_directory / basename
            if destination.exists():
                raise UploadPolicyError("upload_path_conflict", "文档存储位置已存在", status_code=409)
        else:
            destination = accepted / f"{uuid.uuid4().hex}{staged.extension}"
        os.replace(source, destination)
        os.chmod(destination, _FILE_MODE)
        return str(destination.resolve())

    def accepted_reference(self, path: str | Path, tenant_id: str) -> str:
        """Return a tenant-relative opaque reference for an accepted upload."""
        accepted = self._directory(tenant_id, "accepted")
        candidate = Path(path)
        if candidate.is_symlink():
            raise UploadPolicyError("upload_path_invalid", "上传文件引用无效", status_code=404)
        resolved = candidate.resolve()
        if not self._contains(accepted, resolved) or not resolved.is_file():
            raise UploadPolicyError("upload_path_invalid", "上传文件引用无效", status_code=404)
        return resolved.relative_to(accepted).as_posix()

    def resolve_accepted(self, reference: str, tenant_id: str) -> str:
        """Resolve a tenant-relative accepted-upload reference without traversal."""
        normalized = (reference or "").strip().replace("\\", "/")
        token = PurePosixPath(normalized)
        if (
            not normalized
            or token.is_absolute()
            or any(part in {"", ".", ".."} for part in token.parts)
        ):
            raise UploadPolicyError("upload_path_invalid", "上传文件引用无效", status_code=404)
        accepted = self._directory(tenant_id, "accepted")
        candidate = accepted.joinpath(*token.parts)
        if candidate.is_symlink():
            raise UploadPolicyError("upload_path_invalid", "上传文件引用无效", status_code=404)
        resolved = candidate.resolve()
        if not self._contains(accepted, resolved) or not resolved.is_file():
            raise UploadPolicyError("upload_path_invalid", "上传文件引用无效", status_code=404)
        return str(resolved)

    def migrate_legacy_file(self, source_path: str | Path, tenant_id: str) -> str:
        """Atomically move an explicitly-owned shared-root file into tenant storage."""
        source = Path(source_path).resolve()
        if not source.is_file() or source.parent != self.root:
            raise UploadPolicyError("upload_legacy_path_invalid", "旧版上传路径无效")
        accepted = self._directory(tenant_id, "accepted")
        extension = self._extension(source.name)
        destination = accepted / f"{uuid.uuid4().hex}{extension}"
        os.replace(source, destination)
        os.chmod(destination, _FILE_MODE)
        return str(destination.resolve())

    def quarantine(self, staged: StagedUpload) -> str:
        """Quarantine the local upload storage."""
        staging = self._directory(staged.tenant_id, "staging")
        source = Path(staged.path).resolve()
        if not self._contains(staging, source) or not source.is_file():
            raise UploadPolicyError("upload_path_invalid", "上传暂存路径无效")
        quarantine = self._directory(staged.tenant_id, "quarantine")
        destination = quarantine / f"{uuid.uuid4().hex}.quarantine"
        os.replace(source, destination)
        os.chmod(destination, _FILE_MODE)
        return str(destination.resolve())

    def discard(self, staged_or_path: StagedUpload | str) -> bool:
        """Discard the local upload storage."""
        if isinstance(staged_or_path, StagedUpload):
            tenant_id = staged_or_path.tenant_id
            candidate = Path(staged_or_path.path).resolve()
            allowed = self._directory(tenant_id, "staging")
        else:
            candidate = Path(staged_or_path).resolve()
            allowed = self.root
        if not self._contains(allowed, candidate) or not candidate.is_file():
            return False
        try:
            candidate.unlink()
        except OSError:
            return False
        return True

    def cleanup(self, tenant_id: str, *, older_than_seconds: int, now: float | None = None) -> int:
        """Clean up the local upload storage."""
        if older_than_seconds < 0:
            raise ValueError("older_than_seconds must not be negative")
        cutoff = (time.time() if now is None else now) - older_than_seconds
        removed = 0
        for state in ("staging", "quarantine"):
            directory = self._directory(tenant_id, state)
            for candidate in directory.iterdir():
                if not candidate.is_file():
                    continue
                try:
                    if candidate.stat().st_mtime <= cutoff:
                        candidate.unlink()
                        removed += 1
                except OSError:
                    continue
        return removed

    def cleanup_all(
        self,
        *,
        staging_older_than_seconds: int,
        quarantine_older_than_seconds: int,
        now: float | None = None,
    ) -> dict[str, int]:
        """Remove only expired lifecycle files beneath validated tenant-key dirs."""

        if staging_older_than_seconds < 0 or quarantine_older_than_seconds < 0:
            raise ValueError("cleanup ages must not be negative")
        reference = time.time() if now is None else now
        counts = {"staging": 0, "quarantine": 0}
        tenants = self.root / "tenants"
        if not tenants.is_dir():
            return counts
        for tenant_root in tenants.iterdir():
            if not tenant_root.is_dir() or not re.fullmatch(r"[0-9a-f]{32}", tenant_root.name):
                continue
            for state, age in (
                ("staging", staging_older_than_seconds),
                ("quarantine", quarantine_older_than_seconds),
            ):
                directory = (tenant_root / state).resolve()
                if not self._contains(tenant_root, directory) or not directory.is_dir():
                    continue
                cutoff = reference - age
                for candidate in directory.iterdir():
                    if not candidate.is_file():
                        continue
                    try:
                        if candidate.stat().st_mtime <= cutoff:
                            candidate.unlink()
                            counts[state] += 1
                    except OSError:
                        continue
        return counts

    def delete(self, path: str, tenant_id: str | None = None) -> bool:
        """Delete a record through the local upload storage."""
        if not tenant_id:
            return False
        accepted = self._directory(tenant_id, "accepted")
        candidate = Path(path).resolve()
        if not self._contains(accepted, candidate) or not candidate.is_file():
            return False
        with self._lock, self._quota_guard(tenant_id):
            try:
                candidate.unlink()
                parent = candidate.parent
                while parent != accepted and self._contains(accepted, parent):
                    try:
                        parent.rmdir()
                    except OSError:
                        break
                    parent = parent.parent
            except OSError:
                return False
        return True

    def save(self, filename: str, source: BinaryIO) -> str:
        """Legacy shared-root write retained for non-HTTP migration callers."""

        self.root.mkdir(parents=True, exist_ok=True)
        destination = self.root / Path(filename).name
        with destination.open("wb") as target:
            shutil.copyfileobj(source, target)
        return str(destination.resolve())


class UploadValidationPolicy:
    """Validate parser compatibility and bounded content before ingestion."""

    _MIME_TYPES: dict[str, tuple[str, set[str]]] = {
        ".pdf": ("pdf", {"application/pdf"}),
        ".png": ("image", {"image/png"}),
        ".jpg": ("image", {"image/jpeg"}),
        ".jpeg": ("image", {"image/jpeg"}),
        ".tiff": ("image", {"image/tiff"}),
        ".bmp": ("image", {"image/bmp", "image/x-ms-bmp"}),
        ".docx": (
            "word",
            {"application/vnd.openxmlformats-officedocument.wordprocessingml.document"},
        ),
        ".xlsx": (
            "table",
            {"application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"},
        ),
        ".csv": ("table", {"text/csv", "application/csv", "text/plain", "application/vnd.ms-excel"}),
        ".txt": ("text", {"text/plain"}),
        ".md": ("markdown", {"text/markdown", "text/plain"}),
        ".html": ("text", {"text/html"}),
        ".htm": ("text", {"text/html"}),
    }

    def __init__(
        self,
        *,
        max_archive_entries: int,
        max_archive_uncompressed_bytes: int,
        max_archive_compression_ratio: float,
        max_pdf_pages: int,
        max_image_pixels: int,
        max_spreadsheet_rows: int,
        max_text_characters: int,
        require_external_scanner: bool = False,
        external_scanner: Callable[[Path], bool] | None = None,
    ) -> None:
        """Initialize the upload validation policy."""
        self.max_archive_entries = max_archive_entries
        self.max_archive_uncompressed_bytes = max_archive_uncompressed_bytes
        self.max_archive_compression_ratio = max_archive_compression_ratio
        self.max_pdf_pages = max_pdf_pages
        self.max_image_pixels = max_image_pixels
        self.max_spreadsheet_rows = max_spreadsheet_rows
        self.max_text_characters = max_text_characters
        self.require_external_scanner = require_external_scanner
        self.external_scanner = external_scanner

    @staticmethod
    def _safe_error(code: str, message: str, *, status_code: int = 400) -> UploadPolicyError:
        """Create a client-safe upload policy error."""
        return UploadPolicyError(code, message, status_code=status_code)

    @staticmethod
    def _contains_marker(path: Path, marker: bytes) -> bool:
        """Report whether scanned content contains a blocked marker."""
        carry = b""
        with path.open("rb") as source:
            while chunk := source.read(1024 * 1024):
                combined = carry + chunk
                if marker in combined:
                    return True
                carry = combined[-max(len(marker) - 1, 0):]
        return False

    def _scan(self, path: Path) -> None:
        """Scan the upload validation policy."""
        if self._contains_marker(path, _EICAR_MARKER):
            raise self._safe_error("upload_malicious", "上传内容被安全策略拒绝")
        if self.require_external_scanner and self.external_scanner is None:
            raise self._safe_error(
                "upload_scanner_unavailable",
                "上传扫描服务暂不可用",
                status_code=503,
            )
        if self.external_scanner is not None:
            try:
                allowed = self.external_scanner(path)
            except Exception as error:
                raise self._safe_error(
                    "upload_scanner_unavailable",
                    "上传扫描服务暂不可用",
                    status_code=503,
                ) from error
            if not allowed:
                raise self._safe_error("upload_malicious", "上传内容被安全策略拒绝")

    def _inspect_zip(self, path: Path, extension: str) -> set[str]:
        """Inspect the zip."""
        try:
            archive = zipfile.ZipFile(path)
        except (OSError, zipfile.BadZipFile) as error:
            raise self._safe_error("upload_type_mismatch", "文件内容与类型不匹配") from error

        with archive:
            infos = archive.infolist()
            if len(infos) > self.max_archive_entries:
                raise self._safe_error("upload_archive_unsafe", "压缩容器超过安全限制")
            total_size = 0
            total_compressed = 0
            names: set[str] = set()
            for info in infos:
                name = info.filename.replace("\\", "/")
                normalized = PurePosixPath(name)
                lowered = name.lower()
                if normalized.is_absolute() or ".." in normalized.parts or info.flag_bits & 0x1:
                    raise self._safe_error("upload_archive_unsafe", "压缩容器违反安全策略")
                if lowered.endswith(_ACTIVE_ARCHIVE_SUFFIXES) or "/embeddings/" in f"/{lowered}":
                    raise self._safe_error("upload_malicious", "上传内容被安全策略拒绝")
                total_size += info.file_size
                total_compressed += max(info.compress_size, 1)
                if total_size > self.max_archive_uncompressed_bytes:
                    raise self._safe_error("upload_archive_unsafe", "压缩容器超过安全限制")
                if info.file_size / max(info.compress_size, 1) > self.max_archive_compression_ratio:
                    raise self._safe_error("upload_archive_unsafe", "压缩容器超过安全限制")
                if info.file_size:
                    carry = b""
                    with archive.open(info) as member:
                        while chunk := member.read(1024 * 1024):
                            combined = carry + chunk
                            if _EICAR_MARKER in combined:
                                raise self._safe_error(
                                    "upload_malicious",
                                    "上传内容被安全策略拒绝",
                                )
                            carry = combined[-(len(_EICAR_MARKER) - 1):]
                names.add(name)
            if total_size / max(total_compressed, 1) > self.max_archive_compression_ratio:
                raise self._safe_error("upload_archive_unsafe", "压缩容器超过安全限制")

            if extension == ".docx":
                if "[Content_Types].xml" not in names or "word/document.xml" not in names:
                    raise self._safe_error("upload_type_mismatch", "文件内容与类型不匹配")
            elif extension == ".xlsx":
                if "[Content_Types].xml" not in names or "xl/workbook.xml" not in names:
                    raise self._safe_error("upload_type_mismatch", "文件内容与类型不匹配")
                rows = 0
                for info in infos:
                    if not info.filename.startswith("xl/worksheets/") or not info.filename.endswith(".xml"):
                        continue
                    carry = b""
                    with archive.open(info) as source:
                        while chunk := source.read(1024 * 1024):
                            combined = carry + chunk
                            rows += combined.count(b"<row")
                            if rows > self.max_spreadsheet_rows:
                                raise self._safe_error(
                                    "upload_work_limit_exceeded",
                                    "上传文件超过解析工作量上限",
                                )
                            carry = combined[-3:]
            return names

    def _validate_pdf(self, path: Path) -> None:
        """Validate the PDF."""
        with path.open("rb") as source:
            if not source.read(5).startswith(b"%PDF-"):
                raise self._safe_error("upload_type_mismatch", "文件内容与类型不匹配")
        try:
            from pypdf import PdfReader

            pages = len(PdfReader(str(path)).pages)
        except Exception as error:
            raise self._safe_error("upload_type_mismatch", "PDF 结构无效") from error
        if pages > self.max_pdf_pages:
            raise self._safe_error("upload_work_limit_exceeded", "上传文件超过解析工作量上限")

    def _validate_image(self, path: Path, extension: str) -> None:
        """Validate the image."""
        signatures = {
            ".png": (b"\x89PNG\r\n\x1a\n",),
            ".jpg": (b"\xff\xd8\xff",),
            ".jpeg": (b"\xff\xd8\xff",),
            ".tiff": (b"II*\x00", b"MM\x00*"),
            ".bmp": (b"BM",),
        }
        with path.open("rb") as source:
            head = source.read(16)
        if not any(head.startswith(signature) for signature in signatures[extension]):
            raise self._safe_error("upload_type_mismatch", "文件内容与类型不匹配")
        try:
            from PIL import Image

            with Image.open(path) as image:
                width, height = image.size
                image.verify()
        except Exception as error:
            raise self._safe_error("upload_type_mismatch", "图片结构无效") from error
        if width * height > self.max_image_pixels:
            raise self._safe_error("upload_work_limit_exceeded", "上传文件超过解析工作量上限")

    def _validate_text(self, path: Path, extension: str) -> None:
        """Validate the text."""
        try:
            text = path.read_text(encoding="utf-8-sig")
        except (OSError, UnicodeDecodeError) as error:
            raise self._safe_error("upload_type_mismatch", "文本编码或内容无效") from error
        if "\x00" in text or any(ord(character) < 8 for character in text):
            raise self._safe_error("upload_type_mismatch", "文本编码或内容无效")
        if len(text) > self.max_text_characters:
            raise self._safe_error("upload_work_limit_exceeded", "上传文件超过解析工作量上限")
        if extension == ".csv" and text.count("\n") > self.max_spreadsheet_rows:
            raise self._safe_error("upload_work_limit_exceeded", "上传文件超过解析工作量上限")
        if extension in {".html", ".htm"}:
            lowered = text.lower()
            if any(token in lowered for token in ("<script", "<iframe", "<object", "<embed", "javascript:")):
                raise self._safe_error("upload_malicious", "上传内容被安全策略拒绝")

    def validate(self, path: str | Path, original_filename: str, content_type: str | None) -> UploadInspection:
        """Validate a value with the upload validation policy."""
        candidate = Path(path).resolve()
        extension = Path(Path(original_filename).name).suffix.lower()
        policy = self._MIME_TYPES.get(extension)
        if policy is None:
            raise self._safe_error("upload_type_unsupported", "不支持的文件类型")
        parser_type, allowed_mimes = policy
        declared_mime = (content_type or "").split(";", 1)[0].strip().lower()
        generic = declared_mime in {"", "application/octet-stream"}
        strong_signature = extension in {".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".docx", ".xlsx"}
        if declared_mime not in allowed_mimes and not (generic and strong_signature):
            raise self._safe_error("upload_type_mismatch", "文件声明类型与扩展名不匹配")

        self._scan(candidate)
        if extension == ".pdf":
            self._validate_pdf(candidate)
        elif extension in {".png", ".jpg", ".jpeg", ".tiff", ".bmp"}:
            self._validate_image(candidate, extension)
        elif extension in {".docx", ".xlsx"}:
            self._inspect_zip(candidate, extension)
        else:
            self._validate_text(candidate, extension)
        return UploadInspection(
            parser_type=parser_type,
            extension=extension,
            size_bytes=candidate.stat().st_size,
        )
