"""从一个本地 DOCX 来源集合准备一个新的 company-demo 部门。"""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from xml.etree import ElementTree
from zipfile import BadZipFile, ZipFile

from evaluation.company_demo.text_cleaning import CleaningValidationError, clean_company_demo

# 需要计数的图片/图形相关 XML 标签局部名
_IMAGE_TAGS = frozenset({"drawing", "pict", "object", "shape", "imagedata", "group"})


class CompanyDemoExpansionError(ValueError):
    """当本地部门扩充无法安全复现时抛出。"""


@dataclass(frozen=True)
class DocxObjectInspection:
    """描述单个 DOCX 包中检出的非文本对象计数。"""

    media_count: int  # word/media/ 下的媒体文件数
    drawing_count: int  # drawing 标签元素数
    pict_count: int  # pict 标签元素数
    object_count: int  # object 标签元素数
    shape_count: int  # shape 标签元素数
    image_data_count: int  # imagedata 标签元素数
    group_count: int  # group 标签元素数
    embedded_object_count: int  # word/embeddings/ 下的嵌入对象数

    @property
    def object_count_total(self) -> int:
        """返回各类非文本对象计数之和，作为跳过契约的总数。"""
        return (
            self.media_count
            + self.drawing_count
            + self.pict_count
            + self.object_count
            + self.shape_count
            + self.image_data_count
            + self.group_count
            + self.embedded_object_count
        )

    @property
    def has_image_content(self) -> bool:
        """返回该 DOCX 是否因含非文本内容而必须在本次扩充中跳过。"""
        return self.object_count_total > 0


@dataclass(frozen=True)
class ExpansionCandidateRecord:
    """记录一个来源候选的检查与准备结果。"""

    source_filename: str  # 来源 DOCX 文件名
    source_document_id: str  # 来源文档 ID
    status: str  # 处理结果状态（included / skipped_image_content）
    reason_code: str | None  # 跳过原因码；成功收录时为 None
    media_count: int  # 媒体文件数（含义同 DocxObjectInspection）
    drawing_count: int  # drawing 标签元素数
    pict_count: int  # pict 标签元素数
    object_count: int  # object 标签元素数
    shape_count: int  # shape 标签元素数
    image_data_count: int  # imagedata 标签元素数
    group_count: int  # group 标签元素数
    embedded_object_count: int  # 嵌入对象数
    repository_path: str | None  # 收录后在语料库中的相对路径；跳过时为 None
    normalized_path: str | None  # 归一化文本的相对路径；跳过时为 None


@dataclass(frozen=True)
class CompanyDemoExpansionReport:
    """汇总一次确定性的第四部门准备运行结果。"""

    department: str  # 本次准备的部门标识
    candidate_count: int  # 候选文档总数
    included_count: int  # 成功收录的文档数
    skipped_count: int  # 因含图片内容被跳过的文档数
    records: tuple[ExpansionCandidateRecord, ...]  # 每个候选的明细记录


def _sha256_file(path: Path) -> str:
    """分块读取文件并返回 SHA-256 十六进制摘要

    Args:
        path: 待计算摘要的文件路径。
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _required_string(record: dict[str, Any], field: str, *, context: str) -> str:
    """校验记录中指定字段为非空字符串，返回去除首尾空白后的值

    Args:
        record: 待提取字段的字典记录。
        field: 必填字段名。
        context: 出错信息中用于定位的上下文（如 selection.candidates[1]）。
    """
    value = record.get(field)
    if not isinstance(value, str) or not value.strip():
        raise CompanyDemoExpansionError(f"{context}.{field} must be a non-empty string")
    return value.strip()


def inspect_docx_objects(path: str | Path) -> DocxObjectInspection:
    """检查一个 DOCX 包内的图片、图形或嵌入对象并返回计数结果

    Args:
        path: 待检查的 DOCX 文件路径。
    """
    source = Path(path)
    try:
        with ZipFile(source) as archive:
            names = archive.namelist()
            # ① 统计 word/media/ 与 word/embeddings/ 下的实际文件条目（排除目录项）
            media_count = sum(
                name.startswith("word/media/") and not name.endswith("/") for name in names
            )
            embedded_count = sum(
                name.startswith("word/embeddings/") and not name.endswith("/")
                for name in names
            )
            counts = {tag: 0 for tag in _IMAGE_TAGS}
            # ② 解析 word/ 下每个 XML，统计图片/图形相关标签的元素出现次数
            for name in names:
                if not name.startswith("word/") or not name.endswith(".xml"):
                    continue
                try:
                    root = ElementTree.fromstring(archive.read(name))
                except ElementTree.ParseError as error:
                    raise CompanyDemoExpansionError(
                        f"invalid DOCX XML while inspecting {source.name}: {name}"
                    ) from error
                for element in root.iter():
                    # 去掉命名空间前缀后按局部名匹配目标标签
                    local_name = element.tag.rsplit("}", 1)[-1]
                    if local_name in counts:
                        counts[local_name] += 1
    except (BadZipFile, KeyError, OSError) as error:
        # 非法 ZIP 或读取失败统一转为领域错误，避免把底层异常泄漏给调用方
        raise CompanyDemoExpansionError(
            f"invalid DOCX package while inspecting {source.name}"
        ) from error

    return DocxObjectInspection(
        media_count=media_count,
        drawing_count=counts["drawing"],
        pict_count=counts["pict"],
        object_count=counts["object"],
        shape_count=counts["shape"],
        image_data_count=counts["imagedata"],
        group_count=counts["group"],
        embedded_object_count=embedded_count,
    )


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    """读取 JSON 文件并要求顶层为对象，否则抛出领域错误

    Args:
        path: JSON 文件路径。
        label: 错误信息中描述该文件的标签。
    """
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise CompanyDemoExpansionError(f"missing {label}: {path}") from error
    except json.JSONDecodeError as error:
        raise CompanyDemoExpansionError(f"invalid {label}: {error.msg}") from error
    if not isinstance(value, dict):
        raise CompanyDemoExpansionError(f"{label} must contain a JSON object")
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    """以排序键的格式化 JSON 原子写入目标文件

    Args:
        path: 目标 JSON 文件路径。
        value: 待写入的对象。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    # 先写 .tmp 临时文件再整体替换，避免中途失败留下半截清单
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def prepare_company_demo_department(
    source_root: str | Path,
    corpus_root: str | Path,
    *,
    selection_path: str | Path,
    report_path: str | Path | None = None,
) -> CompanyDemoExpansionReport:
    """检查、复制、归一化并登记一个选定的 DOCX 部门，返回准备报告

    Args:
        source_root: 本地 DOCX 来源目录。
        corpus_root: company-demo 语料库根目录。
        selection_path: 选定清单 JSON 路径。
        report_path: 准备报告输出路径；为 None 时写入语料库根目录的默认文件。
    """
    source_directory = Path(source_root).resolve()
    target = Path(corpus_root).resolve()
    selection = _load_json(Path(selection_path).resolve(), label="selection manifest")
    manifest_path = target / "corpus_manifest.json"
    manifest = _load_json(manifest_path, label="company-demo manifest")

    # ① 校验选定清单：schema 必须为 v1，且本次扩充只允许 administration 部门
    if selection.get("schema_version") != 1:
        raise CompanyDemoExpansionError("selection manifest schema_version must be 1")
    department = _required_string(selection, "department", context="selection")
    department_label = _required_string(
        selection, "department_label", context="selection"
    )
    if department != "administration":
        raise CompanyDemoExpansionError("this expansion requires administration")
    raw_candidates = selection.get("candidates")
    if not isinstance(raw_candidates, list) or not raw_candidates:
        raise CompanyDemoExpansionError("selection.candidates must be a non-empty list")

    # ② 从现有清单中剔除该部门旧条目，稍后用本次收录结果整体替换
    existing_documents = manifest.get("documents")
    if not isinstance(existing_documents, list) or not existing_documents:
        raise CompanyDemoExpansionError("company-demo manifest documents are invalid")
    retained_documents = [
        item
        for item in existing_documents
        if isinstance(item, dict) and item.get("department") != department
    ]
    records: list[ExpansionCandidateRecord] = []
    accepted_documents: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    seen_ids: set[str] = set()

    # ③ 逐个处理候选：查重、校验文件名安全性、检查 DOCX 内嵌对象后决定收录或跳过
    for index, raw in enumerate(raw_candidates, start=1):
        context = f"selection.candidates[{index}]"
        if not isinstance(raw, dict):
            raise CompanyDemoExpansionError(f"{context} must be an object")
        filename = _required_string(raw, "source_filename", context=context)
        document_id = _required_string(raw, "source_document_id", context=context)
        # 关键安全关卡：文件名与文档 ID 不允许重复，防止清单内自我覆盖
        if filename in seen_names or document_id in seen_ids:
            raise CompanyDemoExpansionError(f"duplicate selection identity: {filename}")
        seen_names.add(filename)
        seen_ids.add(document_id)
        # 关键安全关卡：文件名不得携带任何路径成分，且必须是 .docx 后缀
        if Path(filename).name != filename or not filename.lower().endswith(".docx"):
            raise CompanyDemoExpansionError(f"unsafe or unsupported source filename: {filename}")
        source_path = source_directory / filename
        if not source_path.is_file():
            raise CompanyDemoExpansionError(f"selected source file is missing: {filename}")

        inspection = inspect_docx_objects(source_path)
        object_fields = asdict(inspection)
        # 含图片/图形/嵌入对象的文档按契约跳过，只记录计数与原因码
        if inspection.has_image_content:
            records.append(
                ExpansionCandidateRecord(
                    source_filename=filename,
                    source_document_id=document_id,
                    status="skipped_image_content",
                    reason_code="docx_contains_image_or_drawing",
                    repository_path=None,
                    normalized_path=None,
                    **object_fields,
                )
            )
            continue

        # ④ 收录：复制 DOCX 到语料库并计算副本 SHA-256，登记文档元数据
        repository_path = Path("documents") / department / filename
        destination = target / repository_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_path, destination)
        copied_sha = _sha256_file(destination)
        document = {
            "source_document_id": document_id,
            "department": department,
            "department_label": department_label,
            "document_type": _required_string(raw, "document_type", context=context),
            "title": _required_string(raw, "title", context=context),
            "source_filename": filename,
            "repository_path": repository_path.as_posix(),
            "source_format": "docx",
            "authority": _required_string(raw, "authority", context=context),
            "status": _required_string(raw, "status", context=context),
            "sensitivity": _required_string(raw, "sensitivity", context=context),
            "sha256": copied_sha,
        }
        accepted_documents.append(document)
        records.append(
            ExpansionCandidateRecord(
                source_filename=filename,
                source_document_id=document_id,
                status="included",
                reason_code=None,
                repository_path=repository_path.as_posix(),
                normalized_path=(
                    Path("normalized") / department / f"{source_path.stem}.txt"
                ).as_posix(),
                **object_fields,
            )
        )

    if not accepted_documents:
        raise CompanyDemoExpansionError("selection produced no image-free DOCX documents")

    # ⑤ 清理该部门目录中未被选中的旧文件，保证结果与本次选择一一对应
    documents_dir = target / "documents" / department
    normalized_dir = target / "normalized" / department
    accepted_names = {item["source_filename"] for item in accepted_documents}
    if documents_dir.is_dir():
        for path in documents_dir.iterdir():
            if path.is_file() and path.name not in accepted_names:
                path.unlink()
    if normalized_dir.is_dir():
        accepted_stems = {Path(name).stem for name in accepted_names}
        for path in normalized_dir.glob("*.txt"):
            if path.stem not in accepted_stems:
                path.unlink()

    # ⑥ 合并新旧文档条目并按 (部门, 文件名) 排序，同步更新 selection 元数据后写回清单
    manifest["documents"] = sorted(
        retained_documents + accepted_documents,
        key=lambda item: (str(item.get("department")), str(item.get("source_filename"))),
    )
    selection_block = manifest.setdefault("selection", {})
    if not isinstance(selection_block, dict):
        raise CompanyDemoExpansionError("company-demo selection must be an object")
    selection_block["departments"] = [
        "human_resources",
        "finance",
        "procurement_warehouse",
        "administration",
    ]
    labels = selection_block.setdefault("department_labels", {})
    if not isinstance(labels, dict):
        raise CompanyDemoExpansionError("selection.department_labels must be an object")
    labels[department] = department_label
    selection_block["document_count"] = len(manifest["documents"])
    _write_json(manifest_path, manifest)

    # ⑦ 运行清洗流水线生成 normalized 文本；清洗校验失败统一转为领域错误
    try:
        cleaning_report = clean_company_demo(target)
    except CleaningValidationError as error:
        raise CompanyDemoExpansionError(str(error)) from error

    # ⑧ 把清洗产物（路径/摘要/字数/状态）回填到刷新后的清单并再次写回
    refreshed = _load_json(manifest_path, label="refreshed company-demo manifest")
    cleaned_by_id = {
        item.source_document_id: item for item in cleaning_report.documents
    }
    for document in refreshed.get("documents") or []:
        cleaned = cleaned_by_id.get(str(document.get("source_document_id") or ""))
        if cleaned is None:
            raise CompanyDemoExpansionError("cleaning report omitted a manifest document")
        document["normalized_path"] = cleaned.output_path
        document["normalized_sha256"] = cleaned.output_sha256
        document["normalized_characters"] = cleaned.output_characters
        document["cleaning_status"] = cleaned.status
    refreshed_selection = refreshed.get("selection")
    if isinstance(refreshed_selection, dict):
        refreshed_selection["document_count"] = len(refreshed.get("documents") or [])
        refreshed_selection["normalized_document_count"] = len(
            refreshed.get("documents") or []
        )
    _write_json(manifest_path, refreshed)

    # ⑨ 汇总内存报告并写出 JSON 准备报告（未指定路径时写到语料库根目录默认文件）
    report = CompanyDemoExpansionReport(
        department=department,
        candidate_count=len(records),
        included_count=sum(item.status == "included" for item in records),
        skipped_count=sum(item.status == "skipped_image_content" for item in records),
        records=tuple(records),
    )
    output_report = (
        Path(report_path).resolve()
        if report_path is not None
        else target / "administration_preparation_report.json"
    )
    _write_json(
        output_report,
        {
            "schema_version": 1,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "source_directory_label": source_directory.name,
            "department": report.department,
            "summary": {
                "candidate_count": report.candidate_count,
                "included_count": report.included_count,
                "skipped_count": report.skipped_count,
            },
            "candidates": [asdict(item) for item in report.records],
        },
    )
    return report


__all__ = [  # 对外导出领域错误、报告/检查数据类与准备入口
    "CompanyDemoExpansionError",
    "CompanyDemoExpansionReport",
    "DocxObjectInspection",
    "ExpansionCandidateRecord",
    "inspect_docx_objects",
    "prepare_company_demo_department",
]
