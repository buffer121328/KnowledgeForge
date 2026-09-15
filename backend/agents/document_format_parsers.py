"""
文档解析 Agent — 多模态文档解析，支持 PDF / Word / 图片 / 表格 / 纯文本

核心能力:
  1. PDF 解析（文字 + 嵌入图片 + 表格）
  2. Word 解析（.docx / .doc）
  3. 图片 OCR + LLM 视觉理解
  4. 表格结构化提取
  5. 文档分块（Chunking）与元数据标注
"""

from __future__ import annotations

import hashlib
import logging
import os
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from domain.documents import DocType, DocumentChunk, IngestDocumentInput
from shared.config import settings


logger = logging.getLogger(__name__)


_CACHE_UNSET = object()


@dataclass(frozen=True)
class _ChunkSpan:
    """Represent one chunk as an exact local character span."""

    content: str
    start: int
    end: int




class DocumentWorkLimitError(ValueError):
    """Raised when parser work would exceed configured resource policy."""




def _extract_docx_textbox_text(file_path: str) -> list[str]:
    """Extract text from w:txbxContent elements as a fallback for flowchart docx.

    流程图类 Word 文档把正文放在文本框里，python-docx 的段落/表格接口读取
    不到。这里直接从 document.xml 提取每个文本框的纯文本，按出现顺序返回；
    提取结果仍受 upload_max_text_characters 上限约束。
    """
    import zipfile
    from xml.etree import ElementTree

    namespace = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
    with zipfile.ZipFile(file_path) as archive:
        xml = archive.read("word/document.xml")
    root = ElementTree.fromstring(xml)
    parts: list[str] = []
    total = 0
    for textbox in root.iter(f"{namespace}txbxContent"):
        lines: list[str] = []
        for node in textbox.iter(f"{namespace}t"):
            if node.text and node.text.strip():
                lines.append(node.text.strip())
        text = "\n".join(dict.fromkeys(lines)).strip()
        if not text:
            continue
        total += len(text)
        if total > settings.upload_max_text_characters:
            raise DocumentWorkLimitError("Word character limit exceeded")
        parts.append(text)
    return parts





class FormatParsingMixin:
    """多格式解析能力 mixin：PDF/Word/图片 OCR/表格/纯文本。"""

    """
    文档解析 Agent

    工作流:
      classify → parse → chunk → enrich_metadata → output
    """

    SUPPORTED_EXTENSIONS: dict[str, DocType] = {
        ".pdf": DocType.PDF,
        ".docx": DocType.WORD,
        ".doc": DocType.WORD,
        ".png": DocType.IMAGE,
        ".jpg": DocType.IMAGE,
        ".jpeg": DocType.IMAGE,
        ".tiff": DocType.IMAGE,
        ".bmp": DocType.IMAGE,
        ".csv": DocType.TABLE,
        ".xlsx": DocType.TABLE,
        ".xls": DocType.TABLE,
        ".txt": DocType.TEXT,
        ".md": DocType.MARKDOWN,
        ".html": DocType.TEXT,
        ".htm": DocType.TEXT,
    }

    CHUNK_SIZE = 400
    CHUNK_OVERLAP = 128
    CHUNK_SEPARATORS = ("\n\n", "\n", "。", "！", "？", "；", "，", "、", "")


    def __init__(
        self,
        *,
        embeddings: Any | None = None,
        embedding_cache: Any = _CACHE_UNSET,
    ) -> None:
        """Initialize parser model clients with lazy semantic collaborators."""
        self.vision_llm = ChatOpenAI(
            model=settings.vision_model,
            api_key=settings.dashscope_api_key,
            base_url=settings.dashscope_base_url,
            temperature=0,
        )
        self._embeddings = embeddings
        self._embedding_cache = embedding_cache

    # ── public API ───────────────────────────────────────────

    async def parse(
        self,
        file_path: str,
        tenant_id: str = "",
        document_input: IngestDocumentInput | None = None,
    ) -> list[DocumentChunk]:
        """Parse one file and retain optional stable catalog context on every chunk."""

        effective_tenant = document_input.tenant_id if document_input else tenant_id
        doc_type = self._classify(file_path)
        doc_id = document_input.doc_id if document_input else self._make_doc_id(file_path)

        raw_texts: list[str] = []
        if doc_type == DocType.PDF:
            raw_texts = await self._parse_pdf(file_path)
        elif doc_type == DocType.WORD:
            raw_texts = await self._parse_word(file_path)
        elif doc_type == DocType.IMAGE:
            raw_texts = await self._parse_image(file_path)
        elif doc_type == DocType.TABLE:
            raw_texts = await self._parse_table(file_path)
        elif doc_type in (DocType.TEXT, DocType.MARKDOWN):
            raw_texts = self._parse_text(file_path)
        else:
            raise ValueError(f"不支持的文件类型: {os.path.splitext(file_path)[1]}")

        chunks = await self._chunk_texts_for_strategy(
            raw_texts, doc_id, doc_type, file_path, effective_tenant
        )
        if document_input is not None:
            context_metadata = document_input.chunk_metadata()
            for chunk in chunks:
                chunk.metadata.update(context_metadata)
                chunk.tenant_id = document_input.tenant_id
        return chunks

    async def parse_batch(
        self,
        file_paths: list[str],
        tenant_id: str = "",
        document_inputs: list[IngestDocumentInput] | None = None,
    ) -> list[DocumentChunk]:
        """Parse multiple files with one aligned catalog context per document."""

        if document_inputs is not None and len(document_inputs) != len(file_paths):
            raise ValueError("document input count must match file path count")
        all_chunks: list[DocumentChunk] = []
        for index, file_path in enumerate(file_paths):
            document_input = document_inputs[index] if document_inputs is not None else None
            all_chunks.extend(
                await self.parse(
                    file_path,
                    tenant_id=tenant_id,
                    document_input=document_input,
                )
            )
        return all_chunks

    # ── classification ───────────────────────────────────────

    def _classify(self, file_path: str) -> DocType:
        """Classify the doc parser agent."""
        ext = os.path.splitext(file_path)[1].lower()
        return self.SUPPORTED_EXTENSIONS.get(ext, DocType.UNKNOWN)

    @staticmethod
    def _make_doc_id(file_path: str) -> str:
        """Create the doc ID."""
        return hashlib.sha256(file_path.encode()).hexdigest()[:16]

    # ── PDF parsing ──────────────────────────────────────────

    async def _parse_pdf(self, file_path: str) -> list[str]:
        """
        PDF 多模态解析:
          1. 提取文字页面
          2. 如果页面包含图片 / 表格，调用 LLM 视觉理解
        """
        texts: list[str] = []
        try:
            from pypdf import PdfReader

            reader = PdfReader(file_path)
            if len(reader.pages) > settings.upload_max_pdf_pages:
                raise DocumentWorkLimitError("PDF page limit exceeded")
            for page in reader.pages:
                page_text = page.extract_text() or ""
                if page_text.strip():
                    texts.append(page_text.strip())
        except DocumentWorkLimitError:
            raise
        except Exception:
            texts.append(f"[PDF 解析失败] {file_path}")

        if not texts:
            texts = await self._pdf_vision_fallback(file_path)

        return texts

    async def _pdf_vision_fallback(self, file_path: str) -> list[str]:
        """当 PDF 纯文本提取失败时，使用 LLM 视觉能力"""
        try:
            from pdf2image import convert_from_path

            images = convert_from_path(file_path, dpi=150, first_page=1, last_page=5)
            texts: list[str] = []
            for img in images:
                description = await self._describe_image_with_llm(img)
                texts.append(description)
            return texts
        except Exception:
            return [f"[PDF 视觉解析失败] {file_path}"]

    # ── Word parsing ─────────────────────────────────────────

    async def _parse_word(self, file_path: str) -> list[str]:
        """Word 解析: .docx 用 python-docx；旧版 .doc 尝试 unstructured"""
        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".docx":
            return self._parse_docx(file_path)
        return await self._parse_doc_legacy(file_path)

    @staticmethod
    def _parse_docx(file_path: str) -> list[str]:
        """Parse the docx."""
        try:
            from docx import Document
        except ImportError as e:
            raise RuntimeError("缺少 python-docx 依赖，无法解析 Word 文档") from e

        try:
            doc = Document(file_path)
            parts: list[str] = []
            para_buf: list[str] = []
            for p in doc.paragraphs:
                text = (p.text or "").strip()
                if text:
                    para_buf.append(text)
                    if sum(len(item) for item in para_buf) > settings.upload_max_text_characters:
                        raise DocumentWorkLimitError("Word character limit exceeded")
            if para_buf:
                parts.append("\n".join(para_buf))
            for table in doc.tables:
                rows: list[str] = []
                for row in table.rows:
                    cells = [((c.text or "").strip().replace("\n", " ")) for c in row.cells]
                    if any(cells):
                        rows.append(" | ".join(cells))
                if rows:
                    parts.append("\n".join(rows))
                if sum(len(item) for item in parts) > settings.upload_max_text_characters:
                    raise DocumentWorkLimitError("Word character limit exceeded")
            if not parts:
                # 流程图/示意图类 docx 的正文常放在文本框（w:txbxContent）里，
                # python-docx 的 paragraphs/tables 都不会读取。这里从 document
                # XML 兜底提取文本框文本，避免这类文档被解析成空壳。
                textbox_parts = _extract_docx_textbox_text(file_path)
                if textbox_parts:
                    return textbox_parts
                return [f"[Word 文档无文本内容] {os.path.basename(file_path)}"]
            return parts
        except DocumentWorkLimitError:
            raise
        except Exception as e:
            raise RuntimeError(f"Word 文档解析失败: {e}") from e

    async def _parse_doc_legacy(self, file_path: str) -> list[str]:
        """旧版 .doc：优先 unstructured，失败则明确报错"""
        try:
            from unstructured.partition.auto import partition
            elements = partition(filename=file_path)
            texts = [str(el).strip() for el in elements if str(el).strip()]
            if texts:
                return texts
        except Exception:
            pass
        raise RuntimeError("不支持解析旧版 .doc，请另存为 .docx 后再上传")

    # ── image parsing ────────────────────────────────────────

    async def _parse_image(self, file_path: str) -> list[str]:
        """图片解析: OCR + LLM 视觉理解"""
        from PIL import Image

        with Image.open(file_path) as probe:
            if probe.width * probe.height > settings.upload_max_image_pixels:
                raise DocumentWorkLimitError("image pixel limit exceeded")
        texts: list[str] = []
        ocr_text = self._ocr(file_path)
        if ocr_text.strip():
            texts.append(ocr_text)

        img = Image.open(file_path)
        description = await self._describe_image_with_llm(img)
        texts.append(description)
        return texts

    @staticmethod
    def _ocr(file_path: str) -> str:
        """Extract image text with OCR, returning an empty string on failure."""
        try:
            import pytesseract
            from PIL import Image
            return pytesseract.image_to_string(Image.open(file_path), lang="chi_sim+eng")
        except Exception:
            return ""

    async def _describe_image_with_llm(self, image: Any) -> str:
        """调用 LLM 多模态能力描述图片内容"""
        import base64
        import io

        buf = io.BytesIO()
        image.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()

        messages = [
            SystemMessage(content="你是一个专业的文档分析助手，请详细描述图片中的内容，包括文字、表格、图表信息。"),
            HumanMessage(content=[
                {"type": "text", "text": "请描述这张图片的所有内容："},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            ]),
        ]
        resp = await self.vision_llm.ainvoke(messages)
        return resp.content

    # ── table parsing ────────────────────────────────────────

    async def _parse_table(self, file_path: str) -> list[str]:
        """表格解析: CSV / Excel → 结构化文本"""
        ext = os.path.splitext(file_path)[1].lower()
        try:
            if ext == ".csv":
                return self._parse_csv(file_path)
            else:
                return self._parse_excel(file_path)
        except DocumentWorkLimitError:
            raise
        except Exception:
            return [f"[表格解析失败] {file_path}"]

    @staticmethod
    def _parse_csv(file_path: str) -> list[str]:
        """Parse the CSV."""
        import csv
        texts: list[str] = []
        with open(file_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames or []
            rows: list[str] = []
            for row_index, row in enumerate(reader, start=1):
                if row_index > settings.upload_max_spreadsheet_rows:
                    raise DocumentWorkLimitError("CSV row limit exceeded")
                rows.append(" | ".join(f"{h}: {row.get(h, '')}" for h in headers))
            for i in range(0, len(rows), 20):
                batch = rows[i : i + 20]
                texts.append(f"表头: {' | '.join(headers)}\n" + "\n".join(batch))
        return texts or ["[空 CSV]"]

    @staticmethod
    def _parse_excel(file_path: str) -> list[str]:
        """Parse the excel."""
        try:
            import openpyxl
            wb = openpyxl.load_workbook(file_path, read_only=True)
            texts: list[str] = []
            for sheet in wb.worksheets:
                row_iterator = sheet.iter_rows(values_only=True)
                first_row = next(row_iterator, None)
                if first_row is None:
                    continue
                headers = [str(c) if c else "" for c in first_row]
                data_rows: list[str] = []
                for row_index, row in enumerate(row_iterator, start=1):
                    if row_index > settings.upload_max_spreadsheet_rows:
                        raise DocumentWorkLimitError("Excel row limit exceeded")
                    data_rows.append(" | ".join(
                        f"{headers[j]}: {row[j]}" if j < len(headers) else str(row[j])
                        for j in range(len(row))
                    ))
                for i in range(0, len(data_rows), 20):
                    batch = data_rows[i : i + 20]
                    texts.append(f"工作表: {sheet.title}\n表头: {' | '.join(headers)}\n" + "\n".join(batch))
            return texts or ["[空 Excel]"]
        except DocumentWorkLimitError:
            raise
        except Exception:
            return [f"[Excel 解析失败] {file_path}"]

    # ── text / markdown ──────────────────────────────────────

    @staticmethod
    def _parse_text(file_path: str) -> list[str]:
        """Parse the text."""
        with open(file_path, encoding="utf-8") as f:
            text = f.read(settings.upload_max_text_characters + 1)
        if len(text) > settings.upload_max_text_characters:
            raise DocumentWorkLimitError("text character limit exceeded")
        return [text]

    # ── chunking ─────────────────────────────────────────────

    async def _chunk_texts_for_strategy(
        self,
        texts: list[str],
        doc_id: str,
        doc_type: DocType,
        source: str,
        tenant_id: str = "",
    ) -> list[DocumentChunk]:
        """Dispatch configured chunking while keeping semantic I/O asynchronous."""
        if settings.chunking_strategy == "recursive":
            return self._chunk_texts(texts, doc_id, doc_type, source, tenant_id)

        chunks: list[DocumentChunk] = []
        chunk_index = 0
        for text in texts:
            spans, effective_strategy = await self._semantic_spans_with_fallback(text)
            built = self._build_document_chunks(
                spans,
                doc_id=doc_id,
                doc_type=doc_type,
                source=source,
                tenant_id=tenant_id,
                start_index=chunk_index,
                strategy=effective_strategy,
            )
            chunks.extend(built)
            chunk_index += len(built)
        return chunks

