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

import logging
import os
import re
from dataclasses import dataclass
from typing import Any

import numpy as np
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from domain.documents import DocType, DocumentChunk
from infrastructure.cache.redis import EmbeddingCache, get_cache_service
from shared.config import settings


logger = logging.getLogger(__name__)


_CACHE_UNSET = object()


@dataclass(frozen=True)
class _ChunkSpan:
    """Represent one chunk as an exact local character span."""

    content: str
    start: int
    end: int





class DocumentChunkingMixin:
    """递归与语义分块能力 mixin（嵌入缓存注入，语义窗口 95 分位断点）。"""

    def _chunk_texts(
        self,
        texts: list[str],
        doc_id: str,
        doc_type: DocType,
        source: str,
        tenant_id: str = "",
    ) -> list[DocumentChunk]:
        """Recursively split parsed text with Chinese-aware ordered separators."""
        chunks: list[DocumentChunk] = []
        chunk_index = 0
        for text in texts:
            spans = self._recursive_spans(text)
            built = self._build_document_chunks(
                spans,
                doc_id=doc_id,
                doc_type=doc_type,
                source=source,
                tenant_id=tenant_id,
                start_index=chunk_index,
                strategy="recursive",
            )
            chunks.extend(built)
            chunk_index += len(built)
        return chunks

    def _recursive_spans(
        self,
        text: str,
        *,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> list[_ChunkSpan]:
        """Return recursive chunks with exact start indexes in one source text."""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size or settings.chunk_size,
            chunk_overlap=(
                settings.chunk_overlap if chunk_overlap is None else chunk_overlap
            ),
            separators=list(self.CHUNK_SEPARATORS),
            keep_separator=True,
            add_start_index=True,
        )
        spans: list[_ChunkSpan] = []
        search_cursor = 0
        for document in splitter.create_documents([text]):
            content = document.page_content
            if not content.strip():
                continue
            start_index = int(document.metadata.get("start_index", -1))
            if start_index < 0:
                start_index = text.find(content, search_cursor)
            if start_index < 0:
                start_index = text.find(content)
            if start_index < 0:
                raise ValueError("recursive chunk start index could not be resolved")
            end_index = start_index + len(content)
            spans.append(_ChunkSpan(content=content, start=start_index, end=end_index))
            overlap = settings.chunk_overlap if chunk_overlap is None else chunk_overlap
            search_cursor = max(start_index + 1, end_index - overlap)
        return spans

    def _build_document_chunks(
        self,
        spans: list[_ChunkSpan],
        *,
        doc_id: str,
        doc_type: DocType,
        source: str,
        tenant_id: str,
        start_index: int,
        strategy: str,
    ) -> list[DocumentChunk]:
        """Convert local text spans into tenant-safe domain chunks."""
        return [
            DocumentChunk(
                content=span.content,
                doc_id=doc_id,
                chunk_index=start_index + offset,
                doc_type=doc_type,
                metadata={
                    "source": source,
                    "file_name": os.path.basename(source),
                    "char_start": span.start,
                    "char_end": span.end,
                    "chunking_strategy": strategy,
                },
                tenant_id=tenant_id,
            )
            for offset, span in enumerate(spans)
        ]

    async def _semantic_spans_with_fallback(
        self, text: str
    ) -> tuple[list[_ChunkSpan], str]:
        """Return semantic spans or recursive spans when semantic work is unsafe."""
        sentence_spans = self._sentence_spans(text)
        if len(sentence_spans) < 4:
            return self._recursive_spans(text), "recursive"

        try:
            windows = self._semantic_windows(text, sentence_spans)
            embeddings = await self._embed_semantic_windows(windows)
            breakpoint_scores = self._semantic_breakpoint_scores(
                sentence_spans, embeddings
            )
            spans = self._partition_semantic_spans(
                text, sentence_spans, breakpoint_scores
            )
            return spans, "semantic"
        except Exception:
            logger.warning(
                "semantic document chunking failed; using recursive fallback"
            )
            return self._recursive_spans(text), "recursive"

    @staticmethod
    def _sentence_spans(text: str) -> list[_ChunkSpan]:
        """Split Chinese text on retained sentence punctuation and newlines."""
        if not text.strip():
            return []
        left = len(text) - len(text.lstrip())
        right = len(text.rstrip())
        content = text[left:right]
        spans: list[_ChunkSpan] = []
        for match in re.finditer(r".+?(?:[。！？；]+|\n+|$)", content, re.DOTALL):
            start = left + match.start()
            end = left + match.end()
            segment = text[start:end]
            if segment.strip():
                spans.append(_ChunkSpan(content=segment, start=start, end=end))
        return spans

    @staticmethod
    def _semantic_windows(text: str, spans: list[_ChunkSpan]) -> list[str]:
        """Build previous/current/next sentence context windows for embedding."""
        windows: list[str] = []
        for index in range(len(spans)):
            first = spans[max(0, index - 1)].start
            last = spans[min(len(spans) - 1, index + 1)].end
            windows.append(text[first:last].strip())
        return windows

    async def _embed_semantic_windows(
        self, windows: list[str]
    ) -> list[list[float]]:
        """Resolve semantic window vectors from best-effort cache and batch I/O."""
        vectors: list[list[float] | None] = [None] * len(windows)
        cache = self._get_embedding_cache()
        cache_reads_enabled = cache is not None

        for index, window in enumerate(windows):
            if not cache_reads_enabled or cache is None:
                continue
            try:
                vectors[index] = self._coerce_embedding(
                    await cache.get(window, model=settings.embedding_model)
                )
            except Exception:
                logger.warning(
                    "semantic embedding cache read failed; continuing without cache"
                )
                cache_reads_enabled = False

        missing_indices = [
            index for index, vector in enumerate(vectors) if vector is None
        ]
        if missing_indices:
            embeddings = self._get_semantic_embeddings()
            missing_windows = [windows[index] for index in missing_indices]
            provided = await embeddings.aembed_documents(missing_windows)
            if len(provided) != len(missing_indices):
                raise ValueError("embedding provider returned an unexpected vector count")
            for index, value in zip(missing_indices, provided):
                vector = self._coerce_embedding(value)
                if vector is None:
                    raise ValueError("embedding provider returned an invalid vector")
                vectors[index] = vector

            if cache is not None:
                cache_writes_enabled = True
                for index in missing_indices:
                    if not cache_writes_enabled:
                        break
                    try:
                        await cache.set(
                            windows[index],
                            vectors[index] or [],
                            model=settings.embedding_model,
                        )
                    except Exception:
                        logger.warning(
                            "semantic embedding cache write failed; continuing without cache"
                        )
                        cache_writes_enabled = False

        if any(vector is None for vector in vectors):
            raise ValueError("semantic embeddings are incomplete")
        return [vector for vector in vectors if vector is not None]

    def _get_semantic_embeddings(self) -> Any:
        """Create the existing DashScope-compatible embedding client lazily."""
        if self._embeddings is None:
            self._embeddings = OpenAIEmbeddings(
                model=settings.embedding_model,
                api_key=settings.dashscope_api_key,
                base_url=settings.dashscope_base_url,
                check_embedding_ctx_length=False,
                chunk_size=settings.embedding_batch_size,
            )
        return self._embeddings

    def _get_embedding_cache(self) -> EmbeddingCache | Any | None:
        """Resolve the existing Redis embedding cache without making it mandatory."""
        if self._embedding_cache is _CACHE_UNSET:
            try:
                self._embedding_cache = EmbeddingCache(get_cache_service())
            except Exception:
                logger.warning(
                    "semantic embedding cache initialization failed; continuing without cache"
                )
                self._embedding_cache = None
        return self._embedding_cache

    @staticmethod
    def _coerce_embedding(value: Any) -> list[float] | None:
        """Validate one cached or provider vector as finite one-dimensional data."""
        if value is None:
            return None
        try:
            vector = np.asarray(value, dtype=float)
        except (TypeError, ValueError):
            return None
        if vector.ndim != 1 or vector.size == 0 or not np.isfinite(vector).all():
            return None
        return vector.tolist()

    @staticmethod
    def _semantic_breakpoint_scores(
        spans: list[_ChunkSpan], embeddings: list[list[float]]
    ) -> dict[int, float]:
        """Map sentence-end offsets to cosine distances above the percentile."""
        matrix = np.asarray(embeddings, dtype=float)
        if matrix.ndim != 2 or matrix.shape[0] != len(spans):
            raise ValueError("semantic embedding matrix shape is invalid")
        if not np.isfinite(matrix).all():
            raise ValueError("semantic embedding matrix contains non-finite values")
        norms = np.linalg.norm(matrix, axis=1)
        if np.any(norms == 0):
            raise ValueError("semantic embedding matrix contains zero vectors")
        normalized = matrix / norms[:, None]
        distances = 1.0 - np.sum(normalized[:-1] * normalized[1:], axis=1)
        distances = np.clip(distances, 0.0, 2.0)
        threshold = float(
            np.percentile(distances, settings.semantic_chunk_percentile)
        )
        return {
            spans[index].end: float(distance)
            for index, distance in enumerate(distances)
            if distance >= threshold
        }

    def _partition_semantic_spans(
        self,
        text: str,
        sentence_spans: list[_ChunkSpan],
        breakpoint_scores: dict[int, float],
    ) -> list[_ChunkSpan]:
        """Partition text at preferred boundaries while enforcing size bounds."""
        if not sentence_spans:
            return []
        content_start = sentence_spans[0].start
        content_end = sentence_spans[-1].end
        minimum = settings.semantic_chunk_min_size
        maximum = settings.semantic_chunk_max_size
        fallback_boundaries = {span.end for span in sentence_spans[:-1]}
        fallback_boundaries.update(
            span.end
            for span in self._recursive_spans(
                text, chunk_size=maximum, chunk_overlap=0
            )[:-1]
        )

        spans: list[_ChunkSpan] = []
        current = content_start
        while current < content_end:
            upper = min(current + maximum, content_end)
            semantic_choices = [
                boundary
                for boundary in breakpoint_scores
                if self._is_legal_boundary(
                    boundary, current, content_end, minimum, upper
                )
            ]
            if semantic_choices:
                chosen = max(
                    semantic_choices,
                    key=lambda boundary: (breakpoint_scores[boundary], boundary),
                )
            elif content_end - current <= maximum:
                chosen = content_end
            else:
                fallback_choices = [
                    boundary
                    for boundary in fallback_boundaries
                    if self._is_legal_boundary(
                        boundary, current, content_end, minimum, upper
                    )
                ]
                if fallback_choices:
                    chosen = max(fallback_choices)
                else:
                    chosen = self._hard_bounded_cut(
                        current, content_end, minimum, maximum
                    )

            content = text[current:chosen]
            if content.strip():
                spans.append(_ChunkSpan(content=content, start=current, end=chosen))
            current = chosen

        return spans

    @staticmethod
    def _is_legal_boundary(
        boundary: int,
        current: int,
        content_end: int,
        minimum: int,
        upper: int,
    ) -> bool:
        """Return whether a boundary satisfies current and remaining minimums."""
        remaining = content_end - boundary
        return (
            current + minimum <= boundary <= upper
            and (remaining == 0 or remaining >= minimum)
        )

    @staticmethod
    def _hard_bounded_cut(
        current: int, content_end: int, minimum: int, maximum: int
    ) -> int:
        """Choose a deterministic character cut when no natural boundary is legal."""
        chosen = min(current + maximum, content_end)
        trailing = content_end - chosen
        if 0 < trailing < minimum:
            adjusted = content_end - minimum
            if adjusted - current >= minimum:
                chosen = adjusted
        if chosen <= current:
            raise ValueError("semantic chunk bounds cannot advance the partition")
        return chosen
