"""DocParserAgent 单元测试"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agents.document_parser import DocParserAgent
from domain.documents import DocType, DocumentChunk
from shared.config.settings import Settings


@pytest.fixture
def agent():
    """跳过 ChatOpenAI 真实初始化"""
    with patch("agents.document_format_parsers.ChatOpenAI"):
        return DocParserAgent()


class TestDocTypeClassification:
    def test_classify_pdf(self, agent):
        assert agent._classify("report.pdf") == DocType.PDF

    def test_classify_image(self, agent):
        assert agent._classify("photo.png") == DocType.IMAGE
        assert agent._classify("diagram.jpg") == DocType.IMAGE
        assert agent._classify("photo.jpeg") == DocType.IMAGE

    def test_classify_table(self, agent):
        assert agent._classify("data.csv") == DocType.TABLE
        assert agent._classify("report.xlsx") == DocType.TABLE
        assert agent._classify("report.xls") == DocType.TABLE

    def test_classify_text_and_markdown(self, agent):
        assert agent._classify("notes.txt") == DocType.TEXT
        assert agent._classify("readme.md") == DocType.MARKDOWN

    def test_classify_word(self, agent):
        assert agent._classify("report.docx") == DocType.WORD
        assert agent._classify("legacy.doc") == DocType.WORD

    def test_classify_unknown(self, agent):
        assert agent._classify("archive.zip") == DocType.UNKNOWN
        assert agent._classify("file.xyz") == DocType.UNKNOWN


class TestDocIdGeneration:
    def test_make_doc_id_deterministic(self):
        id1 = DocParserAgent._make_doc_id("/path/to/file.pdf")
        id2 = DocParserAgent._make_doc_id("/path/to/file.pdf")
        assert id1 == id2

    def test_make_doc_id_different_paths(self):
        id1 = DocParserAgent._make_doc_id("/path/to/file1.pdf")
        id2 = DocParserAgent._make_doc_id("/path/to/file2.pdf")
        assert id1 != id2

    def test_make_doc_id_length(self):
        doc_id = DocParserAgent._make_doc_id("/some/path")
        assert len(doc_id) == 16


class TestChunkTexts:
    def test_chunk_short_text(self, agent):
        """A short text produces one recursive chunk with traceable metadata."""
        chunks = agent._chunk_texts(
            ["短文本"],
            doc_id="doc_001",
            doc_type=DocType.TEXT,
            source="test.txt",
        )

        assert len(chunks) == 1
        assert chunks[0].doc_id == "doc_001"
        assert chunks[0].doc_type == DocType.TEXT
        assert chunks[0].chunk_index == 0
        assert chunks[0].content == "短文本"
        assert chunks[0].metadata["char_start"] == 0
        assert chunks[0].metadata["char_end"] == 3
        assert chunks[0].metadata["chunking_strategy"] == "recursive"

    def test_recursive_chunking_prefers_chinese_boundaries(self, agent, monkeypatch):
        """Recursive chunking keeps Chinese sentence punctuation with its text."""
        monkeypatch.setattr("agents.document_chunking.settings.chunk_size", 12)
        monkeypatch.setattr("agents.document_chunking.settings.chunk_overlap", 0)
        text = "第一段内容。第二段内容！第三段内容？"

        chunks = agent._chunk_texts(
            [text],
            doc_id="doc_001",
            doc_type=DocType.TEXT,
            source="test.txt",
        )

        assert "".join(chunk.content for chunk in chunks) == text
        assert all(len(chunk.content) <= 12 for chunk in chunks)
        for sentence in ("第一段内容", "第二段内容", "第三段内容"):
            assert sum(sentence in chunk.content for chunk in chunks) == 1

    def test_recursive_chunking_retains_configured_overlap(self, agent, monkeypatch):
        """Character fallback retains the configured recursive overlap."""
        monkeypatch.setattr("agents.document_chunking.settings.chunk_size", 12)
        monkeypatch.setattr("agents.document_chunking.settings.chunk_overlap", 4)

        chunks = agent._chunk_texts(
            ["0123456789abcdefghijklmnop"],
            doc_id="doc_001",
            doc_type=DocType.TEXT,
            source="test.txt",
        )

        assert len(chunks) >= 2
        assert chunks[0].content[-4:] == chunks[1].content[:4]
        assert all(len(chunk.content) <= 12 for chunk in chunks)

    def test_chunk_index_source_and_tenant_are_preserved(self, agent, monkeypatch):
        """All recursive chunks preserve ordered source and tenant metadata."""
        monkeypatch.setattr("agents.document_chunking.settings.chunk_size", 8)
        monkeypatch.setattr("agents.document_chunking.settings.chunk_overlap", 2)

        chunks = agent._chunk_texts(
            ["甲" * 20],
            doc_id="doc_001",
            doc_type=DocType.TEXT,
            source="/tmp/file.txt",
            tenant_id="org_001",
        )

        assert [chunk.chunk_index for chunk in chunks] == list(range(len(chunks)))
        assert all(chunk.tenant_id == "org_001" for chunk in chunks)
        assert all(chunk.metadata["source"] == "/tmp/file.txt" for chunk in chunks)
        assert all(chunk.metadata["file_name"] == "file.txt" for chunk in chunks)


class TestSemanticChunkTexts:
    @staticmethod
    def _agent(embeddings, cache=None):
        """Build a parser with isolated semantic embedding collaborators."""
        with patch("agents.document_format_parsers.ChatOpenAI"):
            return DocParserAgent(embeddings=embeddings, embedding_cache=cache)

    @pytest.mark.asyncio
    async def test_short_text_skips_embeddings_and_falls_back(self, monkeypatch):
        """Semantic mode skips embedding I/O for fewer than four sentences."""
        monkeypatch.setattr("agents.document_chunking.settings.chunking_strategy", "semantic")
        embeddings = MagicMock()
        embeddings.aembed_documents = AsyncMock()
        agent = self._agent(embeddings)

        chunks = await agent._chunk_texts_for_strategy(
            ["第一句。第二句。第三句。"],
            doc_id="doc_001",
            doc_type=DocType.TEXT,
            source="test.txt",
            tenant_id="org_001",
        )

        embeddings.aembed_documents.assert_not_awaited()
        assert chunks[0].metadata["chunking_strategy"] == "recursive"
        assert all(chunk.tenant_id == "org_001" for chunk in chunks)

    @pytest.mark.asyncio
    async def test_semantic_distance_creates_legal_breakpoint(self, monkeypatch):
        """A high adjacent cosine distance creates a semantic boundary."""
        monkeypatch.setattr("agents.document_chunking.settings.chunking_strategy", "semantic")
        monkeypatch.setattr("agents.document_chunking.settings.semantic_chunk_min_size", 200)
        monkeypatch.setattr("agents.document_chunking.settings.semantic_chunk_max_size", 512)
        embeddings = MagicMock()
        embeddings.aembed_documents = AsyncMock(
            return_value=[
                [1.0, 0.0],
                [1.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [0.0, 1.0],
                [0.0, 1.0],
            ]
        )
        agent = self._agent(embeddings)
        text = "".join(f"{character * 69}。" for character in "甲乙丙丁戊己")

        chunks = await agent._chunk_texts_for_strategy(
            [text],
            doc_id="doc_001",
            doc_type=DocType.TEXT,
            source="test.txt",
        )

        assert [len(chunk.content) for chunk in chunks] == [210, 210]
        assert all(chunk.metadata["chunking_strategy"] == "semantic" for chunk in chunks)
        embeddings.aembed_documents.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_semantic_chunks_enforce_minimum_and_maximum(self, monkeypatch):
        """Semantic mode ignores tiny candidates and uses bounded fallback cuts."""
        monkeypatch.setattr("agents.document_chunking.settings.chunking_strategy", "semantic")
        monkeypatch.setattr("agents.document_chunking.settings.semantic_chunk_min_size", 200)
        monkeypatch.setattr("agents.document_chunking.settings.semantic_chunk_max_size", 300)
        embeddings = MagicMock()
        embeddings.aembed_documents = AsyncMock(
            return_value=[[1.0, 0.0], [0.0, 1.0]] + [[0.0, 1.0]] * 6
        )
        agent = self._agent(embeddings)
        text = "".join(f"{character * 69}。" for character in "甲乙丙丁戊己庚辛")

        chunks = await agent._chunk_texts_for_strategy(
            [text],
            doc_id="doc_001",
            doc_type=DocType.TEXT,
            source="test.txt",
        )

        assert "".join(chunk.content for chunk in chunks) == text
        assert all(200 <= len(chunk.content) <= 300 for chunk in chunks)

    @pytest.mark.asyncio
    async def test_cached_windows_skip_embedding_provider(self, monkeypatch):
        """Complete cache hits avoid a provider request."""
        monkeypatch.setattr("agents.document_chunking.settings.chunking_strategy", "semantic")
        embeddings = MagicMock()
        embeddings.aembed_documents = AsyncMock()
        cache = MagicMock()
        cache.get = AsyncMock(return_value=[1.0, 0.0])
        cache.set = AsyncMock()
        agent = self._agent(embeddings, cache)
        text = "".join(f"{character * 69}。" for character in "甲乙丙丁戊己")

        chunks = await agent._chunk_texts_for_strategy(
            [text],
            doc_id="doc_001",
            doc_type=DocType.TEXT,
            source="test.txt",
        )

        embeddings.aembed_documents.assert_not_awaited()
        assert cache.get.await_count == 6
        assert chunks[0].metadata["chunking_strategy"] == "semantic"

    @pytest.mark.asyncio
    async def test_partial_cache_hit_requests_only_missing_windows(self, monkeypatch):
        """The provider receives only windows that were absent from cache."""
        monkeypatch.setattr("agents.document_chunking.settings.chunking_strategy", "semantic")
        embeddings = MagicMock()
        embeddings.aembed_documents = AsyncMock(return_value=[[1.0, 0.0]] * 5)
        cache = MagicMock()
        cache.get = AsyncMock(
            side_effect=[[1.0, 0.0], None, None, None, None, None]
        )
        cache.set = AsyncMock()
        agent = self._agent(embeddings, cache)
        text = "".join(f"{character * 69}。" for character in "甲乙丙丁戊己")

        chunks = await agent._chunk_texts_for_strategy(
            [text],
            doc_id="doc_001",
            doc_type=DocType.TEXT,
            source="test.txt",
        )

        requested_windows = embeddings.aembed_documents.await_args.args[0]
        assert len(requested_windows) == 5
        assert cache.set.await_count == 5
        assert chunks[0].metadata["chunking_strategy"] == "semantic"

    @pytest.mark.asyncio
    async def test_invalid_embedding_vectors_fall_back_to_recursive(self, monkeypatch):
        """Invalid zero vectors trigger recursive fallback without an exception."""
        monkeypatch.setattr("agents.document_chunking.settings.chunking_strategy", "semantic")
        embeddings = MagicMock()
        embeddings.aembed_documents = AsyncMock(return_value=[[0.0, 0.0]] * 6)
        agent = self._agent(embeddings)
        text = "".join(f"{character * 69}。" for character in "甲乙丙丁戊己")

        chunks = await agent._chunk_texts_for_strategy(
            [text],
            doc_id="doc_001",
            doc_type=DocType.TEXT,
            source="test.txt",
        )

        assert chunks
        assert all(chunk.metadata["chunking_strategy"] == "recursive" for chunk in chunks)

    @pytest.mark.asyncio
    async def test_cache_failure_continues_with_provider(self, monkeypatch):
        """Cache failures do not prevent semantic embedding acquisition."""
        monkeypatch.setattr("agents.document_chunking.settings.chunking_strategy", "semantic")
        embeddings = MagicMock()
        embeddings.aembed_documents = AsyncMock(return_value=[[1.0, 0.0]] * 6)
        cache = MagicMock()
        cache.get = AsyncMock(side_effect=ConnectionError("cache unavailable"))
        cache.set = AsyncMock(side_effect=ConnectionError("cache unavailable"))
        agent = self._agent(embeddings, cache)
        text = "".join(f"{character * 69}。" for character in "甲乙丙丁戊己")

        chunks = await agent._chunk_texts_for_strategy(
            [text],
            doc_id="doc_001",
            doc_type=DocType.TEXT,
            source="test.txt",
        )

        embeddings.aembed_documents.assert_awaited_once()
        assert chunks[0].metadata["chunking_strategy"] == "semantic"

    @pytest.mark.asyncio
    async def test_embedding_failure_falls_back_to_recursive(self, monkeypatch):
        """Provider failure falls back without losing tenant metadata."""
        monkeypatch.setattr("agents.document_chunking.settings.chunking_strategy", "semantic")
        embeddings = MagicMock()
        embeddings.aembed_documents = AsyncMock(side_effect=RuntimeError("provider unavailable"))
        agent = self._agent(embeddings)
        text = "".join(f"{character * 69}。" for character in "甲乙丙丁戊己")

        chunks = await agent._chunk_texts_for_strategy(
            [text],
            doc_id="doc_001",
            doc_type=DocType.TEXT,
            source="test.txt",
            tenant_id="org_001",
        )

        assert chunks
        assert all(chunk.metadata["chunking_strategy"] == "recursive" for chunk in chunks)
        assert all(chunk.tenant_id == "org_001" for chunk in chunks)


class TestParseTextFile:
    @pytest.mark.asyncio
    async def test_parse_text_file(self, agent, tmp_path):
        test_file = tmp_path / "test.txt"
        test_file.write_text("测试内容", encoding="utf-8")
        chunks = await agent.parse(str(test_file), tenant_id="org_001")
        assert len(chunks) > 0
        assert "测试内容" in chunks[0].content
        assert chunks[0].tenant_id == "org_001"

    @pytest.mark.asyncio
    async def test_parse_empty_tenant(self, agent, tmp_path):
        test_file = tmp_path / "test.txt"
        test_file.write_text("内容", encoding="utf-8")
        chunks = await agent.parse(str(test_file))
        assert chunks[0].tenant_id == ""


class TestDocumentChunkModel:
    def test_chunk_id_format(self):
        chunk = DocumentChunk(
            content="内容",
            doc_id="abc123",
            chunk_index=5,
            doc_type=DocType.PDF,
        )
        assert chunk.chunk_id == "abc123#chunk-5"

    def test_default_tenant_id_empty(self):
        chunk = DocumentChunk(
            content="x",
            doc_id="d1",
            chunk_index=0,
            doc_type=DocType.TEXT,
        )
        assert chunk.tenant_id == ""


class TestParseCSV:
    def test_parse_csv_basic(self, agent, tmp_path):
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("name,age\nAlice,30\nBob,25", encoding="utf-8")
        texts = agent._parse_csv(str(csv_file))
        assert len(texts) > 0
        assert "Alice" in texts[0]
        assert "Bob" in texts[0]

    def test_parse_csv_empty(self, agent, tmp_path):
        csv_file = tmp_path / "empty.csv"
        csv_file.write_text("name,age\n", encoding="utf-8")
        texts = agent._parse_csv(str(csv_file))
        assert texts == ["[空 CSV]"]


class TestParseBatch:
    @pytest.mark.asyncio
    async def test_parse_batch_multiple_files(self, agent, tmp_path):
        f1 = tmp_path / "a.txt"
        f1.write_text("文件A", encoding="utf-8")
        f2 = tmp_path / "b.txt"
        f2.write_text("文件B", encoding="utf-8")
        chunks = await agent.parse_batch([str(f1), str(f2)], tenant_id="org_001")
        assert len(chunks) >= 2
        assert all(c.tenant_id == "org_001" for c in chunks)


class TestChunkingSettings:
    def test_chunking_defaults_are_recursive_and_bounded(self):
        """Default settings select recursive chunking and documented bounds."""
        configured = Settings(_env_file=None)

        assert configured.chunking_strategy == "recursive"
        assert configured.embedding_batch_size == 10
        assert configured.chunk_size == 400
        assert configured.chunk_overlap == 128
        assert configured.semantic_chunk_percentile == 95
        assert configured.semantic_chunk_min_size == 200
        assert configured.semantic_chunk_max_size == 512

    @pytest.mark.parametrize("strategy", ["fixed", "unknown", ""])
    def test_invalid_chunking_strategy_is_rejected(self, strategy):
        """Only recursive and semantic strategies are accepted."""
        with pytest.raises(ValueError):
            Settings(_env_file=None, chunking_strategy=strategy)

    @pytest.mark.parametrize(
        "overrides",
        [
            {"chunk_size": 100, "chunk_overlap": 100},
            {"semantic_chunk_min_size": 513, "semantic_chunk_max_size": 512},
        ],
    )
    def test_invalid_chunking_bounds_are_rejected(self, overrides):
        """Overlaps and semantic min/max values must form legal ranges."""
        with pytest.raises(ValueError):
            Settings(_env_file=None, **overrides)
