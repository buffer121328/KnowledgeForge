"""KnowledgeExtractAgent 单元测试

覆盖:
  - _parse_response: JSON 解析（正常/代码块/无效JSON）
  - _deduplicate: 跨 chunk 实体/关系去重
  - extract_single: 单文本抽取
  - extract: 批量抽取（mock LLM）
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agents.knowledge_extractor import KnowledgeExtractAgent
from domain.documents import DocType, DocumentChunk
from domain.knowledge import Entity, ExtractionResult, KnowledgeEvent, Relation


@pytest.fixture
def agent():
    """跳过 ChatOpenAI 真实初始化"""
    with patch("agents.knowledge_extractor.ChatOpenAI"):
        return KnowledgeExtractAgent()


class TestEntityModel:
    def test_node_label_replaces_spaces(self):
        e = Entity(name="AI", type="Machine Learning")
        assert e.node_label == "Machine_Learning"

    def test_default_description_empty(self):
        e = Entity(name="x", type="Concept")
        assert e.description == ""

    def test_default_properties_empty_dict(self):
        e = Entity(name="x", type="Concept")
        assert e.properties == {}


class TestRelationModel:
    def test_default_confidence_zero(self):
        r = Relation(head="a", relation="r", tail="b")
        assert r.confidence == 0.0

    def test_default_properties_empty(self):
        r = Relation(head="a", relation="r", tail="b")
        assert r.properties == {}


class TestKnowledgeEventModel:
    def test_default_participants_empty(self):
        ev = KnowledgeEvent(trigger="t", type="event")
        assert ev.participants == []


class TestParseResponse:
    def test_parse_valid_json(self, agent):
        raw = """
        {
          "entities": [{"name": "腾讯", "type": "Organization", "description": "科技公司"}],
          "relations": [{"head": "腾讯", "relation": "developed_by", "tail": "微信", "confidence": 0.9}],
          "events": [{"trigger": "发布", "type": "product_launch", "participants": ["腾讯"]}]
        }
        """
        result = agent._parse_response(raw, source_id="src_001")
        assert len(result.entities) == 1
        assert result.entities[0].name == "腾讯"
        assert result.entities[0].type == "Organization"
        assert len(result.relations) == 1
        assert result.relations[0].confidence == 0.9
        assert len(result.events) == 1
        assert result.events[0].participants == ["腾讯"]
        assert result.source_chunk_id == "src_001"

    def test_parse_json_in_code_block(self, agent):
        raw = '```json\n{"entities": [], "relations": [], "events": []}\n```'
        result = agent._parse_response(raw, source_id="src_002")
        assert result.entities == []
        assert result.relations == []
        assert result.events == []
        assert result.source_chunk_id == "src_002"

    def test_parse_invalid_json_returns_empty(self, agent):
        result = agent._parse_response("not a json", source_id="src_003")
        assert result.entities == []
        assert result.relations == []
        assert result.events == []

    def test_parse_skips_entities_without_name(self, agent):
        raw = '{"entities": [{"name": "", "type": "Person"}, {"name": "张三", "type": "Person"}], "relations": [], "events": []}'
        result = agent._parse_response(raw, source_id="")
        assert len(result.entities) == 1
        assert result.entities[0].name == "张三"

    def test_parse_skips_relations_missing_head_or_tail(self, agent):
        raw = """{
          "entities": [],
          "relations": [
            {"head": "", "relation": "r", "tail": "b"},
            {"head": "a", "relation": "r", "tail": ""},
            {"head": "a", "relation": "r", "tail": "b"}
          ],
          "events": []
        }"""
        result = agent._parse_response(raw, source_id="")
        assert len(result.relations) == 1

    def test_parse_default_confidence_when_missing(self, agent):
        raw = '{"entities": [], "relations": [{"head": "a", "relation": "r", "tail": "b"}], "events": []}'
        result = agent._parse_response(raw, source_id="")
        assert result.relations[0].confidence == 0.5

    def test_parse_default_type_when_missing(self, agent):
        raw = '{"entities": [{"name": "x"}], "relations": [], "events": []}'
        result = agent._parse_response(raw, source_id="")
        assert result.entities[0].type == "Concept"


class TestDeduplicate:
    def test_dedup_entities_same_name_and_type(self, agent):
        """同名同类型实体跨 chunk 去重"""
        r1 = ExtractionResult(
            entities=[Entity(name="腾讯", type="Organization")],
            relations=[],
            events=[],
            source_chunk_id="c1",
        )
        r2 = ExtractionResult(
            entities=[Entity(name="腾讯", type="Organization")],
            relations=[],
            events=[],
            source_chunk_id="c2",
        )
        deduped = agent._deduplicate([r1, r2])
        assert len(deduped[0].entities) == 1
        assert len(deduped[1].entities) == 0

    def test_dedup_entities_same_name_different_type_kept(self, agent):
        """同名不同类型实体保留"""
        r1 = ExtractionResult(
            entities=[Entity(name="苹果", type="Organization")],
            relations=[],
            events=[],
            source_chunk_id="c1",
        )
        r2 = ExtractionResult(
            entities=[Entity(name="苹果", type="Product")],
            relations=[],
            events=[],
            source_chunk_id="c2",
        )
        deduped = agent._deduplicate([r1, r2])
        assert len(deduped[0].entities) == 1
        assert len(deduped[1].entities) == 1

    def test_dedup_relations_same_triple_across_chunks_preserves_evidence_occurrences(self, agent):
        """相同三元组跨 chunk 保留，以便分别持久化 Evidence。"""
        r1 = ExtractionResult(
            entities=[],
            relations=[Relation(head="a", relation="r", tail="b")],
            events=[],
            source_chunk_id="c1",
        )
        r2 = ExtractionResult(
            entities=[],
            relations=[Relation(head="a", relation="r", tail="b")],
            events=[],
            source_chunk_id="c2",
        )
        deduped = agent._deduplicate([r1, r2])
        assert len(deduped[0].relations) == 1
        assert len(deduped[1].relations) == 1

    def test_dedup_relations_repeated_inside_one_chunk(self, agent):
        """同一 chunk 内完全重复的关系只保留一次。"""
        relation = Relation(head="a", relation="r", tail="b")
        result = ExtractionResult(
            entities=[],
            relations=[relation, relation],
            events=[],
            source_chunk_id="c1",
        )

        deduped = agent._deduplicate([result])

        assert len(deduped[0].relations) == 1

    def test_dedup_preserves_events(self, agent):
        r1 = ExtractionResult(
            entities=[],
            relations=[],
            events=[KnowledgeEvent(trigger="t1", type="e1")],
            source_chunk_id="c1",
        )
        deduped = agent._deduplicate([r1])
        assert len(deduped[0].events) == 1

    def test_dedup_empty_list(self, agent):
        assert agent._deduplicate([]) == []


class TestExtractSingle:
    @pytest.mark.asyncio
    async def test_extract_single_returns_result(self, agent):
        mock_resp = MagicMock()
        mock_resp.content = '{"entities": [{"name": "x", "type": "Concept"}], "relations": [], "events": []}'
        agent.llm = AsyncMock()
        agent.llm.ainvoke.return_value = mock_resp

        result = await agent.extract_single("some text", chunk_id="chunk_001")
        assert len(result.entities) == 1
        assert result.source_chunk_id == "chunk_001"

    @pytest.mark.asyncio
    async def test_extract_single_invalid_response(self, agent):
        mock_resp = MagicMock()
        mock_resp.content = "not json"
        agent.llm = AsyncMock()
        agent.llm.ainvoke.return_value = mock_resp

        result = await agent.extract_single("text", chunk_id="chunk_002")
        assert result.entities == []
        assert result.relations == []


class TestExtractBatch:
    @pytest.mark.asyncio
    async def test_extract_deduplicates_across_chunks(self, agent):
        chunks = [
            DocumentChunk(content="text1", doc_id="d1", chunk_index=0, doc_type=DocType.TEXT),
            DocumentChunk(content="text2", doc_id="d1", chunk_index=1, doc_type=DocType.TEXT),
        ]
        mock_resp = MagicMock()
        mock_resp.content = '{"entities": [{"name": "腾讯", "type": "Organization"}], "relations": [], "events": []}'
        agent.llm = AsyncMock()
        agent.llm.ainvoke.return_value = mock_resp

        results = await agent.extract(chunks)
        assert len(results) == 2
        assert len(results[0].entities) == 1
        assert len(results[1].entities) == 0  # 被去重

    @pytest.mark.asyncio
    async def test_extract_empty_chunks(self, agent):
        results = await agent.extract([])
        assert results == []
