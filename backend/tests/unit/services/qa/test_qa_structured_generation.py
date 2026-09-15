"""ATDD contracts for bounded structured QA generation."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from services.qa.generation import StructuredGenerationError, generate_structured_answer
from domain.evidence import QAResponseStatus
from domain.knowledge import QueryIntent, RetrievedContext


def context(
    content: str,
    *,
    context_id: str = "ctx_1",
    source: str = "policy.md",
) -> RetrievedContext:
    return RetrievedContext(
        content=content,
        source=source,
        score=0.9,
        retrieval_type="vector",
        metadata={"context_id": context_id, "tenant_id": "org-001"},
    )


def payload(
    *,
    status: str = "answered",
    answer: str = "预算为100万元。",
    claims: list[dict] | None = None,
    missing_information: list[dict] | None = None,
) -> str:
    return json.dumps(
        {
            "status": status,
            "answer": answer,
            "claims": claims
            if claims is not None
            else [
                {
                    "claim_id": "claim_1",
                    "text": answer,
                    "citation_ids": ["cite_current_1"],
                    "material": True,
                }
            ],
            "missing_information": missing_information or [],
            "schema_version": "structured-answer-v1",
        },
        ensure_ascii=False,
    )


@pytest.mark.asyncio
async def test_structured_generation_hydrates_authoritative_current_run_citations() -> None:
    llm = SimpleNamespace(ainvoke=AsyncMock(return_value=SimpleNamespace(content=payload())))
    evidence = context("培训预算为100万元。")

    result = await generate_structured_answer(
        llm,
        "培训预算是多少？",
        [evidence],
        QueryIntent.FACTOID,
        schema_version="structured-answer-v1",
        citation_id_factory=lambda _: "cite_current_1",
    )

    assert result.answer.status is QAResponseStatus.ANSWERED
    assert result.answer.answer == "预算为100万元。"
    assert result.answer.citations[0].citation_id == "cite_current_1"
    assert result.answer.citations[0].context_id == "ctx_1"
    assert result.answer.citations[0].source == "policy.md"
    assert result.answer.citations[0].content == "培训预算为100万元。"
    assert result.authorized_contexts == {"ctx_1": evidence}
    assert result.attempt_count == 1

    messages = llm.ainvoke.await_args.args[0]
    prompt = "\n".join(str(message.content) for message in messages)
    assert "cite_current_1" in prompt
    assert "ctx_1" not in prompt


@pytest.mark.asyncio
async def test_structured_generation_accepts_partial_answer_with_missing_information() -> None:
    llm = SimpleNamespace(
        ainvoke=AsyncMock(
            return_value=SimpleNamespace(
                content=payload(
                    status="partially_answered",
                    answer="预算为100万元。",
                    missing_information=[
                        {"field": "effective_date", "description": "缺少生效日期。"}
                    ],
                )
            )
        )
    )

    result = await generate_structured_answer(
        llm,
        "预算和生效日期是什么？",
        [context("预算为100万元。")],
        QueryIntent.FACTOID,
        schema_version="structured-answer-v1",
        citation_id_factory=lambda _: "cite_current_1",
    )

    assert result.answer.status is QAResponseStatus.PARTIALLY_ANSWERED
    assert result.answer.missing_information[0].field == "effective_date"


@pytest.mark.asyncio
async def test_structured_generation_discloses_allowlisted_retrieval_degradation() -> None:
    llm = SimpleNamespace(ainvoke=AsyncMock(return_value=SimpleNamespace(content=payload())))

    await generate_structured_answer(
        llm,
        "印章管理的主要要求是什么？",
        [context("行政管理部保管行政章和合同专用章。")],
        QueryIntent.FACTOID,
        schema_version="structured-answer-v1",
        citation_id_factory=lambda _: "cite_current_1",
        degradation_code="bm25_retrieval_unavailable",
    )

    prompt = "\n".join(str(message.content) for message in llm.ainvoke.await_args.args[0])
    assert "BM25 检索分支不可用" in prompt
    assert "省略已被证据支持的事实" in prompt


@pytest.mark.asyncio
async def test_structured_generation_accepts_markdown_fenced_valid_payload() -> None:
    """部分模型即使在纯 JSON 要求下也会用 ```json 围栏包裹输出。

    契约：剥离围栏外包装后内容合规的，必须照常通过；围栏内不是严格
    合法 JSON 的，仍必须被拒绝。
    """
    llm = SimpleNamespace(
        ainvoke=AsyncMock(
            return_value=SimpleNamespace(
                content=f"```json\n{payload()}\n```",
            )
        )
    )

    result = await generate_structured_answer(
        llm,
        "培训预算是多少？",
        [context("培训预算为100万元。")],
        QueryIntent.FACTOID,
        schema_version="structured-answer-v1",
        citation_id_factory=lambda _: "cite_current_1",
    )

    assert result.attempt_count == 1
    assert result.answer.answer == "预算为100万元。"


@pytest.mark.asyncio
async def test_structured_generation_still_rejects_fenced_non_json_payload() -> None:
    llm = SimpleNamespace(
        ainvoke=AsyncMock(
            return_value=SimpleNamespace(
                content="```json\n这不是 JSON\n```",
            )
        )
    )

    with pytest.raises(StructuredGenerationError):
        await generate_structured_answer(
            llm,
            "培训预算是多少？",
            [context("培训预算为100万元。")],
            QueryIntent.FACTOID,
            schema_version="structured-answer-v1",
            citation_id_factory=lambda _: "cite_current_1",
        )


@pytest.mark.asyncio
async def test_structured_generation_retries_once_after_schema_parse_failure() -> None:
    llm = SimpleNamespace(
        ainvoke=AsyncMock(
            side_effect=[
                SimpleNamespace(content="not-json"),
                SimpleNamespace(content=payload()),
            ]
        )
    )

    result = await generate_structured_answer(
        llm,
        "培训预算是多少？",
        [context("培训预算为100万元。")],
        QueryIntent.FACTOID,
        schema_version="structured-answer-v1",
        citation_id_factory=lambda _: "cite_current_1",
    )

    assert result.attempt_count == 2
    assert llm.ainvoke.await_count == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "invalid_claims",
    [
        [
            {
                "claim_id": "claim_1",
                "text": "预算为100万元。",
                "citation_ids": [],
                "material": True,
            }
        ],
        [
            {
                "claim_id": "claim_1",
                "text": "预算为100万元。",
                "citation_ids": ["cite_stale_or_unknown"],
                "material": True,
            }
        ],
    ],
    ids=["missing-citation", "unknown-or-stale-citation"],
)
async def test_structured_generation_rejects_missing_or_unknown_citations(
    invalid_claims: list[dict],
) -> None:
    invalid = payload(claims=invalid_claims)
    llm = SimpleNamespace(
        ainvoke=AsyncMock(
            side_effect=[SimpleNamespace(content=invalid), SimpleNamespace(content=invalid)]
        )
    )

    with pytest.raises(StructuredGenerationError) as captured:
        await generate_structured_answer(
            llm,
            "培训预算是多少？",
            [context("培训预算为100万元。")],
            QueryIntent.FACTOID,
            schema_version="structured-answer-v1",
            citation_id_factory=lambda _: "cite_current_1",
        )

    assert captured.value.reason_code == "structured_output_invalid"
    assert invalid not in str(captured.value)
    assert llm.ainvoke.await_count == 2
