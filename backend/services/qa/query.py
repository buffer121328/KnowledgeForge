"""Local QA intent classification and query rewriting helpers."""

from __future__ import annotations

import re

from domain.knowledge import QueryIntent

INTENT_RULES: list[tuple[QueryIntent, tuple[str, ...]]] = [
    (QueryIntent.COMPARATIVE, ("区别", "对比", "相比", "不同", "vs", "versus")),
    (QueryIntent.PROCEDURAL, ("怎么做", "如何", "步骤", "流程", "怎样")),
    (QueryIntent.ANALYTICAL, ("为什么", "原因", "怎么理解", "为何")),
    (QueryIntent.EXPLORATORY, ("有哪些", "概述", "介绍一下", "列出", "哪些")),
    (QueryIntent.FACTOID, ("是什么", "谁是", "哪里", "何时", "多少")),
]


def classify_intent_local(question: str) -> QueryIntent:
    """Classify the intent local."""
    question_lower = question.lower()
    for intent, keywords in INTENT_RULES:
        if any(keyword in question_lower for keyword in keywords):
            return intent
    return QueryIntent.FACTOID


def rewrite_query_local(question: str) -> dict:
    """Return the existing lightweight query/query-entity representation.

    多主题问题（顿号枚举多个主题再追问）的原句向量会被主题混杂稀释，
    次要主题的相关文档可能落到 top 20-50 之外。这里为多主题问题额外生成
    每主题一条子查询，供向量检索分头召回后由 RRF 统一融合。
    """
    tokens = [token for token in re.split(r"[\s，。？?！!、；;：:（）()【】\[\]\"']+", question) if token]
    stop_words = {"是什么", "什么", "怎么", "如何", "为什么", "哪些", "有没有", "吗", "呢", "的", "了", "和", "与"}
    entities = [token for token in tokens if len(token) >= 2 and token not in stop_words][:5]
    queries = [question]
    if entities:
        queries.append(" ".join(entities))
    # 多主题拆分：仅针对"顿号枚举主题"的句式（如"招聘、培训和绩效考核…"）。
    # 逐段收集主题词：非末段若含疑问谓语则整个停止；末段先剥离谓语部分
    # （首个疑问词起）再取主题。每个主题生成"主题 + 制度"子查询。
    if "、" in question:
        segments = [segment.strip() for segment in question.split("、")]
        topics: list[str] = []
        for index, segment in enumerate(segments):
            if any(
                marker in segment
                for marker in ("什么", "哪些", "如何", "怎么", "为什么", "能否", "是否", "何时")
            ):
                # 谓语段：剥离谓语，只留疑问词前的主题部分；随后整体结束
                match = re.search(r"(?:能否|如何|哪些|什么|怎么|是否|何时|为什么)", segment)
                segment = segment[: match.start()] if match else ""
                has_predicate = True
            else:
                has_predicate = False
            for token in re.split(r"[和与]", segment):
                token = token.strip(" ，,。？?！!；;：:")
                if (
                    len(token) >= 2
                    and len(token) <= 12
                    and token not in topics
                ):
                    topics.append(token)
            if has_predicate:
                break
        for topic in topics[:3]:
            candidate = f"{topic} 制度要求"
            if candidate not in queries:
                queries.append(candidate)
    return {"queries": queries[:4], "entities": entities, "keywords": entities}
