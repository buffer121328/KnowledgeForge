"""基于 LangGraph 的问答工作流。"""

from __future__ import annotations

import time
from typing import TypedDict

from langgraph.graph import END, StateGraph

from agents.qa_agent import QAAgent
from domain.knowledge import QAResult
from infrastructure.cache.semantic import SemanticConfirmationRequired


class QAState(TypedDict, total=False):
    """问答工作流的状态字典（total=False，所有键均可选）。"""

    question: str  # 用户原始问题
    tenant_id: str | None  # 租户标识；None 表示未限定租户
    user_id: str  # 发起用户标识
    retrieval_mode: str  # 检索模式（如 hybrid/vector）
    visible_department_ids: tuple[str, ...] | None  # 可见部门范围；None 表示不做部门过滤
    semantic_bypass_token: str | None  # 语义缓存绕过令牌；消费成功后强制走完整 RAG
    semantic_confirmation_token: str | None  # 语义缓存确认令牌；用于确认待确认答案，无效则回退完整 RAG
    result: QAResult | None  # 问答结果；语义缓存要求确认时为空
    confirmation: SemanticConfirmationRequired | None  # 语义缓存要求人工确认时返回的确认载荷
    cache_hit_type: str  # 缓存命中类型（来自问答结果）
    knowledge_revision: int  # 答案所基于的知识版本号
    duration_ms: int  # 本次问答耗时（毫秒）


def build_ask_question_workflow(qa_agent: QAAgent) -> StateGraph:
    """构建并编译单节点的“提问-回答”工作流图。

    Args:
        qa_agent: 负责检索与答案生成的 QA Agent。
    """
    async def process_question(state: QAState) -> dict:
        """执行一次问答并返回状态更新。

        Args:
            state: 当前工作流状态。
        """
        started = time.monotonic()
        # ① 归一化租户标识：空串与 None 统一为 None（未限定租户）
        tenant_id = state.get("tenant_id") or None
        if tenant_id == "":
            tenant_id = None
        # ② 调用 QA Agent 完成检索与答案生成；retrieval_mode 缺省为 hybrid
        outcome = await qa_agent.answer(
                state.get("question", ""),
                tenant_id=tenant_id,
                user_id=state.get("user_id") or "",
                retrieval_mode=state.get("retrieval_mode") or "hybrid",
                visible_department_ids=state.get("visible_department_ids"),
                semantic_bypass_token=state.get("semantic_bypass_token"),
                semantic_confirmation_token=state.get("semantic_confirmation_token"),
            )
        duration_ms = int((time.monotonic() - started) * 1000)
        # ③ 语义缓存要求确认时不产出结果，只回写确认载荷与耗时
        if isinstance(outcome, SemanticConfirmationRequired):
            return {"confirmation": outcome, "duration_ms": duration_ms}
        # ④ 正常结果：回写结果、缓存命中类型、知识版本与耗时
        return {
            "result": outcome,
            "cache_hit_type": outcome.cache_hit_type,
            "knowledge_revision": outcome.knowledge_revision,
            "duration_ms": duration_ms,
        }

    # 组装单节点图：入口 answer -> END
    graph = StateGraph(QAState)
    graph.add_node("answer", process_question)
    graph.set_entry_point("answer")
    graph.add_edge("answer", END)
    return graph.compile()
