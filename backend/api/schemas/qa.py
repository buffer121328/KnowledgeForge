"""API qa 域共享的请求与响应模型（自 api/schemas.py 拆分）。"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from api.contracts import ResourceId, StrictRequestModel


class QuestionRequest(StrictRequestModel):
    """问答提问请求体。"""
    question: str = Field(min_length=1, max_length=10_000)  # 用户问题文本，1~10000 字符
    conversation_id: ResourceId | None = None  # 所属会话 ID；为空表示新会话
    retrieval_mode: Literal["vector", "hybrid"] = "hybrid"  # 检索模式：纯向量或混合检索
    semantic_bypass_token: str | None = Field(default=None, max_length=512)  # 语义缓存拒绝后签发的一次性绕过令牌
    semantic_confirmation_token: str | None = Field(default=None, max_length=512)  # 语义缓存复用确认令牌


# 问答细分响应状态取值集合（已回答/部分回答/证据不足/需澄清/证据冲突/需人工复核/来源不可用）
QAResponseStatusValue = Literal[
    "answered",
    "partially_answered",
    "insufficient_evidence",
    "needs_clarification",
    "conflicting_evidence",
    "human_review_required",
    "source_unavailable",
]


class QAAnswerClaimResponse(BaseModel):
    """单条有依据的答案断言及其请求内引用。"""

    claim_id: str  # 断言 ID
    text: str  # 断言文本
    citation_ids: list[str] = Field(default_factory=list)  # 支撑该断言的引用 ID 列表
    material: bool = True  # 是否为实质性内容


class QAAnswerCitationResponse(BaseModel):
    """单条已授权引用，不暴露内部上下文 ID。"""

    citation_id: str  # 引用 ID
    source: str = ""  # 来源标识
    content: str = ""  # 引用内容文本
    document_id: str = ""  # 来源文档 ID
    chunk_id: str = ""  # 来源分块 ID
    chunk_index: int | None = None  # 分块在文档中的序号
    highlight: str = ""  # 高亮片段文本


class QAMissingInformationResponse(BaseModel):
    """完成任务所需的有限、可由用户补充的信息。"""

    field: str  # 缺失字段名
    description: str  # 缺失内容说明


class QAGroundingResultResponse(BaseModel):
    """有限的依据校验诊断信息，不含断言或来源正文。"""

    passed: bool  # 校验是否通过
    accepted_claim_ids: list[str] = Field(default_factory=list)  # 通过校验的断言 ID 列表
    rejected_claim_ids: list[str] = Field(default_factory=list)  # 未通过校验的断言 ID 列表
    reason_codes: list[str] = Field(default_factory=list)  # 拒绝原因码列表
    policy_version: str  # 校验策略版本


class QuestionResponse(BaseModel):
    """问答响应体。"""
    status: Literal["answered"] = "answered"  # 响应状态，固定为已回答
    question: str  # 原始问题文本
    answer: str  # 生成的答案
    confidence: float  # 答案置信度
    intent: str  # 识别到的问题意图
    sources: list[dict[str, Any]]  # 引用来源列表
    reasoning_steps: list[str]  # 推理步骤说明列表
    degradation_code: str | None = None  # 降级原因码
    conversation_id: str | None = None  # 所属会话 ID
    qa_run_id: str | None = None  # 本次问答运行 ID
    cache_hit_type: str = "none"  # 缓存命中类型，默认未命中
    history_saved: bool = False  # 是否已保存历史记录
    warning_code: str | None = None  # 警告码
    response_status: QAResponseStatusValue | None = None  # 细分响应状态
    evidence_state: str | None = None  # 证据状态
    evidence_reason_codes: list[str] = Field(default_factory=list)  # 证据判定原因码列表
    claims: list[QAAnswerClaimResponse] = Field(default_factory=list)  # 答案断言列表
    citations: list[QAAnswerCitationResponse] = Field(default_factory=list)  # 引用列表
    missing_information: list[QAMissingInformationResponse] = Field(default_factory=list)  # 缺失信息列表
    grounding_result: QAGroundingResultResponse | None = None  # 依据校验结果
    grounding_passed: bool | None = None  # 依据校验是否通过
    policy_version: str | None = None  # 策略版本


class SemanticConfirmationResponse(BaseModel):
    """在用户显式确认前仅披露安全的候选元数据。"""

    status: Literal["semantic_confirmation_required"] = "semantic_confirmation_required"  # 固定为需要语义确认
    similar_question: str  # 相似的历史问题文本
    similarity: float  # 相似度得分
    cached_at: str  # 缓存时间
    confirmation_token: str  # 确认令牌


class SemanticCacheDecisionRequest(StrictRequestModel):
    """将语义缓存决策绑定到精确的原始请求作用域。"""

    question: str = Field(min_length=1, max_length=10_000)  # 原始问题文本
    confirmation_token: str = Field(min_length=1, max_length=512)  # 待使用的确认令牌
    conversation_id: ResourceId | None = None  # 所属会话 ID
    retrieval_mode: Literal["vector", "hybrid"] = "hybrid"  # 检索模式


class SemanticCacheRejectResponse(BaseModel):
    """返回一次性令牌，使下一次请求强制走完整 RAG 流程。"""

    status: Literal["semantic_rejected"] = "semantic_rejected"  # 固定为语义缓存已拒绝
    semantic_bypass_token: str | None = None  # 一次性绕过令牌


class QAConversationCreateRequest(StrictRequestModel):
    """显式创建空问答会话的请求体。"""

    title: str = Field(default="", max_length=255)  # 会话标题


class QAConversationItem(BaseModel):
    """单条问答会话摘要。"""

    id: str  # 会话 ID
    title: str  # 会话标题
    status: str  # 会话状态
    created_at: str  # 创建时间
    updated_at: str  # 更新时间


class QAConversationPageResponse(BaseModel):
    """单页游标分页的问答会话列表。"""

    items: list[QAConversationItem]  # 当前页会话项
    next_cursor: str | None = None  # 下一页游标；为空表示没有更多数据


class QAConversationDetailResponse(BaseModel):
    """属于当前用户的会话详情，含有序消息与运行记录。"""

    conversation: QAConversationItem  # 会话摘要
    messages: list[dict[str, Any]]  # 有序消息列表
    runs: list[dict[str, Any]]  # 问答运行记录列表


class QAFeedbackRequest(StrictRequestModel):
    """对一次问答运行的受限用户反馈。"""

    rating: Literal["up", "down", "issue"]  # 反馈类型：赞同/反对/问题
    note: str = Field(default="", max_length=1000)  # 补充说明


class QAConversationDeletedResponse(BaseModel):
    """确认删除一条属于自己的问答会话。"""

    status: Literal["deleted"] = "deleted"  # 固定为已删除
    conversation_id: ResourceId  # 被删除的会话 ID


class QAFeedbackResponse(BaseModel):
    """返回调用者已保存的反馈记录。"""

    feedback_id: ResourceId  # 反馈记录 ID
    qa_run_id: ResourceId  # 关联的问答运行 ID
    rating: Literal["up", "down", "issue"]  # 反馈类型
    note: str  # 补充说明

