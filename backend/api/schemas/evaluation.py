"""API evaluation 域共享的请求与响应模型（自 api/schemas.py 拆分）。"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from api.contracts import ResourceId, StrictRequestModel


class EvaluationRunSummary(BaseModel):
    """管理面板中列出的单次评测运行。"""
    run_id: str  # 运行 ID
    started_at: str | None = None  # 开始时间
    incomplete: bool = False  # 运行是否未完成
    retrieval_modes: list[str] = Field(default_factory=list)  # 覆盖的检索模式列表
    run_classification: str | None = None  # 运行分类
    record_count: int = 0  # 记录总数
    completed: int = 0  # 完成数
    failed: int = 0  # 失败数
    invalid_provenance: int = 0  # 溯源无效的记录数
    skipped_lines: int = 0  # 跳过的行数


class EvaluationRunListResponse(BaseModel):
    """有限的运行列表，按时间倒序。"""
    runs: list[EvaluationRunSummary] = Field(default_factory=list)  # 运行摘要列表


class EvaluationContextSummary(BaseModel):
    """单条检索上下文摘要，不含完整正文。"""
    rank: int | None = None  # 检索排名
    source: str | None = None  # 来源标识
    score: float | None = None  # 检索得分
    retrieval_type: str | None = None  # 检索类型
    source_document_id: str | None = None  # 来源文档 ID


class EvaluationRecordItem(BaseModel):
    """单条按问题维度的评测记录，含截断后的回答与上下文摘要。"""
    benchmark_id: str | None = None  # 基准条目 ID
    retrieval_mode: str | None = None  # 检索模式
    category: str | None = None  # 问题分类
    status: str | None = None  # 记录状态
    exception: str | None = None  # 异常信息
    refused: bool | None = None  # 模型是否拒绝回答
    latency_ms: float | None = None  # 延迟（毫秒）
    question: str | None = None  # 问题文本
    response: str | None = None  # 截断后的回答文本
    context_count: int = 0  # 检索上下文数
    contexts: list[EvaluationContextSummary] = Field(default_factory=list)  # 上下文摘要列表
    retrieved_context_ids: list[str] = Field(default_factory=list)  # 实际检索到的上下文 ID 列表
    reference_context_ids: list[str] = Field(default_factory=list)  # 基准参考上下文 ID 列表
    ragas_scores: dict[str, float] = Field(default_factory=dict)  # Ragas 评分
    # 逐指标评分状态：scored / not_applicable（按评审契约不适用）/ failed + 原因码
    ragas_metric_states: dict[str, dict[str, str | None]] = Field(default_factory=dict)


class EvaluationRecordsResponse(BaseModel):
    """单页按问题维度的评测记录。"""
    run_id: str  # 运行 ID
    page: int  # 页码
    page_size: int  # 每页数量
    total: int  # 记录总数
    skipped_lines: int = 0  # 跳过的行数
    records: list[EvaluationRecordItem] = Field(default_factory=list)  # 当前页记录列表


class EvaluationRunDetail(BaseModel):
    """单次运行的元数据、质量报告与 Ragas 评分汇总。"""
    run_id: str  # 运行 ID
    incomplete: bool  # 是否未完成
    metadata: dict[str, Any] = Field(default_factory=dict)  # 运行元数据
    quality_report: dict[str, Any] | None = None  # 质量报告
    ragas_summaries: dict[str, Any] = Field(default_factory=dict)  # Ragas 评分汇总


class EvaluationReportMetricEntry(BaseModel):
    """报告中的单个指标条目（率型指标为百分比值）。"""
    key: str  # 指标稳定 key
    kind: str  # ragas / evidence / retrieval / safety 分层
    unit: str  # score（0-1）或 percent
    value: float | None = None  # 指标值（率已转百分比）
    scored: int | None = None  # 已评样本数
    total: int | None = None  # 适用的样本总数
    failed: int | None = None  # 评分失败数
    direction: str | None = None  # safety 类为 lower_is_better


class EvaluationReportAdvisory(BaseModel):
    """报告注意事项条目。"""
    level: str  # info / warning / critical
    message: str  # 说明文本


class EvaluationRunReportResponse(BaseModel):
    """单次运行的综合评测报告（RAGAS + 自定义指标 + 契约）。"""
    run_id: str  # 运行 ID
    incomplete: bool  # 是否未完成
    run_classification: str | None = None  # 冒烟/正式
    counts: dict[str, Any] = Field(default_factory=dict)  # 执行计数
    sections: dict[str, list[EvaluationReportMetricEntry]] = Field(default_factory=dict)  # 分层指标
    advisories: list[EvaluationReportAdvisory] = Field(default_factory=list)  # 注意事项


class EvaluationSpotCheckItem(BaseModel):
    """单条异常样本及其人工抽检状态。"""
    benchmark_id: str  # 基准条目 ID
    retrieval_mode: str | None = None  # 检索模式
    category: str | None = None  # 问题分类
    anomaly_codes: list[str] = Field(default_factory=list)  # 命中的异常类型
    status: str | None = None  # 记录状态
    exception: str | None = None  # 异常信息
    expected_response_status: str | None = None  # 预期响应状态
    observed_response_status: str | None = None  # 实际响应状态
    expected_evidence_states: list[str] | None = None  # 预期证据状态
    observed_evidence_state: str | None = None  # 实际证据状态
    expected_evidence_context_ids: list[str] = Field(default_factory=list)  # 期望证据 ID
    retrieved_context_ids: list[str] = Field(default_factory=list)  # 实际检索 ID
    question: str | None = None  # 问题文本
    response: str | None = None  # 截断后的回答文本
    ragas_scores: dict[str, float] = Field(default_factory=dict)  # Ragas 评分
    spot_check: dict[str, Any] | None = None  # 已有抽检结论（verdict/note/时间）


class EvaluationSpotCheckResponse(BaseModel):
    """单次运行的异常样本清单。"""
    run_id: str  # 运行 ID
    total_records: int  # 记录总数
    skipped_lines: int = 0  # 跳过行数
    anomaly_count: int  # 异常样本数
    spot_checked: int  # 已完成人工抽检数
    low_score_threshold: float  # 低分阈值
    anomalies: list[EvaluationSpotCheckItem] = Field(default_factory=list)  # 异常清单


class EvaluationSpotCheckSubmitRequest(StrictRequestModel):
    """提交一条人工抽检结论的请求体。"""
    benchmark_id: str = Field(min_length=1, max_length=200)  # 基准条目 ID
    verdict: str = Field(min_length=1, max_length=32)  # 结论：judge_error/system_issue/confirmed_ok/needs_data_fix
    note: str = Field(default="", max_length=600)  # 复核备注


class CurrentCorpusSourceReference(BaseModel):
    """真实现行语料文档的绑定元数据，不含内容与路径。"""

    document_id: str = Field(min_length=1, max_length=256)  # 绑定的文档 ID
    source: str = Field(min_length=1, max_length=256)  # 来源标识
    department: str = Field(min_length=1, max_length=128)  # 归属部门名


class CurrentCorpusDraftCandidate(BaseModel):
    """有限的现行语料对齐候选，不含来源文档内容。"""

    candidate_id: str = Field(pattern=r"^[a-z][a-z0-9_]{2,95}$")  # 候选 ID
    fixture: str = Field(min_length=1, max_length=96)  # 对应的 Fixture 名称
    source_document_ids: list[str] = Field(default_factory=list, max_length=2)  # 绑定来源文档 ID 列表（至多 2 个）
    source_references: list[CurrentCorpusSourceReference] = Field(default_factory=list, max_length=2)  # 来源引用列表（至多 2 个）
    source_document_count: int = Field(ge=0, le=2)  # 来源文档数
    alignment_status: str = Field(min_length=1, max_length=96)  # 对齐状态
    reason_code: str = Field(min_length=1, max_length=96)  # 原因码
    review_status: Literal["draft", "pending_review", "approved"]  # 审核状态
    submitted_at: str | None = None  # 提交时间
    reviewed_at: str | None = None  # 审核时间
    review_reason: str = Field(default="", max_length=1_000)  # 审核说明


class CurrentCorpusDraft(BaseModel):
    """由当前目录元数据生成、仅管理员可见的候选草稿。"""

    draft_id: str = Field(pattern=r"^ccd_[a-f0-9]{32}$")  # 草稿 ID
    status: Literal["draft", "pending_review", "approved"]  # 草稿状态
    revision: int = Field(ge=1)  # 乐观锁版本号
    created_at: str  # 创建时间
    updated_at: str  # 更新时间
    document_count: int = Field(ge=0, le=100_000)  # 快照文档数
    department_count: int = Field(ge=0, le=10_000)  # 快照部门数
    candidate_count: int = Field(ge=0, le=7)  # 候选总数
    reviewable_candidate_count: int = Field(ge=0, le=7)  # 可审核候选数
    approved_candidate_count: int = Field(ge=0, le=7)  # 已通过候选数
    authoring_dataset_id: str | None = Field(default=None, pattern=r"^[a-z0-9][a-z0-9-]{2,63}$")  # 生成的编辑数据集 ID
    current_corpus_snapshot_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")  # 目录快照 SHA-256
    authoring_dataset_created_at: str | None = None  # 编辑数据集创建时间
    counts: dict[str, int] = Field(default_factory=dict)  # 分类计数
    candidates: list[CurrentCorpusDraftCandidate] = Field(default_factory=list, max_length=7)  # 候选列表


class CurrentCorpusDraftListResponse(BaseModel):
    """有限的现行语料候选草稿历史。"""

    drafts: list[CurrentCorpusDraft] = Field(default_factory=list, max_length=20)  # 草稿列表


class EvidenceDatasetSummary(BaseModel):
    """单个编辑数据集的受限治理摘要。"""
    dataset_id: str  # 数据集 ID
    revision: int  # 工作区版本号
    status: str  # 数据集状态
    title: str | None = None  # 标题
    case_count: int  # 案例数
    context_count: int = 0  # 上下文数
    source_document_count: int = Field(default=0, ge=0, le=100_000)  # 来源文档数
    category_counts: dict[str, int] = Field(default_factory=dict)  # 分类计数
    required_category_count: int  # 必需分类数
    covered_category_count: int = 0  # 已覆盖分类数
    counts: dict[str, int] = Field(default_factory=dict)  # 其他计数
    fixture_status: str | None = None  # Fixture 状态
    last_frozen_version: str | None = None  # 最近冻结版本号
    last_frozen_manifest_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")  # 最近冻结清单 SHA-256
    last_frozen_at: str | None = None  # 最近冻结时间
    last_frozen_by: str | None = None  # 最近冻结操作人
    last_frozen_by_display_name: str | None = None  # 最近冻结操作人中文展示名
    source_frozen_version: str | None = None  # 来源冻结版本号
    source_type: str | None = Field(default=None, max_length=64)  # 数据集来源类型
    active: bool = True  # 是否为当前可运行版本
    superseded_by: str | None = Field(default=None, pattern=r"^[a-z0-9][a-z0-9-]{2,63}$")  # 替代数据集 ID
    current_corpus_draft_id: str | None = Field(default=None, pattern=r"^ccd_[a-f0-9]{32}$")  # 关联现行语料草稿 ID
    current_corpus_snapshot_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")  # 关联目录快照 SHA-256
    updated_at: str | None = None  # 更新时间


class EvidenceDatasetListResponse(BaseModel):
    """管理员可见的组织内证据数据集列表。"""
    datasets: list[EvidenceDatasetSummary] = Field(default_factory=list)  # 数据集摘要列表


class EvidenceContextSummary(BaseModel):
    """展示在证据案例旁的受限上下文元数据。"""
    context_id: str  # 上下文 ID
    source_document_id: str | None = None  # 来源文档 ID
    title: str | None = None  # 标题
    department: str | None = None  # 部门
    document_version: int | None = Field(default=None, ge=1)  # 文档版本
    chunk_index: int | None = None  # 分块序号
    content_sha256: str | None = None  # 内容 SHA-256
    content_excerpt: str = ""  # 内容摘录


class EvidenceCaseItem(BaseModel):
    """单个可编辑的证据门案例及服务端维护的审核元数据。"""
    model_config = {"extra": "allow"}  # 允许携带服务端附加字段

    id: str  # 案例 ID
    question: str  # 问题文本
    reference_answer: str = ""  # 独立审核的参考答案
    category: str  # 分类
    expected_response_status: str  # 期望的响应状态
    expected_evidence_states: list[str] = Field(default_factory=list)  # 期望的证据状态列表
    expected_reason_codes: list[str] = Field(default_factory=list)  # 期望的原因码列表
    expected_source_document_ids: list[str] = Field(default_factory=list)  # 期望的来源文档 ID 列表
    expected_evidence_context_ids: list[str] = Field(default_factory=list)  # 期望的证据上下文 ID 列表
    expected_evidence_sections: list[str] = Field(default_factory=list)  # 期望的证据章节列表
    expected_citation_context_ids: list[str] = Field(default_factory=list)  # 期望的引用上下文 ID 列表
    expected_missing_information_fields: list[str] = Field(default_factory=list)  # 期望的缺失信息字段列表
    required_fixture: str = "standard"  # 所需 Fixture
    review_status: str = "draft"  # 审核状态
    case_revision: int = 1  # 案例版本号
    last_editor_id: str | None = None  # 最近编辑人 ID
    last_editor_display_name: str | None = None  # 最近编辑人展示名
    department_id: str  # 归属部门 ID
    last_edited_at: str | None = None  # 最近编辑时间
    submitted_at: str | None = None  # 提交时间
    reviewer_id: str | None = None  # 审核人 ID
    reviewer_display_name: str | None = None  # 审核人展示名
    reviewed_at: str | None = None  # 审核时间
    review_reason: str = ""  # 审核说明
    rejection_reason: str = ""  # 拒绝原因
    context_summaries: list[EvidenceContextSummary] = Field(default_factory=list)  # 上下文摘要列表


class EvidenceCaseListResponse(BaseModel):
    """单个数据集修订的分页案例列表。"""
    dataset_id: str  # 数据集 ID
    revision: int  # 工作区版本号
    page: int  # 页码
    page_size: int  # 每页数量
    total: int  # 案例总数
    cases: list[EvidenceCaseItem] = Field(default_factory=list)  # 当前页案例列表


class EvidenceCaseMutationRequest(StrictRequestModel):
    """在期望工作区版本号下创建或整体替换一个案例。"""
    expected_revision: int = Field(ge=1)  # 期望的工作区版本号
    case: dict[str, Any]  # 案例字段字典


class EvidenceRevisionRequest(StrictRequestModel):
    """受工作区乐观锁保护的操作请求。"""
    expected_revision: int = Field(ge=1)  # 期望的工作区版本号


class EvidenceDraftFromVersionRequest(EvidenceRevisionRequest):
    """从选定不可变冻结版本派生可编辑草稿。"""

    version: str = Field(min_length=1, max_length=128)  # 来源冻结版本号


class EvidenceFrozenVersion(BaseModel):
    """安全的冻结版本标识，不含产物路径或数据集内容。"""

    version: str  # 版本号
    manifest_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")  # 清单 SHA-256
    frozen_at: str | None = None  # 冻结时间
    frozen_by: str | None = None  # 冻结操作人
    frozen_by_display_name: str | None = None  # 冻结操作人展示名
    source_workspace_revision: int | None = Field(default=None, ge=1)  # 来源工作区版本号
    source_document_count: int = Field(default=0, ge=0, le=100_000)  # 来源文档数
    current_corpus_draft_id: str | None = Field(default=None, pattern=r"^ccd_[a-f0-9]{32}$")  # 关联现行语料草稿 ID
    current_corpus_snapshot_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")  # 关联目录快照 SHA-256


class EvidenceSnapshotContextListResponse(BaseModel):
    """冻结语料版本的受限文档目录。"""

    dataset_id: str
    version: str
    source_document_count: int = Field(ge=0, le=100_000)
    contexts: list[EvidenceContextSummary] = Field(default_factory=list, max_length=100_000)


class EvidenceFrozenVersionListResponse(BaseModel):
    """按时间倒序的冻结版本摘要。"""

    versions: list[EvidenceFrozenVersion] = Field(default_factory=list)  # 版本列表


class EvidenceBulkActionRequest(EvidenceRevisionRequest):
    """按显式案例列表或受限服务端过滤条件选中案例。"""

    selection_mode: Literal["selected", "filtered"]  # 选择模式：显式列表或过滤条件
    case_ids: list[ResourceId] = Field(default_factory=list, max_length=10_000)  # 显式选中的案例 ID 列表
    category: str | None = Field(default=None, max_length=128)  # 过滤：分类
    review_status: Literal["draft", "pending_review", "approved"] | None = None  # 过滤：审核状态
    query: str | None = Field(default=None, max_length=200)  # 过滤：检索词

    @model_validator(mode="after")
    def validate_selection(self):
        """拒绝歧义的选择条件，而不是把空列表当作全选。"""
        # ① 案例列表不允许重复
        if len(self.case_ids) != len(set(self.case_ids)):
            raise ValueError("case_ids must not contain duplicates")
        # ② selected 模式必须至少提供一个 case_id
        if self.selection_mode == "selected" and not self.case_ids:
            raise ValueError("selected mode requires at least one case_id")
        # ③ filtered 模式不允许携带 case_ids
        if self.selection_mode == "filtered" and self.case_ids:
            raise ValueError("filtered mode must not include case_ids")
        return self


class EvidenceSubmitRequest(EvidenceRevisionRequest):
    """将草稿提交给显式指定的审核人。"""
    reviewer_id: str = Field(min_length=1, max_length=128)  # 指定审核人用户 ID


class EvidenceCurrentCorpusBulkSubmitRequest(EvidenceRevisionRequest):
    """把全部符合条件的现行语料草稿分派给一名同级公司管理员。"""
    reviewer_id: str = Field(min_length=1, max_length=128)  # 指定审核人用户 ID


class EvidenceCurrentCorpusBulkApproveRequest(EvidenceRevisionRequest):
    """仅审核分派给调用者本人的现行语料审核队列。"""


class EvidenceReviewerItem(BaseModel):
    """可安全用于选择列表的合格审核人身份。"""
    user_id: str  # 用户 ID
    username: str  # 用户名
    display_name: str  # 展示名


class EvidenceReviewRequest(EvidenceRevisionRequest):
    """Maker-Checker 审核决定；操作者身份一律取自认证上下文。"""
    decision: str = Field(pattern="^(approve|reject)$")  # 审核决定
    reason: str = Field(default="", max_length=1000)  # 审核说明


class EvidenceFixtureItem(BaseModel):
    """只读的历史 Fixture 档案元数据；绝非当前校验记录。"""
    legacy_manual_verified: bool = False  # 历史上是否人工验证过
    has_legacy_configuration: bool = False  # 是否存在历史配置


class EvidenceFixtureProfileResponse(BaseModel):
    """只读的历史 Fixture 档案元数据。"""
    dataset_id: ResourceId  # 数据集 ID
    revision: int = Field(ge=1)  # 工作区版本号
    status: str = Field(pattern="^(historical_only|pending_operator_configuration|ready)$")  # 档案状态
    fixtures: dict[str, EvidenceFixtureItem] = Field(default_factory=dict)  # Fixture 名称 → 档案元数据


class EvidenceFixtureValidationStartRequest(StrictRequestModel):
    """针对精确冻结版本来源修订启动校验。"""
    expected_revision: int = Field(ge=1)  # 期望的工作区版本号


class EvidenceFixtureValidationResult(BaseModel):
    """单个 Fixture 的校验结果。"""
    fixture: str = Field(min_length=1, max_length=96)  # Fixture 名称
    status: Literal["passed", "failed"]  # 校验状态
    reason_code: str = Field(min_length=1, max_length=96)  # 结果原因码
    started_at: str  # 开始时间
    finished_at: str  # 结束时间
    summary: str = Field(max_length=128)  # 结果摘要


class EvidenceFixtureValidationRun(BaseModel):
    """一次 Fixture 校验运行的记录。"""
    run_id: str  # 运行 ID
    dataset_id: ResourceId  # 数据集 ID
    version: str  # 冻结版本号
    manifest_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")  # 清单 SHA-256
    source_workspace_revision: int = Field(ge=1)  # 来源工作区版本号
    status: Literal["queued", "running", "succeeded", "failed"]  # 运行状态
    reason_code: str | None = Field(default=None, max_length=96)  # 终态原因码
    fixture_results: list[EvidenceFixtureValidationResult] = Field(default_factory=list, max_length=7)  # 各 Fixture 结果列表
    created_at: str  # 创建时间
    started_at: str | None = None  # 开始时间
    finished_at: str | None = None  # 结束时间


class EvidenceFixtureValidationRunListResponse(BaseModel):
    """Fixture 校验运行列表。"""
    runs: list[EvidenceFixtureValidationRun] = Field(default_factory=list, max_length=50)  # 运行列表


class EvidenceMutationResponse(BaseModel):
    """更新后的案例与权威工作区版本号。"""
    revision: int  # 工作区版本号
    case: EvidenceCaseItem  # 更新后的案例


class EvidenceDeleteResponse(BaseModel):
    """被删案例标识与权威工作区版本号。"""
    revision: int  # 工作区版本号
    deleted_case_id: ResourceId  # 被删除的案例 ID


class EvidenceRepresentativeTemplateSkip(BaseModel):
    """单个代表性模板导入被跳过的受限结果。"""

    sample: str = Field(min_length=1, max_length=128)  # 被跳过的样本标识
    reason_code: str = Field(min_length=1, max_length=96)  # 跳过原因码


class EvidenceRepresentativeTemplateImportResponse(BaseModel):
    """现行语料模板导入结果，不含旧文档元数据。"""

    dataset_id: ResourceId  # 数据集 ID
    revision: int = Field(ge=1)  # 工作区版本号
    imported_case_ids: list[ResourceId] = Field(default_factory=list, max_length=12)  # 导入成功的案例 ID 列表
    skipped: list[EvidenceRepresentativeTemplateSkip] = Field(default_factory=list, max_length=12)  # 跳过项列表


class EvidenceBulkSkippedItem(BaseModel):
    """单个未变更案例及其受限的批处理跳过原因。"""

    case_id: ResourceId  # 案例 ID
    code: str = Field(min_length=1, max_length=96)  # 跳过原因码


class EvidenceReferenceAnswerCandidateResponse(BaseModel):
    """从已校验冻结文档生成候选答案的受限结果。"""

    dataset_id: ResourceId
    revision: int = Field(ge=1)
    populated_case_ids: list[ResourceId] = Field(default_factory=list, max_length=100)
    skipped: list[EvidenceBulkSkippedItem] = Field(default_factory=list, max_length=100)


class EvidenceBulkActionResponse(BaseModel):
    """一次批量治理操作的权威版本号与受限结果。"""

    revision: int  # 工作区版本号
    matched_count: int = Field(ge=0, le=10_000)  # 匹配到的案例数
    processed_count: int = Field(ge=0, le=10_000)  # 实际处理的案例数
    skipped_count: int = Field(ge=0, le=10_000)  # 跳过的案例数
    processed_case_ids: list[ResourceId] = Field(default_factory=list, max_length=10_000)  # 已处理案例 ID 列表
    skipped_items: list[EvidenceBulkSkippedItem] = Field(default_factory=list, max_length=10_000)  # 跳过项列表


class EvidenceFreezeResponse(BaseModel):
    """新冻结产物的公开标识，不含文件系统路径。"""
    status: str  # 冻结状态
    version: str  # 冻结版本号
    manifest_sha256: str  # 清单 SHA-256
    revision: int  # 工作区版本号

class EvidenceReleaseWorkflowStartRequest(StrictRequestModel):
    """从路径指定的冻结 current_corpus 版本启动发布工作流。"""
    evaluation_dataset_id: ResourceId  # 正式评测数据集 ID
    evaluation_version: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")  # 正式评测冻结版本号


class EvidenceReleaseWorkflowActionRequest(StrictRequestModel):
    """仅含乐观锁版本号的工作流操作；操作者身份一律来自认证。"""
    expected_revision: int = Field(ge=1)  # 期望的工作流版本号


class EvidenceReleaseWorkflowReviewRequest(EvidenceReleaseWorkflowActionRequest):
    """审批或驳回发布评测时的受控输入。"""
    reason: str = Field(min_length=1, max_length=1000)  # 审批说明，不允许为空
    target_mode: str | None = Field(default=None, pattern=r"^(shadow|enforce)$")


class EvidenceGateRollbackRequest(StrictRequestModel):
    """受控回滚公司 Gate 的乐观锁请求。"""
    expected_revision: int = Field(ge=1)
    reason: str = Field(min_length=1, max_length=1000)


class EvidenceGateConfigurationResponse(BaseModel):
    """不含评测原文的公司 Gate 当前配置。"""
    mode: str = Field(pattern=r"^(off|shadow|enforce)$")
    revision: int = Field(ge=0)
    calibration_version: str | None = None
    source: str
    updated_at: str | None = None
    updated_by: str | None = None


class EvidenceReleaseWorkflowResponse(BaseModel):
    """发布工作流的当前状态。"""
    workflow_id: str  # 工作流 ID
    dataset_id: str  # 数据集 ID
    version: str  # 冻结版本号
    manifest_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")  # 清单 SHA-256
    current_corpus_draft_id: str | None = Field(default=None, pattern=r"^ccd_[a-f0-9]{32}$")  # 关联现行语料草稿 ID
    current_corpus_snapshot_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")  # 关联目录快照 SHA-256
    fixture_validation_run_id: str | None = Field(default=None, pattern=r"^fvr_[a-f0-9]{32}$")  # 关联 Fixture 校验运行 ID
    evaluation_dataset_id: str | None = None  # 正式评测数据集 ID
    evaluation_version: str | None = None  # 正式评测冻结版本号
    evaluation_manifest_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")  # 正式评测清单 SHA-256
    evaluation_case_count: int | None = Field(default=None, ge=1, le=10_000)  # 正式评测案例数
    status: str  # 工作流状态
    stage: str  # 当前阶段
    revision: int  # 工作流版本号
    initiated_by: str  # 发起人用户 ID
    target_mode: str | None = None  # 审批通过的目标 Gate 模式
    initiated_by_display_name: str = Field(min_length=1, max_length=64)  # 发起人展示名
    calibration_version: str | None = None  # 校准版本
    reason_code: str | None = None  # 终态原因码
    created_at: str | None = None  # 创建时间
    updated_at: str | None = None  # 更新时间


class EvidenceReleaseWorkflowListResponse(BaseModel):
    """公司范围内有限的工作流历史。"""

    workflows: list[EvidenceReleaseWorkflowResponse] = Field(default_factory=list, max_length=100)  # 工作流列表


class EvidenceDiagnosticRootCauseResponse(BaseModel):
    """Bounded evidence identity for one diagnostic attribution."""

    decision: str = Field(max_length=64)
    reason_code: str = Field(max_length=96)
    paired_count: int | None = Field(default=None, ge=0, le=10_000)
    effect_direction: str | None = Field(default=None, max_length=32)
    risk: str | None = Field(default=None, max_length=96)
    recommend_embedding_change: bool = False
    compared_variants: list[str] = Field(default_factory=list, max_length=10)
    supporting_hashes: list[str] = Field(default_factory=list, max_length=10)


class EvidenceDiagnosticSummaryResponse(BaseModel):
    """Additive, allowlisted diagnostic summary safe for an admin API."""

    availability: Literal["available", "not_available", "unsupported_schema"]
    schema_version: str | None = Field(default=None, max_length=64)
    policy_identities: dict[str, str] = Field(default_factory=dict, max_length=8)
    variant_identities: dict[str, str] = Field(default_factory=dict, max_length=12)
    categories: dict[str, dict[str, int | float | str | None]] = Field(default_factory=dict, max_length=11)
    stages: dict[str, dict[str, int | float | str | None]] = Field(default_factory=dict, max_length=10)
    metric_coverage: dict[str, dict[str, int | float | str | None]] = Field(default_factory=dict, max_length=32)
    route_transitions: dict[str, int] = Field(default_factory=dict, max_length=32)
    hard_gates: list[str] = Field(default_factory=list, max_length=20)
    root_cause: EvidenceDiagnosticRootCauseResponse | None = None
    case_ids: list[str] = Field(default_factory=list, max_length=20)


class EvidenceReleaseAttemptResponse(BaseModel):
    """单次发布尝试的记录。"""
    attempt_id: str  # 尝试 ID
    stage: str  # 尝试所处阶段
    attempt_number: int  # 尝试序号
    status: str  # 尝试状态
    reason_code: str | None = None  # 终态原因码
    metrics: dict[str, Any] = Field(default_factory=dict)  # 尝试指标
    diagnostic_summary: EvidenceDiagnosticSummaryResponse
    started_at: str | None = None  # 开始时间
    finished_at: str | None = None  # 结束时间


class EvidenceReleaseAttemptListResponse(BaseModel):
    """发布工作流的尝试记录列表。"""
    attempts: list[EvidenceReleaseAttemptResponse] = Field(default_factory=list)  # 尝试列表
