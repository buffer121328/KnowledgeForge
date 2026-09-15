"""API documents 域共享的请求与响应模型（自 api/schemas.py 拆分）。"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from api.contracts import FilePath, StrictRequestModel


# 文档入库状态 → (展示阶段名, 阶段序号, 中文标签) 的归一映射；未知状态回退 staging
_STAGE_FIELDS = {
    "staging": ("staging", 1, "上传校验与落盘"),
    "accepted": ("accepted", 2, "文件已接收"),
    "processing": ("processing", 3, "解析与知识抽取"),
    "ingested": ("ingested", 4, "入库完成"),
    "success": ("ingested", 4, "入库完成"),
    "failed": ("failed", 4, "处理失败"),
    "legacy": ("ingested", 4, "入库完成"),
}


def _stage_fields(status: str) -> tuple[str, int, str]:
    """将入库状态归一化为 (展示阶段名, 阶段序号, 中文标签)，未知状态按 staging 处理。

    Args:
        status: 文档入库状态字符串。
    """
    return _STAGE_FIELDS.get(status, _STAGE_FIELDS["staging"])


# 处理子步骤 → (步骤名, 步骤序号, 中文标签) 的映射；总步数固定为 6
_PROCESSING_STAGE_FIELDS = {
    "parse": ("parse", 1, "解析文件"),
    "extract": ("extract", 2, "知识抽取"),
    "store_vectors": ("store_vectors", 3, "写向量库"),
    "store_graph": ("store_graph", 4, "写知识图谱"),
    "store_sparse": ("store_sparse", 5, "写稀疏索引"),
    "advance_revision": ("advance_revision", 6, "收尾提交"),
    "finalize": ("finalize", 6, "收尾提交"),
}


def _processing_stage_fields(value: str | None) -> tuple[str, int, int, str]:
    """将处理子步骤归一化为 (步骤名, 步骤序号, 总步数, 中文标签)，无效值回退 parse。

    Args:
        value: 原始处理子步骤标识，可为空。
    """
    normalized = str(value or "").strip()
    # 未识别的步骤名一律回退到第一步 parse
    stage = normalized if normalized in _PROCESSING_STAGE_FIELDS else "parse"
    key, index, label = _PROCESSING_STAGE_FIELDS[stage]
    return key, index, 6, label


class IngestResponse(BaseModel):
    """单份文档入库结果。"""
    file_name: str  # 文件名
    chunks_count: int  # 生成的分块数
    entities_count: int  # 抽取的实体数
    relations_count: int  # 抽取的关系数
    status: str  # 入库状态
    task_id: str = ""  # 关联后台任务 ID
    doc_id: str = ""  # 文档 ID
    client_file_id: str = ""  # 客户端文件标识
    relative_path: str = ""  # 文件夹内相对路径
    department_id: str = ""  # 归属部门 ID
    display_name: str = ""  # 展示名
    provenance_source_filename: str = ""  # 来源文件名
    version: int = 1  # 文档版本号
    error_code: str = ""  # 错误码
    message: str = ""  # 提示消息
    ingest_stage: str = "staging"  # 入库展示阶段名
    ingest_stage_index: int = 1  # 入库展示阶段序号
    ingest_stage_total: int = 4  # 入库展示阶段总数
    ingest_stage_label: str = "上传校验与落盘"  # 入库展示阶段中文标签
    processing_step: str = "parse"  # 处理子步骤名
    processing_step_index: int = 1  # 处理子步骤序号
    processing_step_total: int = 6  # 处理子步骤总数
    processing_step_label: str = "解析文件"  # 处理子步骤中文标签
    processing_step: str = "parse"  # 处理子步骤名（重复定义）
    processing_step_index: int = 1  # 处理子步骤序号（重复定义）
    processing_step_total: int = 6  # 处理子步骤总数（重复定义）
    processing_step_label: str = "解析文件"  # 处理子步骤中文标签（重复定义）

    @model_validator(mode="after")
    def normalize_stage_fields(self):
        """按 status 归一化入库展示阶段字段，并把处理子步骤重置为第一步。"""
        stage, index, label = _stage_fields(self.status)  # ① 按入库状态取阶段三元组
        self.ingest_stage = stage
        self.ingest_stage_index = index
        self.ingest_stage_total = 4
        self.ingest_stage_label = label
        step, step_index, step_total, step_label = _processing_stage_fields(None)  # ② 子步骤重置为 parse
        self.processing_step = step
        self.processing_step_index = step_index
        self.processing_step_total = step_total
        self.processing_step_label = step_label
        return self


class IngestProgressItem(BaseModel):
    """上传批次中属于当前租户的单个成员进度。"""

    doc_id: str = ""  # 文档 ID
    client_file_id: str = ""  # 客户端文件标识
    file_name: str  # 文件名
    status: str  # 成员状态
    ingest_stage: str  # 入库展示阶段名
    ingest_stage_index: int  # 入库展示阶段序号
    ingest_stage_total: int = 4  # 入库展示阶段总数
    ingest_stage_label: str  # 入库展示阶段中文标签
    processing_step: str = "parse"  # 处理子步骤名
    processing_step_index: int = 1  # 处理子步骤序号
    processing_step_total: int = 6  # 处理子步骤总数
    processing_step_label: str = "解析文件"  # 处理子步骤中文标签
    error_code: str = ""  # 错误码
    message: str = ""  # 提示消息


class IngestProgressResponse(BaseModel):
    """一个上传批次的整体与逐文件进度。"""

    upload_id: str  # 上传批次 ID
    total_count: int  # 批次文件总数
    completed_count: int  # 已完成数
    failed_count: int  # 失败数
    stage_index: int  # 整体入库阶段序号
    stage_total: int = 4  # 整体入库阶段总数
    stage_label: str  # 整体入库阶段中文标签
    processing_step: str = "parse"  # 当前处理子步骤名
    processing_step_index: int = 1  # 当前处理子步骤序号
    processing_step_total: int = 6  # 处理子步骤总数
    processing_step_label: str = "解析文件"  # 处理子步骤中文标签
    terminal: bool  # 批次是否已进入终态
    status: Literal["processing", "success", "failed"]  # 批次整体状态
    items: list[IngestProgressItem]  # 逐文件进度列表


class FolderManifestFileRequest(StrictRequestModel):
    """文件夹上传清单中的单个逻辑文件。"""

    client_file_id: str = Field(min_length=1, max_length=128)  # 客户端文件标识
    relative_path: str = Field(min_length=1, max_length=1024)  # 文件在文件夹内的相对路径
    department_id: str = Field(min_length=1, max_length=128)  # 归属部门 ID
    external_source_id: str = Field(default="", max_length=128)  # 外部来源 ID；为空则由服务端分配
    display_name: str = Field(default="", max_length=255)  # 展示名
    provenance_source_filename: str = Field(default="", max_length=255)  # 来源文件名
    metadata: dict[str, Any] = Field(default_factory=dict)  # 附加元数据


class FolderUploadManifestRequest(StrictRequestModel):
    """随 multipart 文件夹上传一并提交的 JSON 清单。"""

    root_folder_name: str = Field(min_length=1, max_length=255)  # 根目录名称
    company_id: str = Field(min_length=1, max_length=128)  # 归属公司 ID
    department_mappings: dict[str, str] = Field(max_length=100)  # 顶层目录名 → 部门 ID 映射
    files: list[FolderManifestFileRequest] = Field(max_length=50)  # 清单文件条目列表
    upload_id: str = Field(default="", max_length=128)  # 上传批次 ID


class DepartmentItem(BaseModel):
    """可用于文件夹映射与图谱范围的部门。"""

    department_id: str  # 部门 ID
    company_id: str  # 归属公司 ID
    name: str  # 部门名称
    normalized_key: str  # 归一化目录键
    status: str = "active"  # 部门状态


class DocListItem(BaseModel):
    """基于目录记录的文档列表项。"""
    doc_id: str  # 文档 ID
    file_name: str  # 展示文件名
    source: str = ""  # 来源标识
    file_reference: str = ""  # 已授权的文件引用
    doc_type: str  # 文档类型
    chunks_count: int  # 分块数
    tenant_id: str = ""  # 租户 ID
    company_id: str = ""  # 公司 ID
    department_id: str = ""  # 部门 ID
    folder_path: str = ""  # 文件夹路径
    relative_path: str = ""  # 相对路径
    uploaded_filename: str = ""  # 上传时的文件名
    display_name: str = ""  # 展示名
    provenance_source_filename: str = ""  # 来源文件名
    content_sha256: str = ""  # 内容 SHA-256 摘要
    version: int = 1  # 版本号
    ingest_status: str = ""  # 入库状态
    authority: str = ""  # 权威等级
    review_status: str = ""  # 审核状态
    sensitivity: str = ""  # 敏感度
    entities_count: int = 0  # 实体数
    relations_count: int = 0  # 关系数
    created_at: str = ""  # 创建时间
    updated_at: str = ""  # 更新时间
    error_code: str = ""  # 错误码
    retryable: bool = False  # 失败后是否可重试
    legacy: bool = False  # 是否旧数据
    legacy_name_unresolved: bool = False  # 旧数据文件名是否无法解析
    ingest_stage: str = "staging"  # 入库展示阶段名
    ingest_stage_index: int = 1  # 入库展示阶段序号
    ingest_stage_total: int = 4  # 入库展示阶段总数
    ingest_stage_label: str = "上传校验与落盘"  # 入库展示阶段中文标签

    @model_validator(mode="after")
    def normalize_stage_fields(self):
        """旧记录缺省阶段信息时按 ingest_status 回填入库展示阶段。"""
        # 仅当阶段字段仍是默认初值且状态已前进时才回填，避免覆盖显式提供的阶段
        if self.ingest_stage == "staging" and self.ingest_stage_index == 1 and self.ingest_status != "staging":
            stage, index, label = _stage_fields(self.ingest_status)
            self.ingest_stage = stage
            self.ingest_stage_index = index
            self.ingest_stage_total = 4
            self.ingest_stage_label = label
        return self


class ChunkItem(BaseModel):
    """文档分块内容项。"""
    chunk_id: str  # 分块 ID
    content: str  # 分块文本
    chunk_index: int  # 分块序号
    doc_id: str  # 所属文档 ID
    doc_type: str = ""  # 文档类型
    source: str = ""  # 来源标识
    tenant_id: str = ""  # 租户 ID


class DocDeleteResponse(BaseModel):
    """文档删除响应。"""
    doc_id: str  # 被删除的文档 ID
    vectors_deleted: int  # 删除的向量数
    entities_deleted: int  # 删除的实体数
    file_deleted: bool = False  # 是否删除了源文件
    status: str = "success"  # 删除状态


class StatsResponse(BaseModel):
    """系统统计响应。"""
    vector_store: dict[str, Any]  # 向量库统计
    knowledge_graph: dict[str, Any]  # 知识图谱统计
    request_trend: list[dict[str, Any]] = []  # 请求趋势点列表
    request_detail: dict[str, Any] = Field(default_factory=dict)  # 请求明细


class RequestTrendSummary(BaseModel):
    """单个请求趋势窗口的汇总值。"""

    requests: int  # 请求总数
    qa: int  # 问答请求数
    errors: int  # 错误数
    average_latency_ms: float  # 平均延迟（毫秒）
    details: dict[str, Any]  # 分类明细


class RequestTrendResponse(BaseModel):
    """轻量实时请求趋势响应。"""

    window: Literal["60m", "24h"]  # 统计窗口
    timezone: Literal["UTC"]  # 时区，固定 UTC
    generated_at: str  # 生成时间
    points: list[dict[str, Any]]  # 趋势点列表
    summary: RequestTrendSummary  # 窗口汇总



class UpdateRequest(StrictRequestModel):
    """知识更新请求。"""
    file_path: FilePath  # 变更文件路径
    change_type: Literal["created", "modified", "deleted"] = "modified"  # 变更类型


class UpdateResponse(BaseModel):
    """知识更新响应。"""
    file_path: str  # 变更文件路径
    vectors_added: int  # 新增向量数
    vectors_deleted: int  # 删除向量数
    entities_added: int  # 新增实体数
    relations_added: int  # 新增关系数
    success: bool  # 是否成功
    processing_time_ms: float  # 处理耗时（毫秒）
