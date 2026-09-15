"""API knowledge_graph 域共享的请求与响应模型（自 api/schemas.py 拆分）。"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from api.contracts import StrictRequestModel


class GraphNodeItem(BaseModel):
    """组织、文档、断言或实体类图谱节点。"""

    id: str  # 节点 ID
    label: str  # 节点标签
    type: str = "Concept"  # 节点类型，默认概念
    description: str = ""  # 节点描述
    company_id: str = ""  # 公司 ID
    department_id: str = ""  # 部门 ID
    document_count: int = 0  # 关联文档数


class GraphEdgeItem(BaseModel):
    """图谱边或跨部门聚合边。"""

    source: str  # 起点节点 ID
    target: str  # 终点节点 ID
    label: str  # 边标签
    claim_id: str = ""  # 关联断言 ID
    path_count: int = 1  # 聚合路径数
    claim_ids: list[str] = Field(default_factory=list)  # 聚合的断言 ID 列表
    relation_types: list[str] = Field(default_factory=list)  # 关系类型列表
    relation_details: list[dict[str, Any]] = Field(default_factory=list)  # 关系明细列表


class GraphEntityUpdateRequest(StrictRequestModel):
    """用户对抽取实体的修正请求。"""

    node_id: str = Field(min_length=1, max_length=255)  # 待修正的节点 ID
    label: str | None = Field(default=None, max_length=255)  # 修正后的标签；为空表示不修改
    type: str | None = Field(default=None, max_length=64)  # 修正后的类型
    description: str | None = Field(default=None, max_length=1000)  # 修正后的描述


class GraphEntityUpdateResponse(BaseModel):
    """返回修正后的实体投影。"""

    node: GraphNodeItem  # 修正后的节点
    status: str = "ok"  # 操作状态


class GraphRelationUpdateRequest(StrictRequestModel):
    """用户对抽取关系断言的修正请求。"""

    claim_id: str = Field(min_length=1, max_length=128)  # 待修正的断言 ID
    relation_type: str = Field(min_length=1, max_length=64)  # 修正后的关系类型


class GraphRelationUpdateResponse(BaseModel):
    """返回修正后的关系断言摘要。"""

    claim_id: str  # 断言 ID
    relation_type: str  # 关系类型
    status: str = "ok"  # 操作状态


class GraphDepartmentItem(BaseModel):
    """公司图谱概览中的部门摘要。"""

    department_id: str  # 部门 ID
    name: str  # 部门名称
    document_count: int = 0  # 部门文档数


class GraphSubgraphResponse(BaseModel):
    """限定范围的图谱子图响应。"""

    nodes: list[GraphNodeItem]  # 节点列表
    edges: list[GraphEdgeItem]  # 边列表
    status: str = "ok"  # 操作状态
    scope: str = "entity"  # 查询范围
    department_id: str = ""  # 限定部门 ID


class CompanyGraphOverviewResponse(BaseModel):
    """属于当前租户的公司与部门图谱概览。"""

    company_id: str = ""  # 公司 ID
    company_name: str = ""  # 公司名称
    departments: list[GraphDepartmentItem] = Field(default_factory=list)  # 部门摘要列表
    nodes: list[GraphNodeItem] = Field(default_factory=list)  # 节点列表
    edges: list[GraphEdgeItem] = Field(default_factory=list)  # 边列表
    status: str = "ok"  # 操作状态


class GraphPathResponse(BaseModel):
    """两个所选节点之间有限的具体路径。"""

    paths: list[dict[str, Any]] = Field(default_factory=list)  # 路径列表
    path_count: int = 0  # 路径数
    status: str = "ok"  # 操作状态


class GraphEvidenceResponse(BaseModel):
    """单条语义断言的可追溯证据记录。"""

    claim_id: str  # 断言 ID
    evidence: list[dict[str, Any]] = Field(default_factory=list)  # 证据记录列表
    status: str = "ok"  # 操作状态

