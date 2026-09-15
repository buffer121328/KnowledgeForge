"""多模态文档解析 Agent（门面模块）。

实现拆分为：
- ``document_format_parsers.FormatParsingMixin``：构造、分类与 PDF/Word/图片
  OCR/表格/纯文本解析、``parse``/``parse_batch`` 公共入口；
- ``document_chunking.DocumentChunkingMixin``：递归与语义分块（双策略）。
公开类 ``DocParserAgent``、``DocumentChunk`` 与 ``DocumentWorkLimitError`` 保持不变。
"""

from __future__ import annotations

from domain.documents import DocumentChunk  # noqa: F401  (canonical contract re-export)

from agents.document_chunking import DocumentChunkingMixin
from agents.document_format_parsers import (  # noqa: F401
    _CACHE_UNSET,
    DocumentWorkLimitError,
    FormatParsingMixin,
)

__all__ = ["DocParserAgent", "DocumentChunk", "DocumentWorkLimitError"]


class DocParserAgent(FormatParsingMixin, DocumentChunkingMixin):
    """解析 7 类格式并按双策略生成分块。"""
