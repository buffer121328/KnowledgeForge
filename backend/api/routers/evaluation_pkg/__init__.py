"""评测治理路由包：按资源域拆分的子模块在 import 时向共享 router 注册端点。

历史上本包是单个 2400+ 行的 ``api/routers/evaluation.py``；现按
results（只读看板）/ release_workflows（发布流）/ current_corpus_drafts（现行语料
草稿）/ datasets（数据集与版本）/ cases（案例 CRUD 与批量流转）/ fixtures（Fixture
校验与冻结）拆分。共享的 ``evaluation_router``、路径常量、单例、错误映射与
审计/变更管道定义在 ``_common``，各子模块 import 即完成端点注册。
"""

from __future__ import annotations

from api.routers.evaluation_pkg import (  # noqa: F401
    cases,
    current_corpus_drafts,
    datasets,
    fixtures,
    release_workflows,
    results,
)
from api.routers.evaluation_pkg._common import evaluation_router  # noqa: F401

__all__ = ["evaluation_router"]
