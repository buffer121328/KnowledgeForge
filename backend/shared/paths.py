"""Single source of truth for filesystem anchors inside the backend tree.

模块拆分或层级变化会改变 ``Path(__file__)`` 的深度，各处自算 parents[N] 极易漂移；
所有需要定位 backend 根或评测数据/运行时产物的代码一律引用本模块。
"""

from __future__ import annotations

from pathlib import Path

#: backend/ 源码根（容器内即 /app/backend，compose 卷挂载依赖该布局）。
BACKEND_ROOT = Path(__file__).resolve().parents[1]

#: 仓库根（仅本地开发脚本使用；生产容器内不要求存在）。
REPO_ROOT = BACKEND_ROOT.parent

#: 评测运行时数据：证据门 bundle（生产 API 与 worker 只读）。
EVALUATION_DATA_ROOT = BACKEND_ROOT / "evaluation" / "data"
EVIDENCE_GATES_DATA_ROOT = EVALUATION_DATA_ROOT / "evidence-gates"
COMPANY_DEMO_DATA_ROOT = EVALUATION_DATA_ROOT / "company-demo"
COMPANY_DEMO_CORPUS_MANIFEST = COMPANY_DEMO_DATA_ROOT / "corpus_manifest.json"

#: 评测运行产物（只写不提交）。
EVALUATION_RESULTS_ROOT = BACKEND_ROOT / "evaluation" / "results"

#: 证据评审工作区（maker-checker 双人复核状态）。
EVIDENCE_REVIEW_ROOT = BACKEND_ROOT / "data" / "evidence-review"
