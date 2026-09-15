"""API 各功能路由共享的请求与响应模型，按域拆分为子模块后聚合导出。"""

from api.schemas.admin import *  # noqa: F401,F403
from api.schemas.documents import *  # noqa: F401,F403
from api.schemas.evaluation import *  # noqa: F401,F403
from api.schemas.knowledge_graph import *  # noqa: F401,F403
from api.schemas.qa import *  # noqa: F401,F403
