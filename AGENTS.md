<!-- CODEGRAPH_START -->
## CodeGraph

In repositories indexed by CodeGraph (a `.codegraph/` directory exists at the repo root), reach for it BEFORE grep/find or reading files when you need to understand or locate code:

- **MCP tool** (when available): `codegraph_explore` answers most code questions in one call — the relevant symbols' verbatim source plus the call paths between them, including dynamic-dispatch hops grep can't follow. Name a file or symbol in the query to read its current line-numbered source. If it's listed but deferred, load it by name via tool search.
- **Shell** (always works): `codegraph explore "<symbol names or question>"` prints the same output.

If there is no `.codegraph/` directory, skip CodeGraph entirely — indexing is the user's decision.
<!-- CODEGRAPH_END -->

--- project-doc ---

# KnowledgeForge 协作指南

本文件适用于仓库根目录及所有子目录；开始任何改动前先阅读本索引，再按改动范围阅读对应规则。

## 项目简介

KnowledgeForge 是企业级多 Agent 知识管理系统。后端以 FastAPI 提供 API，由文档解析、知识抽取、问答和知识更新 Agent 及工作流协同处理知识；前端是 React/Vite 管理后台。

## 文档分层

- `rules/`：开发、测试、安全、目录、Git、兼容清理等强约束，默认必须遵守。
- `docs/knowledge.md`：只记录无法从源码或 CodeGraph 稳定推导的业务、产品和领域背景。
- `reference/`：从旧 `docs/` 迁移来的历史参考资料。使用前必须结合 CodeGraph、当前源码、测试和 OpenSpec 判断是否仍然准确。

代码结构、文件位置、符号定义、调用链和影响范围以 CodeGraph 或当前源码为准，不在 AGENTS.md、`rules/` 或 `docs/` 维护第二套代码地图。

## 工具使用原则

- 查现有代码结构、符号、调用链和影响范围：优先使用 CodeGraph（`codegraph_explore` 或 `codegraph explore`）。
- 做整体理解、知识图谱、onboarding、架构可视化：可使用 Understand Anything。
- 做会改变用户可见行为、API、Agent 行为、配置、部署或运维结果的变更：优先走 OpenSpec，明确目标、范围、验收场景和任务。
- OpenSpec 完成并通过验证后：先让用户确认；确认后自动归档 change，并把本阶段相关变更创建为本地 commit；默认不 push、不建 PR、不合并分支。
- 查业务语境：先看 `docs/knowledge.md`；历史资料只从 `reference/` 辅助判断。

## 规则索引

- [开发流程规则](rules/development.md)：OpenSpec、ATDD、实施顺序和收尾要求。
- [后端规则](rules/backend.md)：FastAPI、Agent、RAG、工作流、模型调用和安全边界。
- [前端规则](rules/frontend.md)：React、Vite、Zustand、API client 和前端安全边界。
- [目录与放置规则](rules/directory.md)：新代码、测试和文档应该放哪里；现有结构查询走 CodeGraph。
- [测试规则](rules/testing.md)：pytest、前端检查、评测测试和外部服务测试边界。
- [接口与兼容代码清理规则](rules/compatibility.md)：替换入口后的旧入口审计、清理和验证要求。
- [Git 规范](rules/git.md)：忽略文件、安全提交、暂存范围、分支和历史操作边界。

## Codex / sandbox 运行提示

- 本项目后端用 `uv` 进行管理。如果 `uv` 因缓存目录权限、用户级缓存不可写或 sandbox 写入限制失败，优先把缓存放到项目内：`UV_CACHE_DIR=.uv-cache uv run ...` 或 `UV_CACHE_DIR=.uv-cache uv sync`。`.uv-cache/` 是本地缓存目录，只用于加速依赖下载或构建，可删除且不得提交。
- 运行环境是本机 Docker Compose（compose 项目名 `knowledgeforge`）。构建镜像、创建、重建、启动或重启容器按仓库根目录 `docker-compose.yml` 与 `config/.env` 执行；不要假设服务运行在远程服务器，也不要连接任何 SSH 主机做部署或排障。禁止在宿主机上为项目创建 Python 虚拟环境，包括执行 `uv venv`、`python -m venv`、`virtualenv` 或 `conda create`；Dockerfile 在镜像构建阶段按项目依赖定义创建镜像内部环境不属于宿主机虚拟环境。

## 安全与禁止事项

- 只使用 `config/.env.example` 作为配置模板；不得提交真实 API Key、JWT 密钥、数据库密码、令牌或用户数据。
- 不提交本地环境、缓存、覆盖率、运行日志或上传文件等生成物；`backend/pyproject.toml`、`backend/uv.lock` 与前端锁文件是可追踪的依赖定义，不得忽略。
- 不要绕过认证、RBAC、租户隔离、输入校验、内容安全、限流或审计/日志要求。
- 不要在无明确需求时修改依赖、部署配置、数据存储结构或无关文件。
- 外部 LLM、Neo4j、ChromaDB、Redis、Kafka 等服务在测试中应隔离或 mock，不要把真实凭据或生产端点写入测试。
- 不执行破坏性 Git、数据库、消息队列或基础设施操作，除非任务明确要求且影响范围已确认。

## 真实性原则

- 不从其他项目复制目录、命令、架构层或工具约定。
- 只声明仓库中存在、CodeGraph/源码可确认或任务明确要求的能力。
- 配置、服务端口和依赖命令以仓库文件为准，新增约定须明确记录其目的与适用范围。
- 后端依赖的唯一入口是 `backend/pyproject.toml` 与 `backend/uv.lock`；前端依赖以 `frontend/package.json` 和对应锁文件为准。
