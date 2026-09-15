
## 扩充 company-demo 行政部门并生成 100 条候选

行政部门源文件来自用户本地目录，使用代码检查并跳过含图片/绘图对象的 Word：

```bash
UV_CACHE_DIR=.uv-cache uv --directory backend run \
  python evaluation/scripts/prepare_company_demo_department.py \
  --source-root '/Users/cheng/Desktop/公司管理常用制度合集230份' \
  --corpus-root evaluation/data/company-demo \
  --selection evaluation/data/company-demo/administration-selection.json
```

准备报告写入 `administration_preparation_report.json`。`skipped_image_content` 文件不会复制、不会生成 normalized 文本，也不会使用 OCR 或人工摘录。随后确定性生成 80 条 company-demo benchmark（保留原 30 条，新增 50 条待复核候选）和 100 条 evidence-gate 候选：

```bash
UV_CACHE_DIR=.uv-cache uv --directory backend run \
  python evaluation/scripts/expand_company_demo_benchmark.py \
  --corpus-root evaluation/data/company-demo

UV_CACHE_DIR=.uv-cache uv --directory backend run \
  python evaluation/scripts/prepare_evidence_gate_candidates.py \
  --company-demo-root evaluation/data/company-demo \
  --output-root evaluation/data/evidence-gates
```

当前冻结语料为 41 份文档（人力资源 21、财务 5、采购/仓储 5、行政 10），evidence-gate 为 100 条。自动结构、哈希和 provenance 校验通过仍只表示 `pending_human_review`，不能替代逐题语义审核或直接晋升为发布基线。

# 本地 RAG 评测与演示资料

当前主方案是 **company-demo 中文企业知识库语料**（`data/company-demo/`）：从用户提供的公司制度资料中精选的四部门中文文档子集，建立可追溯的解析、vector/hybrid 检索、回答忠实度和拒答边界评测。`data/mudabench/`（2021 年 A 股年报 PDF 语料）已于 2026-08-01 整体移除，不再作为评测语料。

> 评测目录仍保存离线工具与不可变材料；公司管理员可在前端发起受控发布评测，由后台 Worker 调用同一 Python 评测 API 执行。它不是同步 HTTP benchmark，也不允许浏览器传入组织号、租户、部署凭据或模型密钥。原始 PDF、运行结果和任何真实 judge 凭据都不得提交；题目、人工答案、manifest 和 review 文件也绝不上传为知识库文档。

## 安装隔离的评测依赖

```bash
UV_CACHE_DIR=.uv-cache uv --directory backend sync --locked --group eval
```

`eval` 组固定 Ragas；其版本可在 `backend/uv.lock` 和每次运行的 `run_metadata.json` 中追溯。应用在线模型的 `OPENAI_*` 与评测 judge 的 `EVAL_OPENAI_*` 必须分开配置。

```bash
# 只用于离线 Ragas judge；请在本地 .env 或进程环境中配置真实值，绝不提交。
export EVAL_OPENAI_API_KEY='...'
export EVAL_OPENAI_BASE_URL='https://api.openai.com/v1'  # OpenAI 兼容端点可替换
export EVAL_OPENAI_MODEL='gpt-4o-mini'
```

## 评测数据与入库边界

| 文件 / 目录 | 用途 | 是否上传到 KnowledgeForge |
|---|---|---|
| `data/company-demo/normalized/` | 清洗后的中文企业文档评测语料 | **是**；仅按受控导入流程写入固定公司的评测语料范围，不提供隔离租户选择器 |
| `data/company-demo/corpus_manifest.json` | 文档清单、哈希、清洗状态与 provenance | 否 |
| `data/company-demo/benchmark.jsonl` | 80 条 RAG benchmark：原 30 条金标与新增 50 条待复核题，包含 reference、章节 evidence 和拒答预期 | 否 |
| `data/company-demo/review-record.json` | 金标审核记录与 2026-08-01 金标升级说明 | 否 |
| `data/agent-benchmark/agent_benchmark.jsonl` | Agent 目标、范围主题、拒答预期和真实工作流操作契约 | 否 |
| `results/<run_id>/` | 完整 contexts、响应、分数、trace 元数据 | 否，且 Git 忽略；目录 `0700`、文件 `0600` |

## company-demo：四部门中文企业知识库 Demo

为演示 KnowledgeForge 面向企业内部知识库的完整潜力，仓库新增了一个独立的 `data/company-demo/` profile。它不是外部公共数据集，而是从用户提供的公司制度资料中精选的中文四部门子集：

- 人力资源部：21 份，覆盖招聘、入职、培训、考勤、休假、薪酬、绩效、晋升和离职；
- 财务部：5 份，覆盖出差费用、现金、资金、资产盘点和财务岗位职责；
- 采购/仓储部：5 份，覆盖采购流程、采购合同、合同管理和仓库入库；
- 行政管理部：10 份，覆盖会议、档案、办公用品、车辆、接待、印章、环境卫生和安全管理。

当前 profile 包含 41 份源文档和 80 条中文 Demo benchmark（原 30 条已保留，新增 50 条仍待人工复核）。题目覆盖单文档事实、流程/职责、多文档关联、文档比较、证据不足和预期拒答。它适合展示：

- 多部门混合检索和来源引用；
- 招聘—入职—培训—绩效等人力资源链路；
- 采购—合同—验收/入库等跨部门链路；
- 模板/参考资料与“现行正式制度”之间的谨慎边界；
- 无法由制度文本证明真实审批、版本或业务结果时的拒答。

校验 profile：

```bash
UV_CACHE_DIR=.uv-cache uv --directory backend run \
  python evaluation/scripts/validate_company_demo.py
```

先生成清洗后的演示输入：

```bash
UV_CACHE_DIR=.uv-cache uv --directory backend run \
  python evaluation/scripts/clean_company_demo.py
```

清洗/准备链路将 41 份纳入语料的 DOCX 转换为 `data/company-demo/normalized/` 下的 UTF-8 TXT，保留标题、来源文档 ID、段落顺序和可提取表格行，并在 `cleaning_report.json` 中记录输入/输出哈希、字符数、段落数、表格行和占位符命中。它不调用 LLM，也不会覆盖原始 DOCX。

导入演示知识库时，优先只使用 `data/company-demo/normalized/`；不要同时上传 `documents/` 和 `normalized/`，否则会形成重复内容。`corpus_manifest.json`、`cleaning_report.json`、`benchmark.jsonl`、`review-record.json` 和 `selection_notes.md` 是 provenance、评测和治理资产，不能上传为运行时知识。所有文档目前标记为 `needs_review` / `unverified_reference`，因此该 profile 只用于演示和离线回归，不代表任何真实企业的正式制度。

company-demo 的 `benchmark.jsonl` 覆盖 80 条中文题，其中原 30 条为已升级金标，新增 50 条保持待人工复核。2026-08-01 已把全部 `reference` 从"回答指引"改写为**确定性金标答案**（可直接从文档核对的事实），`evidence` 按真实章节标题定位；由于 DOCX 无稳定页码模型，`evidence.page` 保持占位 `1`。审核状态如实记录在 [`data/company-demo/review-record.json`](data/company-demo/review-record.json)：基于本地 `normalized/` 文本复核，**待人工最终确认**章节定位、版本状态与正式制度后才可用于对外正式结论。语料来源与边界见 `data/company-demo/selection_notes.md`。

## 执行 vector / hybrid 检索快照

`BenchmarkRunner` 是 Python API：调用方必须先从固定公司的受控内部命名空间构造已初始化的 `QAAgent`（或兼容的异步执行器），然后直接传入 `QAAgent.answer`。它不调用 `/api/v1/qa/ask`，因此不会使用 HTTP 的 200 字符来源摘要，也不会暴露或要求用户填写租户标识。

```python
from pathlib import Path

from agents.qa_agent import QAAgent
from evaluation.benchmark import load_benchmark
from evaluation.runner import BenchmarkRunner

# vector_store、knowledge_graph 必须指向固定公司中只含 company-demo normalized 文档的受控评测范围。
# qa_cache=None 是额外保护；runner 也会传 user_id="" 禁用常规 QA 缓存。
agent = QAAgent(
    vector_store=vector_store,
    knowledge_graph=knowledge_graph,
    qa_cache=None,
)
benchmark = load_benchmark(Path("evaluation/data/company-demo/benchmark.jsonl"))
runner = BenchmarkRunner(
    qa_executor=agent.answer,
    results_root=Path("evaluation/results"),
)
run = await runner.run(
    benchmark,
    tenant_id="eval-company-demo-2026-08",
    retrieval_modes=("vector", "hybrid"),
    run_metadata={
        "answer_model": "<application model>",
        "evaluator_model": "<EVAL_OPENAI_MODEL>",
        "corpus_manifest": "evaluation/data/company-demo/corpus_manifest.json",
    },
)
print(run.responses_path)
```

每条结果都保存完整、有序的 `contexts[*].content`、`source_document_id`、分数、来源类型、模式、响应、耗时和异常类型。异常消息不写入产物；benchmark metadata 只保存源文件名和 SHA-256，不保存绝对路径。`vector` 模式不会调用图谱检索；`hybrid` 保持线上默认的向量 + 图谱并行检索。任何无 `metadata.source_document_id` 的上下文保留原文，但记录为 `invalid_provenance`，不参与 ID-based 指标。

runner 还会为每条记录保存 `category`、`expected_refusal` 和显式 `refused` 标记，并在 `run_metadata.json` 保存 `smoke_threshold`。`refused` 只由受控的 `insufficient_verified_evidence` 降级契约产生，不从任意回答文本猜测。

## 生成离线质量报告与发布门禁

质量聚合器只读取本地 `run_metadata.json` 和 `responses.jsonl`，不连接 LLM、Ragas、向量库或图数据库：

```bash
UV_CACHE_DIR=.uv-cache uv --directory backend run \
  python evaluation/scripts/run_quality_gate.py \
  --metadata-json evaluation/results/<run_id>/run_metadata.json \
  --responses-jsonl evaluation/results/<run_id>/responses.jsonl \
  --output-json evaluation/results/<run_id>/quality_report.json
```

报告包含：成功/失败/无 provenance 计数、provenance 完整度、按文档 ID 的 Context Precision/Recall、拒答 Precision/Recall/F1、延迟 p50/p95，以及按 retrieval mode/category 的分层汇总。ID-based 指标只使用 provenance 完整的成功记录；缺失值保持为 `null`，不静默当作 0。

少于每个比较组默认 30 条已完成记录的运行会标记为 `smoke_only`，且不会输出统计显著性结论。company-demo 的 30 条题已升级为确定性金标，但仍属于 demo 语料规模；在扩大并人工确认样本、做分层与置信区间之前，不输出显著性结论。

需要把结构完整性和失败样本纳入发布判断时，加上 `--release-gate`：

```bash
UV_CACHE_DIR=.uv-cache uv --directory backend run \
  python evaluation/scripts/run_quality_gate.py \
  --metadata-json evaluation/results/<run_id>/run_metadata.json \
  --responses-jsonl evaluation/results/<run_id>/responses.jsonl \
  --output-json evaluation/results/<run_id>/quality_report.json \
  --release-gate
```

门禁对缺少 metadata、失败记录、`invalid_provenance`、缺失 retrieval mode 分层或 `smoke_only` 运行 fail-closed，并以结构化 reason 返回非零退出码。聚合报告不复制问题、回答、上下文、绝对路径、secret 或异常消息。

## Agent trace 与离线契约检查

`QAAgent.answer()` 现在会在内部 `QAResult.trace` 保存当前请求实际执行过的工作流事件。trace 是 request-local、owner-only 评测证据，不是公开 HTTP 字段，也不是自主 LLM tool call；事件只记录操作名、状态、耗时、计数、模式和受控异常类型，不保存问题、回答、prompt、上下文、token、secret 或绝对路径。

```bash
UV_CACHE_DIR=.uv-cache uv --directory backend run   python evaluation/scripts/run_agent_contract.py
```

该命令只校验 `agent_benchmark.jsonl` 的版本、字段和操作名，并用确定性 trace fixture 验证通过/失败/无效 trace 分类；不会调用 LLM、Neo4j、ChromaDB、Langfuse 或其他外部服务。`BenchmarkRunner` 的 `responses.jsonl` 会保留完整 QA 上下文以及安全的 trace 摘要，便于后续把真实运行记录接入 Agent 目标/操作评测。

当前 benchmark 的 `expected_operations` 描述的是 KnowledgeForge 已存在的 Python 工作流边界（例如 `retrieval.vector`、`retrieval.graph`），不能被解读成模型自主工具调用。只有未来引入有明确参数/结果契约的真实工具接口后，才可另行评估 Tool Call Accuracy/F1。

## 配对 bootstrap（仅获批离线分数）

达到每个比较组至少 30 条人工审核题、并完成组织审批后，才可对已经保存的配对分数运行默认 1,000 次 bootstrap。输入只允许包含稳定 `question_id`、`vector_score` 和 `hybrid_score`，不会读取回答、上下文、PDF 或调用 judge：

```bash
UV_CACHE_DIR=.uv-cache uv --directory backend run \
  python evaluation/scripts/run_bootstrap.py \
  --scores-jsonl evaluation/results/<run_id>/paired_scores.jsonl \
  --output-json evaluation/results/<run_id>/bootstrap_report.json \
  --resamples 1000 \
  --seed 0
```

输出包含 hybrid-minus-vector 的观测均值差、95% 百分位区间和是否稳定跨过 0 的布尔结果。区间跨 0 时只能报告“未观察到稳定收益”；工具不会把 smoke run 或缺失分数静默转换成统计结论。

## 对比 fixed / recursive / semantic 切块

这条链路不连接生产向量库，而是在内存中对同一份 normalized 中文语料执行三档切块、同一 embedding 和同一 Top-K 余弦检索：

- `fixed`：纯字符基线，400 字符、128 字符重叠，不识别任何句界；
- `recursive`：生产默认，400/128，并按段落、换行和中文标点优先切分；
- `semantic`：生产灰度算法，中文分句、三句窗口、95 分位断点、200/512 字符硬边界；短文或 embedding 异常会记录为实际 `recursive` 降级。

先确认 `data/company-demo/normalized/` 已由前述清洗命令生成，且 manifest 中每份 normalized 文件的哈希一致。真实运行使用在线 DashScope-compatible embedding，因此需要在本地进程环境设置 `DASHSCOPE_API_KEY`，会产生 embedding 调用成本：

```bash
UV_CACHE_DIR=.uv-cache uv --directory backend run \
  python evaluation/scripts/run_chunking_benchmark.py \
  --corpus-root evaluation/data/company-demo \
  --top-k 5
```

runner 在 `evaluation/results/chunking/<run_id>/` 写入：

- `chunking_snapshot.jsonl`：逐题逐策略的完整 Top-K 文本、文档 ID、字符范围、相似度、请求策略与实际策略；
- `chunking_summary.json`：语料/题目哈希、模型名、统一参数，以及每种策略的块数、最小/平均/最大字符数和降级计数。

缺失 normalized 文件、路径逃逸、哈希不符、空文本或非法/零向量都会在产出质量分数前失败。结果目录为 `0700`、文件为 `0600`；快照包含完整内部文档片段，不得提交或放入不受信任的共享卷。

随后只对冻结快照运行 Ragas Context Precision / Context Recall，不会重新切块或检索：

```bash
UV_CACHE_DIR=.uv-cache uv --directory backend run --group eval \
  python evaluation/scripts/run_ragas_benchmark.py \
  --responses-jsonl evaluation/results/chunking/<run_id>/chunking_snapshot.jsonl \
  --metrics context_precision context_recall
```

输出 `ragas_context_quality.jsonl` 与 `ragas_context_quality.summary.json`。前者保留逐题逐指标的 scored/skipped/failed 结果，后者只聚合各指标的数量和均值。Ragas Context Precision / Recall 使用问题、人工 reference 和完整检索文本进行 judge；`quality_gates.py` 中已有的 ID-based precision/recall 只比较来源文档 ID，两者不可互换。

company-demo 当前包含 80 题，其中新增 50 题为规则生成、待人工复核候选。应先检查逐题失败、策略降级和块大小分布，再解释均值；没有扩大并审核样本、做分层与置信区间之前，不得声称某策略显著更优，也不会由脚本自动修改生产默认配置。

## 对保存的上下文运行 Ragas

只有先生成 `responses.jsonl` 后，才使用 Ragas judge 打分：

```bash
UV_CACHE_DIR=.uv-cache uv --directory backend run --group eval \
  python evaluation/scripts/run_ragas_benchmark.py \
  --responses-jsonl evaluation/results/<run_id>/responses.jsonl
```

不传 `--metrics` 时继续默认只运行 Faithfulness，并在同一目录写入 `ragas_faithfulness.jsonl`，保持原调用兼容。Ragas 样本评分默认每个指标最多同时发起 **2** 个 Judge 请求；可按评测服务器与提供方限流，通过 `--concurrency 1` 至 `--concurrency 4` 调整。4 核 4G 服务器建议保持默认值，或在遇到限流/内存压力时降为 `1`；不要使用超过 4 的并发。该适配器仅把以下内容传给 Ragas：

- `user_input`：人工审核后的 benchmark question；
- `response`：对应 `QAResult.answer`；
- `retrieved_contexts`：完整、按返回顺序的 `QAResult.contexts[*].content`。

runner 创建结果根目录和 run 目录时强制使用 owner-only 权限，所有 JSON/JSONL 通过临时文件原子写入并固定为 `0600`。这是进程用户边界，不是静态加密：不要把结果放在不受信任的共享卷，也不要让未获批的外部 judge 读取；加密/KMS、集中保留期和 DLP/SIEM 接入仍需按部署环境另行决策。

Faithfulness 可以在没有 reference 的情况下作为首轮 smoke test，衡量回答是否被这次检索上下文支持。它不等于答案正确性、上下文召回或 GraphRAG 的收益。保存记录包含 reference 时，也可显式选择 `context_precision` / `context_recall`；任何 `invalid_provenance` 记录仍必须跳过 ID-based 指标。

## 当前与后续评测边界

- **当前实现：**已审核金标加载/校验、vector/hybrid 快照、fixed/recursive/semantic 切块检索快照、完整上下文保存、request-local 真实 QA trace、Agent benchmark 契约、离线 trace 检查、provenance 失效记录、Ragas Faithfulness/Context Precision/Context Recall 适配、离线质量指标聚合和 fail-closed 发布门禁。
- **HTTP / 前端边界：**管理后台继续只调用默认 hybrid 的 `/api/v1/qa/ask`，并展示 200 字符来源预览；它不能启动离线 benchmark，也不能作为 Ragas 的上下文输入。自 2026-08-01 起，管理后台新增只读"评测结果"页面（`/api/v1/evaluation/runs` 系列端点，仅 `organization_admin`），用于查看已保存的评测运行、指标与逐题结果；该入口不触发评测、不写入评测产物，评测产物仍保持离线文件边界。
- **Agent 评测：**已实现内部工作流 trace 和 Agent benchmark/契约检查；尚未实现自主多消息工具调用，因此仍不声称已完成 Ragas Tool Call Accuracy/F1。不得根据 `reasoning_steps` 伪造轨迹。
- **Prompt 评测：**尚未实现 A/B runner。未来必须冻结本 runner 产生的同一检索快照，只变更回答 prompt，不能重新检索或同时变更模型/Top-K。
- **对外结论：**company-demo 的 30 条题已完成金标升级，但仍属于 demo 语料，须经人工确认章节定位、版本状态和正式制度后才可用于正式基准。扩展到至少 30 条/比较组的人工审核题并报告分层结果和置信区间之前，不得声称 GraphRAG 显著优于 Vector RAG；本轮门禁也不会为 smoke run 计算显著性。

### 安全事件审计边界

QA 安全拒答、检索/生成依赖 breaker 打开和 Webhook 目标或投递拒绝均写入现有追加式审计存储，使用稳定 action：`security.qa_safety_refusal`、`security.breaker_open`、`security.webhook_rejected`。事件只保留租户/用户/请求标识、问题指纹、依赖或状态类别等受控元数据，不保存原问题、回答、上下文、URL、响应体、异常文本或密钥；已拒绝的操作在审计写入失败时仍保持拒绝。该覆盖不代表已接入外部 SIEM、邮件、Slack 或生产告警调度。

详细的范围、指标、Agent/Prompt 后续阶段和质量门禁见 [RAG、GraphRAG、Agent 与 Prompt 的 Ragas 评测实施计划](../../reference/rag-graphrag-ragas-evaluation-plan.md)。

## 8. Evidence-gate benchmark and promotion gate

The evidence-gate bundle lives in `evaluation/data/evidence-gates/`. Its default
loader validates the manifest hash, cases hash, review-record hash, complete
category coverage, unique case IDs, declared response/evidence states, and full
human-review coverage before a run can be scored:

```python
from pathlib import Path
import json
from evaluation.evidence_gate_benchmark import load_evidence_gate_benchmark

root = Path("evaluation/data/evidence-gates")
baseline = json.loads((root / "baseline-template.json").read_text())
dataset = load_evidence_gate_benchmark(
    root / "manifest.json",
    expected_manifest_sha256=baseline["dataset"]["manifest_sha256"],
)
```

`require_reviewed=False` is only for authoring/linting candidate cases. It MUST
NOT be used for baseline measurement, shadow promotion, or enforce promotion.
As of 2026-08-08 the checked-in bundle contains 50 candidates across all 11
required categories: 10 `fully_answerable` cases and 4 cases for each other
category. The 30 company-demo-derived questions plus 20 fixture-driven cases are
bound to deterministic `<source_document_id>#chunk-<index>` IDs from 707 frozen
recursive chunks. `automated-pre-review.json` passes deterministic identity,
coverage, source-binding, and fixture-declaration checks, but
`review_status=pending_human_review` remains authoritative. The baseline is
unmeasured and the serving gate remains `off`.

Generate or refresh the reviewer packet with:

```bash
UV_CACHE_DIR=.uv-cache uv --directory backend run \
  python evaluation/scripts/prepare_evidence_gate_candidates.py \
  --company-demo-root evaluation/data/company-demo \
  --output-root evaluation/data/evidence-gates
```

The output includes `context-catalog.jsonl`, `pre-review-report.json`,
`automated-pre-review.json`, `review-checklist.md`, and a non-secret
`fixture-profile.example.json`. It is retained only as historical fixture
catalog metadata. It is not copied into a runtime configuration, does not accept
tenant IDs, and has no `verified` checkbox gate. Company administrators freeze
reviewed content, then start the separate backend Fixture validation run; its
Worker resolves the restricted finance scope server-side and records bounded
per-Fixture results for the exact frozen manifest.

The benchmark command always writes `runtime/runtime-preflight.json`. If review,
credentials, the authenticated company namespace, or services are unavailable,
it exits with code 2 and keeps `measured=false` instead of manufacturing metrics:

```bash
UV_CACHE_DIR=.uv-cache uv --directory backend run \
  python evaluation/scripts/run_evidence_gate_benchmark.py \
  --manifest evaluation/data/evidence-gates/manifest.json \
  --baseline-template evaluation/data/evidence-gates/baseline-template.json \
  --output-root evaluation/data/evidence-gates/runtime \
  --env-file ../config/.env \
  --tenant-id <company-internal-namespace>
```

The default run covers `off`, `shadow`, and `enforce`. It probes Chroma/PGVector
and Neo4j, uses the tenant BM25 snapshot, disables QA caches, dispatches the
BM25-only and all-branch outage fixtures through failing dependency adapters,
and uses the fixed company internal namespace for the authorization fixture. A
successful run writes per-mode runner artifacts and `quality-report.json`.

The checked-in preflight currently reports
`blocked_missing_runtime_configuration`, and the checked-in calibration record
reports `blocked_missing_measurements`; neither contains fabricated values.

Evidence-aware runner records add bounded replay fields without copying raw
question, claim, citation source text, tenant ID, endpoint, token, or provider
payload into the replay stages. The run identity includes dataset/manifest,
model, retrieval, threshold, candidate-budget, reranker, structured schema,
grounding policy, date, and code versions. Per-case resources include latency,
CPU time, and peak RSS.


Build a calibration draft from real reports only:

```bash
UV_CACHE_DIR=.uv-cache uv --directory backend run \
  python evaluation/scripts/calibrate_evidence_gate.py \
  --quality-report <off-quality-report.json> \
  --quality-report <shadow-quality-report.json> \
  --quality-report <enforce-quality-report.json> \
  --output-dir evaluation/data/evidence-gates/calibration \
  --calibration-version <approved-version>
```

Missing metrics produce `blocked_missing_measurements` and no candidate threshold
file. Complete metrics produce a candidate threshold file with status
`completed`; this evidence does not change the production Gate.

After a real frozen-corpus validation run, provide a flat JSON threshold
object to the shared quality-gate CLI:

```json
{
  "refusal_precision": 0.98,
  "refusal_recall": 0.98,
  "no_answer_hallucination_rate": 0.01,
  "answerable_false_refusal_rate": 0.02,
  "partial_answer_recognition_rate": 0.95,
  "conflict_recognition_rate": 0.98,
  "claim_citation_coverage": 1.0,
  "citation_correctness": 0.99,
  "groundedness_pass_rate": 0.99
}
```

These values are an example shape, not an approved calibration. The approved
threshold file must be derived from reviewed baseline results and versioned in
the run identity. The gate fails when any required metric is unavailable or
misses its threshold, and reports only bounded failing sample IDs.

```bash
UV_CACHE_DIR=.uv-cache uv --directory backend run \
  python evaluation/scripts/run_quality_gate.py \
  --metadata-json evaluation/results/<run_id>/run_metadata.json \
  --responses-jsonl evaluation/results/<run_id>/responses.jsonl \
  --output-json evaluation/results/<run_id>/quality_report.json \
  --smoke-threshold 30 \
  --release-gate \
  --evidence-thresholds-json /approved/path/evidence-thresholds.json
```

## 证据门数据集治理工作台

管理后台 `/evaluation` 现在提供“数据集工作台”，用于把候选样本制作、Context 绑定、maker-checker 复核和不可变冻结放到受权限保护的企业治理流程中。浏览器不会直接修改仓库中的 JSON/JSONL 文件：所有写操作经认证 API 进入组织隔离的 authoring workspace。

默认边界如下：

```text
前端评测治理中心
  → organization_admin 角色
  → 后端组织隔离 workspace
  → revision 乐观锁
  → 制作者提交
  → 另一管理员批准/退回
  → fail-closed 冻结与 SHA-256 manifest
  → 独立离线 baseline/calibration
```

authoring workspace 默认位于 `backend/data/evidence-review/`，组织目录使用不可逆命名空间，数据集 ID 经过白名单校验。首次访问会从 `backend/evaluation/data/evidence-gates/` 导入候选样本、Context catalog 和仅作历史展示的 fixture profile 模板。工作区保存 `workspace.json`、`cases.jsonl`、`context-catalog.jsonl`、`fixture-profile.json`、追加式 `review-events.jsonl` 和 `versions/<version>/`；冻结版本下的 `fixture-validation-runs/` 只保存有界运行结果；写入使用临时文件和原子替换。

治理规则：

- 所有评测治理读取和变更仅限当前认证组织的 `organization_admin`；
- 客户端必须提交最新 workspace revision，陈旧修改返回 409，不会覆盖其他管理员结果；
- 修改样本会使旧批准失效并回到 `draft`；
- `draft → pending_review → approved`，拒绝会回到 `draft`；
- 最后一位制作者不能批准自己的修改，审核人和审核时间只取服务端认证上下文；
- 审计事件只保存 actor、org、dataset/case/version、revision、结果和有界 reason code，不保存问题正文、Context 正文或凭据；
- 只有全部样本批准、11 类完整和 Context 有效时才能冻结；冻结不要求手工 Fixture 配置；
- API 冻结响应只返回版本、revision 和 manifest SHA-256，不暴露服务端绝对路径。

前端批准或冻结**不会**运行 benchmark、配置模型/API Key、修改 `QA_EVIDENCE_GATE_MODE`，也不会自动执行 `off → shadow → enforce` 晋级。真实 baseline/calibration 仍按本 README 的离线 runner、质量门和独立审批流程执行；只有工作台生成的冻结版本可作为正式测量输入，草稿 workspace 只能用于预览和 lint。

## 公司管理员发布评测

评测治理页面由**公司管理员**（角色码 `organization_admin`）使用；部门负责人和员工不具有发布评测或生产 Gate 权限。系统服务一间固定公司：不会在界面或 API 请求中要求提供组织号/租户 ID，内部命名空间仅保留给认证、审计与数据隔离兼容。

发布材料必须先在数据集工作台完成 maker-checker 审核并冻结。冻结产生不可变版本和 Manifest SHA-256，**不运行 benchmark，也不会修改 `QA_EVIDENCE_GATE_MODE`**。冻结后所有写入会返回 `dataset_frozen`；如需修改，必须从指定版本创建新草稿。

公司管理员在“发布评测”页可发起以下 Worker 工作流：

```text
冻结版本 → preflight → baseline/off → shadow → calibration 草案
→ 第二位公司管理员复核 → 单独确认 Gate 晋级
```

Worker 直接调用 Python 评测、质量聚合和 calibration API，不调用 shell。运行结果和尝试记录保存为无题目、无 Context 正文、无凭据、无服务器绝对路径的受控元数据。Gate 是公司全局的持久化配置，只能按 `off → shadow → enforce` 晋级；回滚逐级追加历史。环境变量 `QA_EVIDENCE_GATE_MODE` 仅是持久化配置不可读时的保守启动默认值，前端不能编辑 `.env`、Docker 或容器。

详细操作步骤与回滚说明见 `docs/company-admin-evaluation-guide.md`。
