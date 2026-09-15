"""Application service layer consumed by agents, workflows and API routes.

子包按主题组织：documents（文档提交与生命周期协调）、qa（QA 检索/排序/生成协作）、
safety（内容安全与工具治理）、evidence（证据资格鉴定）。本层不得反向依赖 agents。
"""
