"""运行已评审的证据门基线，或在无法真实测量时输出如实的阻断产物。"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Sequence

BACKEND_ROOT = Path(__file__).resolve().parents[2]  # backend 根目录
if str(BACKEND_ROOT) not in sys.path:  # 确保 backend 包可被直接导入
    sys.path.insert(0, str(BACKEND_ROOT))

from evaluation.evidence_gate.runtime import execute_evidence_gate_benchmark


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """解析命令行参数并返回 argparse 命名空间。

    Args:
        argv: 参数序列；None 时读取 sys.argv。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)  # 冻结清单路径
    parser.add_argument("--baseline-template", type=Path, required=True)  # 基线模板路径
    parser.add_argument("--output-root", type=Path, required=True)  # 运行产物输出根目录
    parser.add_argument("--tenant-id", default="")  # 租户 ID，空字符串表示不过滤
    parser.add_argument("--env-file", type=Path, default=Path("../config/.env"))  # 基准运行加载的环境文件
    parser.add_argument(
        "--gate-mode",
        action="append",
        choices=("off", "shadow", "enforce"),
        dest="gate_modes",
    )  # 可重复指定证据门模式；未提供时默认三种模式全部执行
    parser.add_argument(
        "--retrieval-mode",
        action="append",
        dest="retrieval_modes",
        default=None,
    )  # 可重复指定检索模式；未提供时默认 dense_bm25_graph
    parser.add_argument("--smoke-threshold", type=int, default=30)  # 冒烟模式的最小用例数阈值
    parser.add_argument(
        "--authoring",
        action="store_true",
        help="Allow pending-review data for pre-review discovery only.",
    )  # 授权 authoring 模式：仅用于评审前发现，允许使用待评审数据
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """执行证据门基准，并把完整结果以 JSON 打印到标准输出。

    Args:
        argv: 命令行参数序列；None 时读取 sys.argv。

    返回 0 表示已真实测量；2 表示被阻断（未测量）。
    """
    args = parse_args(argv)
    # ① 运行基准；未显式指定 gate/retrieval 模式时使用各自默认值。
    result = asyncio.run(
        execute_evidence_gate_benchmark(
            manifest_path=args.manifest,
            baseline_path=args.baseline_template,
            output_root=args.output_root,
            gate_modes=args.gate_modes or ("off", "shadow", "enforce"),
            tenant_id=args.tenant_id,
            authoring=args.authoring,
            env_file=args.env_file,
            retrieval_modes=args.retrieval_modes or ("dense_bm25_graph",),
            smoke_threshold=args.smoke_threshold,
        )
    )
    # ② 以 JSON 输出完整结果。
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    # ③ 返回码如实反映是否测量：只有 measured 为 True 才返回 0，禁止伪造测量结果。
    return 0 if result.get("measured") is True else 2


if __name__ == "__main__":
    # 以 main 的返回码退出，供 CI 判定测量是否真实完成。
    raise SystemExit(main())
