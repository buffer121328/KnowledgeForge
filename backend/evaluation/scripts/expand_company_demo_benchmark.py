"""为第四部门生成待人工复核的 company-demo 基准行。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[2]  # backend 根目录
if str(BACKEND_ROOT) not in sys.path:
    # 脚本直接运行时把 backend 根加入 sys.path，确保能导入 evaluation 包
    sys.path.insert(0, str(BACKEND_ROOT))

from evaluation.company_demo.benchmark_expansion import (
    expand_company_demo_benchmark,
)


def main() -> int:
    """命令行入口：解析参数、执行基准扩充并以 JSON 打印汇总结果。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-root", type=Path, required=True)  # company-demo 语料库根目录
    args = parser.parse_args()
    print(json.dumps(expand_company_demo_benchmark(args.corpus_root), ensure_ascii=False))
    return 0


if __name__ == "__main__":  # 脚本入口：以 main() 返回值作为进程退出码
    raise SystemExit(main())
