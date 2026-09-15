"""从本地 DOCX 集合准备一个无图片的 company-demo 部门。"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[2]  # backend 根目录
if str(BACKEND_ROOT) not in sys.path:
    # 脚本直接运行时把 backend 根加入 sys.path，确保能导入 evaluation 包
    sys.path.insert(0, str(BACKEND_ROOT))

from evaluation.company_demo.department_expansion import (
    CompanyDemoExpansionError,
    prepare_company_demo_department,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """解析命令行参数

    Args:
        argv: 命令行参数列表；为 None 时读取 sys.argv。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)  # 本地 DOCX 来源目录
    parser.add_argument("--corpus-root", type=Path, required=True)  # company-demo 语料库根目录
    parser.add_argument("--selection", type=Path, required=True)  # 选定清单 JSON 路径
    parser.add_argument("--report", type=Path)  # 可选的准备报告输出路径
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """命令行入口：准备部门语料并以 JSON 打印汇总，失败时向 stderr 报错并返回 1

    Args:
        argv: 命令行参数列表；为 None 时读取 sys.argv。
    """
    args = parse_args(argv)
    try:
        report = prepare_company_demo_department(
            args.source_root,
            args.corpus_root,
            selection_path=args.selection,
            report_path=args.report,
        )
    except CompanyDemoExpansionError as error:
        # 领域错误视为可预期失败：打印原因并以退出码 1 结束
        print(f"company-demo department preparation failed: {error}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "department": report.department,
                "candidate_count": report.candidate_count,
                "included_count": report.included_count,
                "skipped_count": report.skipped_count,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":  # 脚本入口：以 main() 返回值作为进程退出码
    raise SystemExit(main())
