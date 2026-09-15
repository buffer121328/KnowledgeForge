"""评测域共享的安全写盘工具。

``_write_text_secure`` 原在 ``evaluation/runner.py``，被 release runtime、
benchmark 脚本与 runner 三方借用；收敛到此处作为单一来源。旧的私有名通过
runner 的 re-export 兼容现有测试与调用方。
"""

from __future__ import annotations

import os
from uuid import uuid4
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    """以 JSON Lines 形式原子写出记录列表，权限 0600（先写排他临时文件再替换）。"""
    content = "".join(
        f"{json.dumps(record, ensure_ascii=False, sort_keys=True, default=str)}\n"
        for record in records
    )
    write_text_secure(path, content)


def write_text_secure(path: Path, content: str) -> None:
    """以 0600 权限原子写出文本文件（评测产物含租户语料摘录，不放宽读权限）。"""
    temp_path = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    descriptor = os.open(
        temp_path,
        os.O_CREAT | os.O_EXCL | os.O_WRONLY,
        0o600,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
        os.chmod(path, 0o600)
    except Exception:
        try:
            temp_path.unlink(missing_ok=True)
        finally:
            raise


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    """以缩进 JSON 写出映射（先写临时文件再替换，权限 0600）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
    os.replace(tmp, path)
