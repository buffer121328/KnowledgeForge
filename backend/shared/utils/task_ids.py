"""Stable tenant-scoped task identifier helpers."""

from __future__ import annotations

import hashlib
import uuid


def task_namespace(org_id: str) -> str:
    """Return the non-secret stable namespace embedded in task IDs."""

    return hashlib.sha256(org_id.encode("utf-8")).hexdigest()[:16]


def new_task_id(org_id: str) -> str:
    """Create a server-owned task ID scoped to one organization namespace."""

    if not org_id:
        raise ValueError("task ID requires an organization")
    return f"tsk_{task_namespace(org_id)}_{uuid.uuid4().hex}"


def task_belongs_to_org(task_id: str, org_id: str) -> bool:
    """Return whether a task ID carries the expected organization namespace."""

    return bool(org_id) and task_id.startswith(f"tsk_{task_namespace(org_id)}_")
