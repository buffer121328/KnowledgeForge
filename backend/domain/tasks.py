from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum

__all__ = ["ChangeType", "DocumentChange", "UpdateResult"]


class ChangeType(str, Enum):
    """Represent change type."""
    CREATED = "created"
    MODIFIED = "modified"
    DELETED = "deleted"


@dataclass
class DocumentChange:
    """Represent document change."""
    file_path: str
    change_type: ChangeType
    timestamp: float = field(default_factory=time.time)
    old_hash: str = ""
    new_hash: str = ""
    diff_chunks: list[str] = field(default_factory=list)


@dataclass
class UpdateResult:
    """Represent the result of update processing."""
    change: DocumentChange
    vectors_added: int = 0
    vectors_deleted: int = 0
    entities_added: int = 0
    entities_updated: int = 0
    relations_added: int = 0
    success: bool = True
    error: str = ""
    processing_time_ms: float = 0
