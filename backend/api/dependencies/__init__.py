"""FastAPI dependency providers."""

from .auth import get_current_user, require_permission, require_role
from .document_reads import DocumentReadError, DocumentReadProvider, get_document_read
from .documents import get_document_lifecycle, get_document_submission
from .qa import QARouteDependencies, get_qa_route_dependencies

__all__ = [
    "DocumentReadError",
    "DocumentReadProvider",
    "QARouteDependencies",
    "get_current_user",
    "get_document_lifecycle",
    "get_document_read",
    "get_document_submission",
    "get_qa_route_dependencies",
    "require_permission",
    "require_role",
]
