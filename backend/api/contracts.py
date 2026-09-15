"""Reusable HTTP-boundary contracts shared across API routers."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints


class StrictRequestModel(BaseModel):
    """Reject undeclared JSON fields without mutating secret-bearing strings."""

    model_config = ConfigDict(extra="forbid")


ResourceId = Annotated[
    str,
    StringConstraints(
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]*$",
        strip_whitespace=True,
    ),
]
RoleName = Annotated[
    str,
    StringConstraints(
        min_length=2,
        max_length=64,
        pattern=r"^[a-z][a-z0-9_]*$",
        strip_whitespace=True,
    ),
]
Username = Annotated[
    str,
    StringConstraints(
        min_length=3,
        max_length=32,
        pattern=r"^[A-Za-z0-9_-]+$",
        strip_whitespace=True,
    ),
]
LoginPassword = Annotated[str, StringConstraints(min_length=1, max_length=128)]
Password = Annotated[str, StringConstraints(min_length=6, max_length=128)]
BootstrapPassword = Annotated[str, StringConstraints(min_length=12, max_length=128)]
BearerToken = Annotated[str, StringConstraints(min_length=1, max_length=8192)]
InvitationToken = Annotated[str, StringConstraints(min_length=20, max_length=512)]
ShortName = Annotated[str, StringConstraints(min_length=1, max_length=128, strip_whitespace=True)]
DisplayName = Annotated[str, StringConstraints(max_length=255, strip_whitespace=True)]
Description = Annotated[str, StringConstraints(max_length=1000, strip_whitespace=True)]
FilePath = Annotated[str, StringConstraints(min_length=1, max_length=4096)]
SearchText = Annotated[str, StringConstraints(min_length=1, max_length=255, strip_whitespace=True)]
CursorToken = Annotated[str, StringConstraints(min_length=1, max_length=512)]
PageLimit = Annotated[int, Field(ge=1, le=100)]
BatchLimit = Annotated[int, Field(ge=1, le=50)]


class MessageResponse(BaseModel):
    """Return a stable human-readable mutation result."""

    message: str


class ToggleResponse(MessageResponse):
    """Return the result of enabling or disabling a resource."""

    is_active: bool


class TaskMutationResponse(BaseModel):
    """Return the stable task identifier and lifecycle status."""

    task_id: ResourceId
    status: str


class DeletedResourceResponse(BaseModel):
    """Return a stable deleted resource identifier."""

    status: Literal["deleted"] = "deleted"
    resource_id: ResourceId


def sanitize_validation_errors(errors: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remove submitted values from validation details before serialization."""

    return [
        {key: value for key, value in item.items() if key not in {"input", "ctx"}}
        for item in errors
    ]
