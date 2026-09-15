"""Acceptance coverage for the role-aware authorization model."""

import pytest

from auth.bootstrap import promote_organization_admin
from auth.jwt_service import AUTHORIZATION_VERSION, AuthService
from auth.user_service import IdentityServiceError, UserService
from domain.identity import Permission, ROLE_PERMISSIONS, TokenPayload, UserRole


def test_employee_roles_are_read_only() -> None:
    """Legacy employee roles must no longer retain content-write capabilities."""
    denied = {
        Permission.DOC_WRITE,
        Permission.DOC_DELETE,
        Permission.DOC_EXPORT,
        Permission.QA_FEEDBACK,
        Permission.GRAPH_EDIT,
        Permission.ADMIN_MANAGE,
        Permission.ADMIN_USER,
        Permission.ADMIN_AUDIT,
    }

    for role in (UserRole.VIEWER, UserRole.EDITOR):
        permissions = set(ROLE_PERMISSIONS[role])
        assert Permission.DOC_READ in permissions
        assert Permission.QA_QUERY in permissions
        assert Permission.QA_HISTORY in permissions
        assert Permission.GRAPH_READ in permissions
        assert permissions.isdisjoint(denied)


def test_department_manager_cannot_govern_the_organization() -> None:
    """The legacy admin role now represents a department manager only."""
    permissions = set(ROLE_PERMISSIONS[UserRole.ADMIN])

    assert {
        Permission.DOC_READ,
        Permission.DOC_WRITE,
        Permission.DOC_DELETE,
        Permission.DOC_EXPORT,
        Permission.QA_QUERY,
        Permission.QA_HISTORY,
        Permission.QA_FEEDBACK,
        Permission.GRAPH_READ,
        Permission.GRAPH_EDIT,
    }.issubset(permissions)
    assert permissions.isdisjoint(
        {Permission.ADMIN_MANAGE, Permission.ADMIN_USER, Permission.ADMIN_AUDIT}
    )


def test_only_organization_admin_receives_governance_permissions() -> None:
    """Organization governance is exclusive to the new elevated role."""
    permissions = set(ROLE_PERMISSIONS[UserRole.ORGANIZATION_ADMIN])

    assert permissions == set(Permission)
    assert ROLE_PERMISSIONS[UserRole.API_USER] == []


def test_stale_authorization_version_is_rejected() -> None:
    """Tokens from the previous permission matrix cannot retain stale grants."""
    service = AuthService()
    stale = TokenPayload(
        sub="user-1",
        name="alice",
        role=UserRole.ADMIN.value,
        org_id="org-1",
        permissions=[Permission.ADMIN_MANAGE.value],
        exp=2_000_000_000,
        iat=1_900_000_000,
        authorization_version=AUTHORIZATION_VERSION - 1,
    )

    assert service.is_authorization_current(stale) is False


def _users() -> dict[str, dict]:
    return {
        "governor": {
            "user_id": "user-governor",
            "username": "governor",
            "password_hash": "unused",
            "display_name": "Governor",
            "email": "",
            "role": UserRole.ORGANIZATION_ADMIN,
            "org_id": "org-1",
            "department_id": None,
            "is_department_manager": False,
            "is_active": True,
            "token_version": 0,
        },
        "manager": {
            "user_id": "user-manager",
            "username": "manager",
            "password_hash": "unused",
            "display_name": "Manager",
            "email": "",
            "role": UserRole.ADMIN,
            "org_id": "org-1",
            "department_id": "finance",
            "is_department_manager": True,
            "is_active": True,
            "token_version": 0,
        },
    }


def test_last_organization_admin_cannot_be_downgraded_or_disabled() -> None:
    service = UserService(users=_users())

    with pytest.raises(IdentityServiceError, match="最后一个组织管理员"):
        service.update_user(
            "user-governor",
            role=UserRole.ADMIN,
            actor_user_id="user-manager",
            scope_org_id="org-1",
        )
    with pytest.raises(IdentityServiceError, match="最后一个组织管理员"):
        service.toggle_active(
            "user-governor",
            actor_user_id="user-manager",
            org_id="org-1",
        )


def test_trusted_promotion_targets_explicit_organization_and_user() -> None:
    service = UserService(users=_users())

    result = promote_organization_admin(
        org_id="org-1",
        user_id="user-manager",
        user_service=service,
    )

    promoted = service.find_by_id("user-manager", org_id="org-1")
    assert result.user_id == "user-manager"
    assert result.org_id == "org-1"
    assert promoted["role"] == UserRole.ORGANIZATION_ADMIN
    assert promoted["department_id"] is None
    assert promoted["is_department_manager"] is False
    assert promoted["token_version"] == 1

    with pytest.raises(IdentityServiceError):
        promote_organization_admin(
            org_id="org-other",
            user_id="user-manager",
            user_service=service,
        )


def test_batch_cannot_create_department_manager_without_department() -> None:
    users = _users()
    users["employee"] = {
        "user_id": "user-employee",
        "username": "employee",
        "password_hash": "unused",
        "display_name": "Employee",
        "email": "",
        "role": UserRole.VIEWER,
        "org_id": "org-1",
        "department_id": None,
        "is_department_manager": False,
        "is_active": True,
        "token_version": 0,
    }
    service = UserService(users=users)

    with pytest.raises(IdentityServiceError, match="必须绑定部门"):
        service.batch_change_role(
            ["user-employee"],
            UserRole.ADMIN,
            org_id="org-1",
        )
