"""CLI for an explicit, tenant-scoped organization-administrator promotion."""

from __future__ import annotations

import argparse

from auth.bootstrap import AdminBootstrapError, promote_organization_admin
from auth.user_service import IdentityServiceError


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Promote one explicit tenant user to organization administrator",
    )
    parser.add_argument("--org-id", required=True)
    parser.add_argument("--user-id", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the bounded trusted promotion command."""
    args = _parser().parse_args(argv)
    try:
        result = promote_organization_admin(org_id=args.org_id, user_id=args.user_id)
    except (AdminBootstrapError, IdentityServiceError) as error:
        print(f"Promotion failed: {error}")
        return 1
    print(
        "Organization administrator promoted: "
        f"org_id={result.org_id} user_id={result.user_id} "
        f"token_version={result.token_version}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
