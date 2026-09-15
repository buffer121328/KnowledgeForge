"""CLI entrypoint for one-time durable administrator bootstrap."""

from __future__ import annotations

import argparse
import getpass

from auth.bootstrap import AdminBootstrapError, bootstrap_initial_admin


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create the first PostgreSQL-backed administrator")
    parser.add_argument("--username", required=True)
    parser.add_argument("--email", required=True)
    parser.add_argument("--org-id", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the secret-safe bootstrap command."""
    args = _parser().parse_args(argv)
    password = getpass.getpass("Administrator password: ")
    confirmation = getpass.getpass("Confirm administrator password: ")
    if password != confirmation:
        print("Bootstrap failed: password confirmation does not match")
        return 2
    try:
        result = bootstrap_initial_admin(
            username=args.username,
            email=args.email,
            org_id=args.org_id,
            password=password,
        )
    except AdminBootstrapError as error:
        print(f"Bootstrap failed: {error}")
        return 1
    print(
        "Administrator created: "
        f"user_id={result.user_id} username={result.username} org_id={result.org_id}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
