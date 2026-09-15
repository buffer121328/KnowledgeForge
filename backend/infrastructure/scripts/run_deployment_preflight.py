"""Repository entrypoint for the static parser deployment preflight."""

from infrastructure.deployment_preflight import main


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
