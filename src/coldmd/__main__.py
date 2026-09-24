"""Allow ``python -m coldmd`` to invoke the command-line interface."""

from .cli import main


if __name__ == "__main__":  # pragma: no cover - exercised through the CLI.
    raise SystemExit(main())
