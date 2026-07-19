"""Repository entry point for the fail-safe Q2 publication workflow."""
from __future__ import annotations

from typing import Sequence

from isrs_scl.cli import main as cli_main


def main(argv: Sequence[str] | None = None) -> int:
    """Run the shared installed CLI so both entry points remain identical."""
    return cli_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
