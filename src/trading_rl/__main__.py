"""Package entrypoint for ``python -m trading_rl``.

This delegates to the existing project CLI implementation in ``src/cli.py``
to avoid duplicating command definitions.
"""

from __future__ import annotations

import runpy
from pathlib import Path


def main() -> None:
    """Execute the top-level CLI script as the module entrypoint."""
    cli_path = Path(__file__).resolve().parents[1] / "cli.py"
    runpy.run_path(str(cli_path), run_name="__main__")


if __name__ == "__main__":
    main()
