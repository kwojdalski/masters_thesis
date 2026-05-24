"""Show which config keys differ across two or more scenario directories.

Usage:
    uv run python scripts/diff_scenarios.py pooled/td3_hft_lob_state_space_pooled_streaming_selected_dsr \\
                                             pooled/ddpg_hft_lob_state_space_pooled_streaming_selected_dsr \\
                                             pooled/ppo_hft_lob_state_space_pooled_streaming_selected_dsr \\
                                             pooled/random_hft_lob_state_space_pooled_streaming_selected_dsr

    # Compare by short name (resolved under src/configs/scenarios/)
    uv run python scripts/diff_scenarios.py pooled/td3_h3_fees_1e4 pooled/td3_h3_fees_1e5 pooled/td3_h3_fees_1e6

    # Show ALL keys, not just differing ones
    uv run python scripts/diff_scenarios.py --all pooled/td3_... pooled/ddpg_...

    # Compare a specific YAML file within the scenario (default: train.yaml)
    uv run python scripts/diff_scenarios.py --file evaluate.yaml pooled/td3_... pooled/ddpg_...
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml
from rich.console import Console
from rich.table import Table

_SCENARIOS_ROOT = Path("src/configs/scenarios")

# Keys whose values are long lists — show length instead of full value
_SUMMARISE_LISTS = {"data_paths", "val_data_paths"}


def _resolve(name: str) -> Path:
    p = Path(name)
    if p.is_dir():
        return p
    candidate = _SCENARIOS_ROOT / name
    if candidate.is_dir():
        return candidate
    raise SystemExit(f"Scenario not found: {name!r}  (tried {p} and {candidate})")


def _load(scenario_dir: Path, filename: str) -> dict[str, Any]:
    yaml_path = scenario_dir / filename
    if not yaml_path.exists():
        return {}
    with yaml_path.open() as f:
        return yaml.safe_load(f) or {}


def _flatten(obj: Any, prefix: str = "") -> dict[str, Any]:
    """Recursively flatten a nested dict to dotted keys."""
    out: dict[str, Any] = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            full_key = f"{prefix}.{k}" if prefix else k
            out.update(_flatten(v, full_key))
    elif isinstance(obj, list) and prefix.split(".")[-1] in _SUMMARISE_LISTS:
        out[prefix] = f"[{len(obj)} paths]"
    else:
        out[prefix] = obj
    return out


def _str(v: Any) -> str:
    if v is None:
        return "[dim]—[/dim]"
    if isinstance(v, bool):
        return str(v).lower()
    if isinstance(v, float):
        # Show scientific notation for very small floats
        return f"{v:g}"
    return str(v)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Show config differences across scenario directories.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("scenarios", nargs="+", help="Scenario names or paths")
    parser.add_argument(
        "--all", "-a",
        action="store_true",
        help="Show all keys, not just differing ones",
    )
    parser.add_argument(
        "--file", "-f",
        default="train.yaml",
        help="YAML file to compare within each scenario dir (default: train.yaml)",
    )
    args = parser.parse_args()

    console = Console()

    if len(args.scenarios) < 2:
        console.print("[red]Need at least 2 scenarios to compare.[/red]")
        sys.exit(1)

    dirs = [_resolve(s) for s in args.scenarios]
    labels = [d.name for d in dirs]
    configs = [_flatten(_load(d, args.file)) for d in dirs]

    all_keys = sorted({k for cfg in configs for k in cfg})

    # Filter to only differing keys unless --all
    if not args.all:
        all_keys = [
            k for k in all_keys
            if len({_str(cfg.get(k)) for cfg in configs}) > 1
        ]

    if not all_keys:
        console.print("[green]All configs are identical.[/green]")
        return

    t = Table(show_header=True, header_style="bold", show_lines=False)
    t.add_column("Key", style="cyan", no_wrap=True)
    for label in labels:
        t.add_column(label, justify="left")

    for key in all_keys:
        values = [_str(cfg.get(key)) for cfg in configs]
        # Highlight cells that differ from the first column
        styled = []
        for i, v in enumerate(values):
            if i == 0 or v == values[0]:
                styled.append(v)
            else:
                styled.append(f"[yellow]{v}[/yellow]")
        t.add_row(key, *styled)

    title = f"Differing keys ({args.file})" if not args.all else f"All keys ({args.file})"
    console.print()
    console.print(f"[bold]{title}[/bold]  [dim]({len(all_keys)} keys shown)[/dim]")
    console.print(t)
    console.print()
    console.print("[dim]Yellow = differs from first scenario.[/dim]")


if __name__ == "__main__":
    main()
