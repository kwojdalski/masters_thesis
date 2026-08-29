#!/usr/bin/env python3
"""Check YAML scenario configs have matching parquet data files.

Scenario configs live one directory deeper than the scenarios root (e.g.
``pooled/<scenario>/train.yaml``), and the pooled scenarios that drive the
thesis reference their data through the ``data_paths`` / ``val_data_paths``
lists rather than the single ``data_path`` string.  Both facts have to be
honoured or the check passes vacuously.
"""

import sys
from pathlib import Path

import yaml


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _data_path_candidates(config: dict) -> list[str]:
    """Return every data path a scenario config references.

    Covers the single-file form (``data_path``) and both list forms
    (``data_paths``, ``val_data_paths``).
    """
    data_cfg = (config or {}).get("data") or {}
    candidates: list[str] = []
    single = data_cfg.get("data_path")
    if single:
        candidates.append(str(single))
    for key in ("data_paths", "val_data_paths"):
        candidates.extend(str(p) for p in (data_cfg.get(key) or []))
    return candidates


def check_consistency(config_dir: Path | None = None) -> bool:
    # Config paths are written relative to the repo root ("./data/raw/..."),
    # so resolve against it rather than the caller's cwd -- otherwise every
    # path reads as missing when the script is run from anywhere else.
    root = _repo_root()
    config_dir = config_dir or root / "src" / "configs" / "scenarios"
    issues: list[str] = []
    n_checked = 0
    n_paths = 0

    # rglob, not glob: every real scenario config is nested at least one level
    # below the scenarios root, so a flat glob inspects only default.yaml.
    for yaml_file in sorted(config_dir.rglob("*.yaml")):
        try:
            with open(yaml_file) as f:
                config = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            issues.append(f"{yaml_file}: unparseable YAML ({exc})")
            continue

        candidates = _data_path_candidates(config)
        if not candidates:
            continue

        n_checked += 1
        rel = yaml_file.relative_to(config_dir)
        for path in candidates:
            n_paths += 1
            resolved = Path(path)
            if not resolved.is_absolute():
                resolved = root / resolved
            if not resolved.exists():
                issues.append(f"{rel}: missing {path}")

    if issues:
        print(f"Checked {n_checked} config(s), {n_paths} data path(s).")
        print(f"\n{len(issues)} issue(s) found:")
        for issue in issues:
            print(f"  {issue}")
        return False

    print(
        f"All {n_paths} data path(s) across {n_checked} config(s) "
        f"under {config_dir} exist."
    )
    return True


if __name__ == "__main__":
    sys.exit(0 if check_consistency() else 1)
