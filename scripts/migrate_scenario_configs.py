"""One-shot migration: split monolithic scenario YAMLs into component files.

For each scenario YAML that is NOT already inside a properly structured directory,
this script creates a directory with the same base name and writes:
  observation.yaml    — data.feature_config + env.feature_columns + env.mode
  train.yaml          — everything else
  evaluate.yaml       — benchmarks + statistical_testing (if present in source)

For already-split directories (have train.yaml but no observation.yaml), it just
extracts the feature keys out of train.yaml and evaluate.yaml into observation.yaml.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import yaml

SCENARIOS_DIR = Path("src/configs/scenarios")
FEATURE_KEYS_DATA = {"feature_config", "feature_groups"}
FEATURE_KEYS_ENV = {"feature_columns", "mode"}
EVAL_ONLY_KEYS = {"benchmarks", "statistical_testing"}


def _extract(d: dict, keys: set[str]) -> dict:
    return {k: v for k, v in d.items() if k in keys}


def _remove(d: dict, keys: set[str]) -> dict:
    return {k: v for k, v in d.items() if k not in keys}


def build_observation_yaml(config: dict) -> dict:
    out: dict = {}
    data_features = _extract(config.get("data", {}), FEATURE_KEYS_DATA)
    if data_features:
        out["data"] = data_features
    env_features = _extract(config.get("env", {}), FEATURE_KEYS_ENV)
    if env_features:
        out["env"] = env_features
    return out


def strip_observation_from(config: dict) -> dict:
    out = dict(config)
    if "data" in out:
        out["data"] = _remove(out["data"], FEATURE_KEYS_DATA)
        if not out["data"]:
            del out["data"]
    if "env" in out:
        out["env"] = _remove(out["env"], FEATURE_KEYS_ENV)
        if not out["env"]:
            del out["env"]
    return out


def build_evaluate_yaml(config: dict) -> dict:
    return _extract(config, EVAL_ONLY_KEYS)


def write_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    print(f"  wrote {path}")


def migrate_flat_yaml(yaml_path: Path) -> None:
    """Convert a monolithic scenario YAML into a component directory."""
    with open(yaml_path) as f:
        config = yaml.safe_load(f) or {}

    scenario_dir = yaml_path.parent / yaml_path.stem
    if scenario_dir.exists():
        print(f"  SKIP (directory already exists): {scenario_dir}")
        return

    scenario_dir.mkdir()

    features = build_observation_yaml(config)
    if features:
        write_yaml(scenario_dir / "observation.yaml", features)

    train = strip_observation_from(config)
    train_eval_only = _extract(train, EVAL_ONLY_KEYS)
    train_no_eval = _remove(train, EVAL_ONLY_KEYS)
    write_yaml(scenario_dir / "train.yaml", train_no_eval)

    if train_eval_only:
        write_yaml(scenario_dir / "evaluate.yaml", train_eval_only)

    yaml_path.unlink()
    print(f"  removed {yaml_path}")


def migrate_split_dir(scenario_dir: Path) -> None:
    """For a directory that has train.yaml but no observation.yaml, extract features."""
    train_path = scenario_dir / "train.yaml"
    evaluate_path = scenario_dir / "evaluate.yaml"
    observation_path = scenario_dir / "observation.yaml"

    if observation_path.exists():
        print(f"  SKIP (observation.yaml already exists): {scenario_dir}")
        return

    with open(train_path) as f:
        train_config = yaml.safe_load(f) or {}

    features = build_observation_yaml(train_config)
    if features:
        write_yaml(observation_path, features)

    # Strip feature keys from train.yaml
    stripped_train = strip_observation_from(train_config)
    write_yaml(train_path, stripped_train)

    # Strip feature keys from evaluate.yaml too (if it exists)
    if evaluate_path.exists():
        with open(evaluate_path) as f:
            eval_config = yaml.safe_load(f) or {}
        stripped_eval = strip_observation_from(eval_config)
        write_yaml(evaluate_path, stripped_eval)


def main() -> None:
    print(f"Migrating scenarios in {SCENARIOS_DIR}\n")

    for path in sorted(SCENARIOS_DIR.rglob("*.yaml")):
        # Skip the top-level default.yaml (kept as legacy fallback)
        rel = path.relative_to(SCENARIOS_DIR)
        parts = rel.parts

        # Already inside a command-component directory (has observation.yaml sibling)
        if (path.parent / "observation.yaml").exists() and path.name in (
            "train.yaml", "evaluate.yaml", "observation.yaml", "feature_selection.yaml"
        ):
            continue

        # Flat scenario YAML (e.g. aapl/td3_hft_lob.yaml)
        if path.is_file() and path.stem not in ("train", "evaluate", "observation", "feature_selection", "default"):
            print(f"Migrating flat: {rel}")
            migrate_flat_yaml(path)
            continue

    # Handle already-split directories (train.yaml + evaluate.yaml, no observation.yaml)
    for d in sorted(SCENARIOS_DIR.rglob("*")):
        if not d.is_dir():
            continue
        if (d / "train.yaml").exists() and not (d / "observation.yaml").exists():
            rel = d.relative_to(SCENARIOS_DIR)
            print(f"Extracting features from existing split: {rel}")
            migrate_split_dir(d)

    print("\nDone.")


if __name__ == "__main__":
    main()
