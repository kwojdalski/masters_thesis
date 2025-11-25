#!/usr/bin/env python3
"""Check YAML files have matching parquet data files"""

from pathlib import Path

import yaml


def check_consistency():
    config_dir = Path("src/configs")
    issues = []

    for yaml_file in config_dir.glob("*.yaml"):
        with open(yaml_file) as f:
            config = yaml.safe_load(f)

        data_path = config.get("data", {}).get("data_path", "")
        if not data_path:
            continue

        data_file = Path(data_path)
        if not data_file.exists():
            issues.append(f"❌ {yaml_file.name}: missing {data_file}")
        else:
            print(f"✅ {yaml_file.name}: {data_file.name}")

    if issues:
        print("\nIssues found:")
        for issue in issues:
            print(issue)
        return False

    print(
        f"\n✅ All {len(list(config_dir.glob('*.yaml')))} YAML files have matching data files"
    )
    return True


if __name__ == "__main__":
    check_consistency()
