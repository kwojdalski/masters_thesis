"""Feature group resolution for composable feature selection.

Provides FeatureGroupResolver, which loads named feature groups from a
YAML configuration and resolves them into flat lists of FeatureConfig
instances. Supports group references, exclusions, and mix-and-match
composition.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from logger import get_logger
from trading_rl.features.base import FeatureConfig

logger = get_logger(__name__)


class FeatureGroupResolver:
    """Resolve named feature groups into flat lists of FeatureConfig instances.

    Feature groups are defined in a YAML file (e.g., feature_groups.yaml)
    with the structure::

        groups:
          imbalance:
            description: "Order book imbalance signals"
            features:
              - name: "hft_book_pressure_l0"
                feature_type: "book_pressure"
                normalize: true
                domain: "hft"
                params: {level: 0}
              ...

    Usage::

        resolver = FeatureGroupResolver.from_yaml("src/configs/features/feature_groups.yaml")
        configs = resolver.resolve(["imbalance", "fair_value"])
        pipeline = FeaturePipeline(configs)
    """

    def __init__(self, groups: dict[str, dict[str, Any]]) -> None:
        """Initialize with a mapping of group names to group definitions.

        Args:
            groups: Mapping of group name to group definition dict.
                Each dict must contain a "features" key with a list of
                feature config dicts.
        """
        self._groups = groups
        self._validate_groups()

    def _validate_groups(self) -> None:
        """Validate group structure on init."""
        for group_name, group_def in self._groups.items():
            if "features" not in group_def:
                raise ValueError(
                    f"Group '{group_name}' must contain a 'features' key. "
                    f"Found keys: {list(group_def.keys())}"
                )
            features = group_def["features"]
            if not isinstance(features, list):
                raise ValueError(
                    f"Group '{group_name}' features must be a list, "
                    f"got {type(features).__name__}"
                )
            for i, feat in enumerate(features):
                if "name" not in feat:
                    raise ValueError(
                        f"Group '{group_name}' feature at index {i} "
                        f"must contain a 'name' key."
                    )
                if "feature_type" not in feat:
                    raise ValueError(
                        f"Group '{group_name}' feature '{feat.get('name', i)}' "
                        f"must contain a 'feature_type' key."
                    )

    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> "FeatureGroupResolver":
        """Load feature groups from a YAML configuration file.

        Args:
            yaml_path: Path to the YAML file containing group definitions.

        Returns:
            FeatureGroupResolver instance.

        Raises:
            FileNotFoundError: If the YAML file does not exist.
            ValueError: If the YAML file has no 'groups' key.
        """
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Feature groups file not found: {yaml_path}")

        with yaml_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if not data or "groups" not in data:
            raise ValueError(
                f"Feature groups file must contain a top-level 'groups' key. "
                f"Found keys: {list(data.keys()) if data else 'none'}"
            )

        logger.info(
            "Loaded %d feature groups from %s",
            len(data["groups"]),
            yaml_path,
        )
        return cls(groups=data["groups"])

    def list_groups(self) -> list[str]:
        """Return the names of all available feature groups."""
        return list(self._groups.keys())

    def get_group_description(self, group_name: str) -> str:
        """Return the description of a feature group.

        Args:
            group_name: Name of the group.

        Returns:
            Group description string.

        Raises:
            KeyError: If the group does not exist.
        """
        if group_name not in self._groups:
            raise KeyError(
                f"Unknown feature group '{group_name}'. "
                f"Available groups: {self.list_groups()}"
            )
        return self._groups[group_name].get("description", "")

    def get_group_features(self, group_name: str) -> list[FeatureConfig]:
        """Return FeatureConfig instances for a single group.

        Args:
            group_name: Name of the group.

        Returns:
            List of FeatureConfig instances for the group.

        Raises:
            KeyError: If the group does not exist.
        """
        if group_name not in self._groups:
            raise KeyError(
                f"Unknown feature group '{group_name}'. "
                f"Available groups: {self.list_groups()}"
            )
        feature_dicts = self._groups[group_name]["features"]
        return [self._dict_to_feature_config(fd) for fd in feature_dicts]

    def resolve(
        self,
        group_names: list[str],
        exclude: list[str] | None = None,
    ) -> list[FeatureConfig]:
        """Resolve a list of group names into a flat list of FeatureConfig instances.

        Args:
            group_names: List of group names to include.
            exclude: Optional list of feature output names to exclude.
                These are matched against the output_name (e.g.,
                "feature_hft_book_pressure_l2").

        Returns:
            Deduplicated list of FeatureConfig instances, preserving
            insertion order.

        Raises:
            KeyError: If any group name is not found.
        """
        exclude_set = set(exclude) if exclude else set()
        seen_names: set[str] = set()
        configs: list[FeatureConfig] = []

        for group_name in group_names:
            group_configs = self.get_group_features(group_name)
            for cfg in group_configs:
                output_name = cfg.output_name or f"feature_{cfg.name}"
                if output_name in seen_names:
                    logger.debug(
                        "Skipping duplicate feature '%s' (already added from another group)",
                        output_name,
                    )
                    continue
                if output_name in exclude_set:
                    logger.debug(
                        "Skipping excluded feature '%s'", output_name
                    )
                    continue
                seen_names.add(output_name)
                configs.append(cfg)

        logger.info(
            "Resolved %d groups into %d features (excluded %d)",
            len(group_names),
            len(configs),
            len(exclude_set),
        )
        return configs

    @staticmethod
    def _dict_to_feature_config(d: dict[str, Any]) -> FeatureConfig:
        """Convert a feature config dict to a FeatureConfig instance.

        Handles type coercion for params and defaults for optional fields.
        """
        return FeatureConfig(
            name=d["name"],
            feature_type=d["feature_type"],
            params=d.get("params"),
            normalize=d.get("normalize", True),
            output_name=d.get("output_name"),
            domain=d.get("domain", "shared"),
        )

    def __repr__(self) -> str:
        group_names = self.list_groups()
        total_features = sum(
            len(self._groups[g]["features"]) for g in group_names
        )
        return (
            f"FeatureGroupResolver(groups={len(group_names)}, "
            f"total_features={total_features})"
        )