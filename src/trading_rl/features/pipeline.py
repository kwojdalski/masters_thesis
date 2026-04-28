"""Feature pipeline for orchestrating feature creation and normalization."""

import pandas as pd

from logger import get_logger
from trading_rl.features.base import Feature, FeatureConfig
from trading_rl.features.registry import FeatureRegistry

logger = get_logger(__name__)


class FeaturePipeline:
    """Orchestrates feature creation and normalization.

    Key features:
    - Proper train/test split handling (fit on train, transform on both)
    - No data leakage from test set
    - Configuration-based feature selection
    - Extensible through feature registry

    Usage:
        # Create pipeline from config
        configs = [
            FeatureConfig(name="log_return", feature_type="log_return"),
            FeatureConfig(name="high", feature_type="high"),
        ]
        pipeline = FeaturePipeline(configs)

        # Fit on training data
        pipeline.fit(train_df)

        # Transform both train and test
        train_features = pipeline.transform(train_df)
        test_features = pipeline.transform(test_df)
    """

    def __init__(self, feature_configs: list[FeatureConfig]):
        """Initialize pipeline with feature configurations.

        Args:
            feature_configs: List of feature configurations
        """
        self.feature_configs = feature_configs
        self.features: list[Feature] = []
        self._is_fitted = False

        # Create feature instances
        for config in feature_configs:
            feature = FeatureRegistry.create(config)
            self.features.append(feature)

        logger.info("build feature pipeline n_features=%d", len(self.features))

    @classmethod
    def from_config_dict(cls, config_dict: list[dict]) -> "FeaturePipeline":
        """Create pipeline from list of config dictionaries.

        Args:
            config_dict: List of feature config dictionaries

        Returns:
            FeaturePipeline instance

        Example:
            configs = [
                {"name": "log_return", "feature_type": "log_return"},
                {"name": "rsi", "feature_type": "rsi", "params": {"period": 14}},
            ]
            pipeline = FeaturePipeline.from_config_dict(configs)
        """
        feature_configs = [FeatureConfig(**cfg) for cfg in config_dict]
        return cls(feature_configs)

    @classmethod
    def from_yaml(cls, config_path: str) -> "FeaturePipeline":
        """Create pipeline from YAML configuration file.

        Args:
            config_path: Path to YAML file with 'features' key containing list of feature configs

        Returns:
            FeaturePipeline instance

        Example:
            pipeline = FeaturePipeline.from_yaml("configs/features/sine_wave_price_action.yaml")

        YAML format:
            features:
              - name: "log_return"
                feature_type: "log_return"
                normalize: true
              - name: "rsi"
                feature_type: "rsi"
                params:
                  period: 14
                normalize: true
        """
        from pathlib import Path

        import yaml

        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Feature config file not found: {config_path}")

        logger.info("load feature pipeline path=%s", config_path)

        with config_file.open("r") as f:
            config_data = yaml.safe_load(f)

        if not config_data or "features" not in config_data:
            raise ValueError(
                f"Invalid feature config: missing 'features' key in {config_path}"
            )

        feature_list = config_data["features"]
        logger.info("load feature pipeline n_features=%d path=%s", len(feature_list), config_path)

        return cls.from_config_dict(feature_list)

    def fit(self, df: pd.DataFrame) -> "FeaturePipeline":
        """Fit normalization parameters on training data.

        CRITICAL: This should ONLY be called on training data to prevent data leakage.

        Args:
            df: Training DataFrame with OHLCV columns

        Returns:
            self for chaining
        """
        logger.info("fit feature pipeline")
        logger.debug("fit feature pipeline n_rows=%d n_cols=%d", *df.shape)

        # Validate required columns
        self._validate_columns(df)

        # Fit each feature
        for feature in self.features:
            feature.fit(df)
            logger.debug("fit feature name=%s", feature.get_output_name())

        self._is_fitted = True
        logger.info("fit feature pipeline complete")
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform data using fitted parameters.

        Can be called on both training and test data after fit().

        Args:
            df: DataFrame to transform

        Returns:
            DataFrame with only feature columns

        Raises:
            RuntimeError: If pipeline not fitted
        """
        if not self._is_fitted:
            raise RuntimeError(
                "Pipeline must be fitted before transform. Call fit() first."
            )

        logger.debug("transform data n_rows=%d n_cols=%d", *df.shape)

        # Validate required columns
        self._validate_columns(df)

        # Create output DataFrame
        result = pd.DataFrame(index=df.index)

        # Transform each feature
        for feature in self.features:
            output_name = feature.get_output_name()
            result[output_name] = feature.transform(df)
            logger.debug("transform feature name=%s", output_name)

        # Drop any remaining NaN rows
        rows_before = len(result)
        result = result.dropna()
        rows_after = len(result)

        if rows_before != rows_after:
            logger.debug(
                "transform drop nan n_dropped=%d n_rows_before=%d n_rows_after=%d",
                rows_before - rows_after, rows_before, rows_after,
            )

        logger.debug("transform output n_rows=%d n_cols=%d", *result.shape)
        return result

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in one step.

        WARNING: Should only be used on training data. For test data, call
        transform() separately to avoid data leakage.

        Args:
            df: Training DataFrame

        Returns:
            DataFrame with transformed features
        """
        return self.fit(df).transform(df)

    def get_feature_names(self) -> list[str]:
        """Get list of output feature column names.

        Returns:
            List of feature names
        """
        return [feature.get_output_name() for feature in self.features]

    def _validate_columns(self, df: pd.DataFrame) -> None:
        """Validate that DataFrame has required columns.

        Args:
            df: DataFrame to validate

        Raises:
            ValueError: If required columns are missing
        """
        required_cols = set()
        for feature in self.features:
            required_cols.update(feature.required_columns())

        missing_cols = required_cols - set(df.columns)
        if missing_cols:
            raise ValueError(
                f"DataFrame missing required columns: {sorted(missing_cols)}\n"
                f"Available columns: {sorted(df.columns)}"
            )

    @classmethod
    def from_groups(
        cls,
        groups_yaml_path: str,
        group_names: list[str],
        exclude: list[str] | None = None,
    ) -> "FeaturePipeline":
        """Create a pipeline from feature group definitions.

        Loads named feature groups from a YAML file and resolves them
        into a flat list of FeatureConfig instances, then constructs
        a FeaturePipeline.

        Args:
            groups_yaml_path: Path to the feature groups YAML file.
            group_names: List of group names to include.
            exclude: Optional list of feature output names to exclude.

        Returns:
            FeaturePipeline with features from the resolved groups.

        Example:
            pipeline = FeaturePipeline.from_groups(
                "src/configs/features/feature_groups.yaml",
                group_names=["imbalance", "fair_value", "flow"],
            )
        """
        from trading_rl.features.groups import FeatureGroupResolver

        resolver = FeatureGroupResolver.from_yaml(groups_yaml_path)
        configs = resolver.resolve(group_names, exclude=exclude)
        logger.info(
            "Built pipeline from groups %s: %d features",
            group_names,
            len(configs),
        )
        return cls(configs)

    def __repr__(self) -> str:
        """String representation of pipeline."""
        feature_names = self.get_feature_names()
        return (
            f"FeaturePipeline(features={len(self.features)}, "
            f"fitted={self._is_fitted}, "
            f"outputs={feature_names})"
        )


def create_default_pipeline() -> FeaturePipeline:
    """Create a feature pipeline with the standard default features.

    Returns:
        FeaturePipeline with default features:
        - feature_log_return: Log returns (normalized)
        - feature_high: High relative to close (normalized)
        - feature_low: Low relative to close (normalized)
        - feature_log_volume: Log volume (normalized)
        - feature_trend: Price trend from start (min-max scaled)

    Example:
        pipeline = create_default_pipeline()
        pipeline.fit(train_df)
        train_features = pipeline.transform(train_df)
        test_features = pipeline.transform(test_df)
    """
    logger.info("build default feature pipeline")

    configs = [
        FeatureConfig(
            name="log_return",
            feature_type="log_return",
            normalize=True,
            params={},
        ),
        FeatureConfig(
            name="high",
            feature_type="high",
            normalize=True,
            params={},
        ),
        FeatureConfig(
            name="low",
            feature_type="low",
            normalize=True,
            params={},
        ),
        FeatureConfig(
            name="log_volume",
            feature_type="log_volume",
            normalize=True,
            params={},
        ),
        FeatureConfig(
            name="trend",
            feature_type="trend",
            normalize=False,  # Uses custom min-max scaling
            params={},
        ),
    ]

    pipeline = FeaturePipeline(configs)
    logger.info("build default feature pipeline complete n_features=%d", len(configs))
    return pipeline
