"""Feature registry for factory pattern."""

from typing import Type

from trading_rl.features.base import Feature, FeatureConfig


class FeatureRegistry:
    """Registry for feature implementations.

    Allows features to be created by name from configuration.
    """

    _registry: dict[str, Type[Feature]] = {}

    @classmethod
    def register(cls, feature_type: str, feature_class: Type[Feature]) -> None:
        """Register a feature implementation.

        Args:
            feature_type: Type identifier (e.g., "log_return")
            feature_class: Feature class to register
        """
        cls._registry[feature_type] = feature_class

    @classmethod
    def create(cls, config: FeatureConfig) -> Feature:
        """Create a feature from configuration.

        Args:
            config: Feature configuration

        Returns:
            Feature instance

        Raises:
            ValueError: If feature_type is not registered
        """
        feature_class = cls._registry.get(config.feature_type)
        if feature_class is None:
            raise ValueError(
                f"Unknown feature type: {config.feature_type}. "
                f"Available types: {list(cls._registry.keys())}"
            )
        return feature_class(config)

    @classmethod
    def list_features(cls) -> list[str]:
        """List all registered feature types.

        Returns:
            List of registered feature type names
        """
        return list(cls._registry.keys())


def register_feature(feature_type: str):
    """Decorator to register a feature implementation.

    Args:
        feature_type: Type identifier for the feature

    Example:
        @register_feature("log_return")
        class LogReturnFeature(Feature):
            ...
    """

    def decorator(feature_class: Type[Feature]) -> Type[Feature]:
        FeatureRegistry.register(feature_type, feature_class)
        return feature_class

    return decorator
