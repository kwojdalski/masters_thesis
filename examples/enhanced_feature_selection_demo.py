"""Demonstration of enhanced feature selection capabilities.

This script shows how to use the new feature selection features:
1. Time-series cross-validation
2. Multi-horizon composite scoring
3. Automated hyperparameter search
4. Ensemble selection across CV splits
"""

import pandas as pd
from trading_rl.features.groups import FeatureGroupResolver
from trading_rl.features.selector import FeatureSelector, FeatureSelectorConfig
from trading_rl.features.pipeline import FeaturePipeline
from trading_rl.features.base import FeatureConfig


def example_standard_selection():
    """Standard single-split feature selection."""
    print("\n=== Standard Single-Split Selection ===")

    # Load feature groups
    resolver = FeatureGroupResolver.from_yaml("src/configs/features/feature_groups.yaml")
    candidates = resolver.resolve(["imbalance", "fair_value"])

    # Standard selection
    config = FeatureSelectorConfig(
        top_k=8,
        horizon=1,
        icir_threshold=0.02,
    )
    selector = FeatureSelector(config)

    # Load your data (replace with actual data loading)
    # train_df = pd.read_pickle("path/to/train.pkl")
    # val_df = pd.read_pickle("path/to/val.pkl")

    # result = selector.select(candidates, train_df, val_df)
    # FeatureSelector.write_selected_yaml(result, "configs/features/selected.yaml")


def example_cross_validated_selection():
    """Cross-validated feature selection with ensemble."""
    print("\n=== Cross-Validated Selection ===")

    resolver = FeatureGroupResolver.from_yaml("src/configs/features/feature_groups.yaml")
    candidates = resolver.resolve(["imbalance", "fair_value", "spread"])

    # Cross-validated selection
    config = FeatureSelectorConfig(
        top_k=8,
        use_cross_validation=True,
        n_cv_splits=5,
        cv_test_size=1000,
        cv_gap=100,
        ensemble_method="rank_average",  # or "majority", "weighted"
    )
    selector = FeatureSelector(config)

    # Load full dataset
    # df = pd.read_pickle("path/to/data.pkl")

    # result = selector.select(candidates, df=df)
    # FeatureSelector.write_selected_yaml(result, "configs/features/cv_selected.yaml")


def example_multi_horizon_selection():
    """Multi-horizon feature selection for composite scoring."""
    print("\n=== Multi-Horizon Selection ===")

    resolver = FeatureGroupResolver.from_yaml("src/configs/features/feature_groups.yaml")
    candidates = resolver.resolve(["imbalance", "fair_value", "flow"])

    # Multi-horizon selection
    config = FeatureSelectorConfig(
        top_k=8,
        use_multi_horizon=True,
        ic_decay_horizons=[1, 5, 10, 20],  # Evaluate at multiple horizons
        horizon_weights=[0.4, 0.3, 0.2, 0.1],  # Weight short-term more
        horizon=1,  # Default horizon for fallback
    )
    selector = FeatureSelector(config)

    # Load train/validation splits
    # train_df = pd.read_pickle("path/to/train.pkl")
    # val_df = pd.read_pickle("path/to/val.pkl")

    # result = selector.select(candidates, train_df, val_df)
    # FeatureSelector.write_selected_yaml(result, "configs/features/multi_horizon_selected.yaml")


def example_hyperparameter_search():
    """Automated hyperparameter search."""
    print("\n=== Hyperparameter Search ===")

    resolver = FeatureGroupResolver.from_yaml("src/configs/features/feature_groups.yaml")
    candidates = resolver.resolve(["imbalance", "fair_value", "spread"])

    # Hyperparameter search
    config = FeatureSelectorConfig(
        enable_hyperparameter_search=True,
        hyperparameter_grid={
            "top_k": [6, 8, 10],
            "icir_threshold": [0.01, 0.02, 0.03],
        },
        use_cross_validation=True,  # Use CV for robust evaluation
        n_cv_splits=3,
        cv_test_size=800,
    )
    selector = FeatureSelector(config)

    # Load full dataset
    # df = pd.read_pickle("path/to/data.pkl")

    # result = selector.select(candidates, df=df)
    # FeatureSelector.write_selected_yaml(result, "configs/features/hyperparameter_optimized.yaml")


def example_combined_approach():
    """Combine CV and multi-horizon for robust selection."""
    print("\n=== Combined CV + Multi-Horizon ===")

    resolver = FeatureGroupResolver.from_yaml("src/configs/features/feature_groups.yaml")
    candidates = resolver.resolve(resolver.list_groups())  # All groups

    # Combined approach
    config = FeatureSelectorConfig(
        top_k=12,
        use_cross_validation=True,
        use_multi_horizon=True,
        n_cv_splits=5,
        cv_test_size=1000,
        cv_gap=100,
        ic_decay_horizons=[1, 5, 10, 20],
        horizon_weights=[0.4, 0.3, 0.2, 0.1],
        ensemble_method="weighted",  # Weight by ICIR scores
    )
    selector = FeatureSelector(config)

    # Load full dataset
    # df = pd.read_pickle("path/to/data.pkl")

    # result = selector.select(candidates, df=df)
    # FeatureSelector.write_selected_yaml(result, "configs/features/final_selected.yaml")


if __name__ == "__main__":
    print("Enhanced Feature Selection Demo")
    print("=" * 50)

    # Uncomment the examples you want to run
    # example_standard_selection()
    # example_cross_validated_selection()
    # example_multi_horizon_selection()
    # example_hyperparameter_search()
    # example_combined_approach()

    print("\n" + "=" * 50)
    print("Examples loaded. Uncomment desired examples to run.")
    print("\nKey Features:")
    print("- Time-series cross-validation for robust ranking")
    print("- Multi-horizon composite scoring (weighted ICIR across horizons)")
    print("- Automated hyperparameter search (ICIR threshold, top_k, horizon)")
    print("- Ensemble selection (majority, rank_average, weighted)")
    print("- Proper temporal split handling with gaps")
