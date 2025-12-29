"""Demo of the new feature engineering system.

This script demonstrates:
1. How to use the new FeaturePipeline
2. Proper train/test split handling
3. Configuration-based feature selection
4. Comparison with old create_features() function
"""

import pandas as pd

from trading_rl.features import FeatureConfig, FeaturePipeline

# Example: Create synthetic OHLCV data
data = {
    "open": [100.0, 101.0, 102.0, 101.5, 103.0, 104.0, 103.5, 105.0],
    "high": [101.0, 102.0, 103.0, 102.5, 104.0, 105.0, 104.5, 106.0],
    "low": [99.0, 100.0, 101.0, 100.5, 102.0, 103.0, 102.5, 104.0],
    "close": [100.5, 101.5, 102.5, 101.0, 103.5, 104.5, 103.0, 105.5],
    "volume": [1000, 1100, 900, 1200, 1300, 1000, 1100, 1400],
}
df = pd.DataFrame(data)

print("=" * 60)
print("Feature Engineering Demo")
print("=" * 60)
print(f"\nOriginal data shape: {df.shape}")
print(df.head())

# Split into train/test
train_size = 6
train_df = df[:train_size].copy()
test_df = df[train_size:].copy()

print(f"\nTrain size: {len(train_df)}, Test size: {len(test_df)}")

# ============================================================================
# Method 1: Basic usage with FeatureConfig
# ============================================================================
print("\n" + "=" * 60)
print("Method 1: Using FeatureConfig objects")
print("=" * 60)

configs = [
    FeatureConfig(name="log_return", feature_type="log_return"),
    FeatureConfig(name="high", feature_type="high"),
    FeatureConfig(name="low", feature_type="low"),
    FeatureConfig(name="log_vol", feature_type="log_volume"),
]

pipeline = FeaturePipeline(configs)
print(f"\nCreated pipeline: {pipeline}")

# Fit on training data ONLY
pipeline.fit(train_df)
print("\nFitted pipeline on training data")

# Transform both train and test
train_features = pipeline.transform(train_df)
test_features = pipeline.transform(test_df)

print(f"\nTrain features shape: {train_features.shape}")
print("Train features:")
print(train_features)

print(f"\nTest features shape: {test_features.shape}")
print("Test features:")
print(test_features)

# ============================================================================
# Method 2: From config dictionary (how it would be loaded from YAML)
# ============================================================================
print("\n" + "=" * 60)
print("Method 2: From config dictionary")
print("=" * 60)

feature_config_dict = [
    {"name": "log_return", "feature_type": "log_return"},
    {"name": "rsi_14", "feature_type": "rsi", "params": {"period": 3}},  # Short period for demo
    {"name": "high", "feature_type": "high"},
]

pipeline2 = FeaturePipeline.from_config_dict(feature_config_dict)
print(f"\nCreated pipeline from dict: {pipeline2}")

pipeline2.fit(train_df)
train_features2 = pipeline2.transform(train_df)

print("\nTrain features with RSI:")
print(train_features2)

# ============================================================================
# Method 3: Custom feature selection for experiment
# ============================================================================
print("\n" + "=" * 60)
print("Method 3: Minimal feature set (for simple experiment)")
print("=" * 60)

minimal_config = [
    {"name": "log_return", "feature_type": "log_return"},
]

pipeline3 = FeaturePipeline.from_config_dict(minimal_config)
pipeline3.fit(train_df)
train_features3 = pipeline3.transform(train_df)

print(f"\nMinimal features shape: {train_features3.shape}")
print("Minimal features:")
print(train_features3)

# ============================================================================
# Demonstration: Why proper train/test split matters
# ============================================================================
print("\n" + "=" * 60)
print("Demonstration: Train/Test Split Importance")
print("=" * 60)

# CORRECT: Fit on train, transform both
pipeline_correct = FeaturePipeline.from_config_dict(
    [{"name": "log_return", "feature_type": "log_return"}]
)
pipeline_correct.fit(train_df)
train_out = pipeline_correct.transform(train_df)
test_out = pipeline_correct.transform(test_df)

print("\nCORRECT approach:")
print(f"  1. Fit on training data (n={len(train_df)})")
print(f"  2. Transform train: mean={train_out['feature_log_return'].mean():.4f}, std={train_out['feature_log_return'].std():.4f}")
print(f"  3. Transform test: mean={test_out['feature_log_return'].mean():.4f}, std={test_out['feature_log_return'].std():.4f}")
print("  ✓ Test mean/std may differ (uses train normalization)")

# WRONG: Fit on all data (simulating old create_features behavior)
pipeline_wrong = FeaturePipeline.from_config_dict(
    [{"name": "log_return", "feature_type": "log_return"}]
)
pipeline_wrong.fit(pd.concat([train_df, test_df]))  # Data leakage!
all_out = pipeline_wrong.transform(pd.concat([train_df, test_df]))

print("\nWRONG approach (old system):")
print(f"  1. Fit on ALL data (n={len(train_df) + len(test_df)})")
print(f"  2. Mean={all_out['feature_log_return'].mean():.4f}, std={all_out['feature_log_return'].std():.4f}")
print("  ✗ Data leakage: normalization uses test data!")

# ============================================================================
# Feature Registry
# ============================================================================
print("\n" + "=" * 60)
print("Available Features")
print("=" * 60)

from trading_rl.features import FeatureRegistry

available = FeatureRegistry.list_features()
print(f"\nRegistered features ({len(available)}):")
for feature_type in sorted(available):
    print(f"  - {feature_type}")

print("\n" + "=" * 60)
print("Demo Complete!")
print("=" * 60)
print("\nKey Takeaways:")
print("1. ✓ Fit on training data ONLY")
print("2. ✓ Transform both train and test with same normalization")
print("3. ✓ Easy to configure different feature sets")
print("4. ✓ No data leakage from test set")
print("5. ✓ Extensible through feature registry")
