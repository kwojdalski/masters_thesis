"""Quick test to verify LOB features work correctly."""

import pandas as pd
import numpy as np
from trading_rl.features import (
    BookPressureFeature,
    MicropriceFeature,
    SpreadBpsFeature,
    DepthRatioFeature,
    OrderBookImbalanceFeature,
    VWMPSkewFeature,
    BidAskSlopeFeature,
)
from trading_rl.features.base import FeatureConfig


def create_test_data(n=100):
    """Create synthetic LOB data for testing."""
    np.random.seed(42)

    data = {}
    base_price = 100.0

    # Create bid/ask prices and sizes for 10 levels
    for i in range(10):
        # Bid prices decrease as levels go deeper
        data[f"bid_px_{i:02d}"] = base_price - 0.01 * (i + 1) + np.random.randn(n) * 0.001
        # Ask prices increase as levels go deeper
        data[f"ask_px_{i:02d}"] = base_price + 0.01 * (i + 1) + np.random.randn(n) * 0.001
        # Sizes are random but positive
        data[f"bid_sz_{i:02d}"] = np.abs(np.random.randn(n) * 100 + 500)
        data[f"ask_sz_{i:02d}"] = np.abs(np.random.randn(n) * 100 + 500)

    return pd.DataFrame(data)


def main():
    print("Creating test data...")
    df = create_test_data()

    print("\nTest data shape:", df.shape)
    print("\nFirst few rows:")
    print(df.head())

    # Test BookPressureFeature
    print("\n" + "="*80)
    print("Testing BookPressureFeature (Level 0)")
    print("="*80)
    config = FeatureConfig(
        name="book_pressure_l0",
        feature_type="book_pressure",
        normalize=False,
        domain="hft",
        params={"level": 0}
    )
    feature = BookPressureFeature(config)
    result = feature.compute(df)
    print(f"Result range: [{result.min():.4f}, {result.max():.4f}]")
    print(f"Result mean: {result.mean():.4f}")
    print(f"First 5 values: {result.head().tolist()}")

    # Test MicropriceFeature
    print("\n" + "="*80)
    print("Testing MicropriceFeature")
    print("="*80)
    config = FeatureConfig(
        name="microprice",
        feature_type="microprice",
        normalize=False,
        domain="hft"
    )
    feature = MicropriceFeature(config)
    result = feature.compute(df)
    print(f"Result range: [{result.min():.4f}, {result.max():.4f}]")
    print(f"Result mean: {result.mean():.4f}")
    print(f"First 5 values: {result.head().tolist()}")

    # Compare with mid-price
    mid_price = (df["bid_px_00"] + df["ask_px_00"]) / 2
    print(f"Mid-price mean: {mid_price.mean():.4f}")
    print(f"Difference from mid: {(result - mid_price).abs().mean():.6f}")

    # Test SpreadBpsFeature
    print("\n" + "="*80)
    print("Testing SpreadBpsFeature")
    print("="*80)
    config = FeatureConfig(
        name="spread_bps",
        feature_type="spread_bps",
        normalize=False,
        domain="hft"
    )
    feature = SpreadBpsFeature(config)
    result = feature.compute(df)
    print(f"Result range: [{result.min():.4f}, {result.max():.4f}]")
    print(f"Result mean: {result.mean():.4f} bps")
    print(f"First 5 values: {result.head().tolist()}")

    # Test DepthRatioFeature
    print("\n" + "="*80)
    print("Testing DepthRatioFeature")
    print("="*80)
    config = FeatureConfig(
        name="depth_ratio",
        feature_type="depth_ratio",
        normalize=False,
        domain="hft",
        params={"levels_deep": 4}
    )
    feature = DepthRatioFeature(config)
    result = feature.compute(df)
    print(f"Result range: [{result.min():.4f}, {result.max():.4f}]")
    print(f"Result mean: {result.mean():.4f}")
    print(f"First 5 values: {result.head().tolist()}")

    # Test OrderBookImbalanceFeature
    print("\n" + "="*80)
    print("Testing OrderBookImbalanceFeature (3 levels)")
    print("="*80)
    config = FeatureConfig(
        name="order_book_imbalance",
        feature_type="order_book_imbalance",
        normalize=False,
        domain="hft",
        params={"levels": 3}
    )
    feature = OrderBookImbalanceFeature(config)
    result = feature.compute(df)
    print(f"Result range: [{result.min():.4f}, {result.max():.4f}]")
    print(f"Result mean: {result.mean():.4f}")
    print(f"First 5 values: {result.head().tolist()}")

    # Test VWMPSkewFeature
    print("\n" + "="*80)
    print("Testing VWMPSkewFeature")
    print("="*80)
    config = FeatureConfig(
        name="vwmp_skew",
        feature_type="vwmp_skew",
        normalize=False,
        domain="hft",
        params={"levels": 3}
    )
    feature = VWMPSkewFeature(config)
    result = feature.compute(df)
    print(f"Result range: [{result.min():.4f}, {result.max():.4f}]")
    print(f"Result mean: {result.mean():.4f}")
    print(f"First 5 values: {result.head().tolist()}")

    # Test BidAskSlopeFeature (Bid)
    print("\n" + "="*80)
    print("Testing BidAskSlopeFeature (Bid side)")
    print("="*80)
    config = FeatureConfig(
        name="bid_slope",
        feature_type="bid_ask_slope",
        normalize=False,
        domain="hft",
        params={"side": "bid", "levels": 5}
    )
    feature = BidAskSlopeFeature(config)
    result = feature.compute(df)
    print(f"Result range: [{result.min():.6f}, {result.max():.6f}]")
    print(f"Result mean: {result.mean():.6f}")
    print(f"First 5 values: {result.head().tolist()}")

    # Test BidAskSlopeFeature (Ask)
    print("\n" + "="*80)
    print("Testing BidAskSlopeFeature (Ask side)")
    print("="*80)
    config = FeatureConfig(
        name="ask_slope",
        feature_type="bid_ask_slope",
        normalize=False,
        domain="hft",
        params={"side": "ask", "levels": 5}
    )
    feature = BidAskSlopeFeature(config)
    result = feature.compute(df)
    print(f"Result range: [{result.min():.6f}, {result.max():.6f}]")
    print(f"Result mean: {result.mean():.6f}")
    print(f"First 5 values: {result.head().tolist()}")

    print("\n" + "="*80)
    print("All tests completed successfully!")
    print("="*80)


if __name__ == "__main__":
    main()
