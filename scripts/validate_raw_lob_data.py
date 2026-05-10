#!/usr/bin/env python3
"""
Validate raw LOB (Limit Order Book) data before feature engineering.

Usage:
    python scripts/validate_raw_lob_data.py <raw_file_path>

Example:
    python scripts/validate_raw_lob_data.py data/raw/stocks/AAPL_2026-02-25_2026-03-03_raw_mbp-10_us_hours.parquet
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd


def analyze_lob_data(df: pd.DataFrame) -> dict:
    """Analyze raw LOB data for common issues."""
    result = {
        "shape": df.shape,
        "columns": list(df.columns),
        "index_type": str(type(df.index).__name__),
        "dtypes": {col: str(dtype) for col, dtype in zip(df.columns, df.dtypes)},
        "issues": []
    }

    # Basic data quality
    result["null_counts"] = df.isnull().sum().to_dict()
    total_nulls = df.isnull().sum().sum()
    result["total_null_count"] = int(total_nulls)
    result["total_null_percentage"] = float(total_nulls / (df.shape[0] * df.shape[1]) * 100)

    if result["total_null_percentage"] > 0:
        result["issues"].append(f"High null rate: {result['total_null_percentage']:.2f}%")

    # Check for timestamp issues
    if not isinstance(df.index, pd.DatetimeIndex):
        result["issues"].append(f"Index is not DatetimeIndex: {result['index_type']}")
    else:
        # Check for duplicate timestamps
        duplicates = df.index.duplicated().sum()
        if duplicates > 0:
            result["issues"].append(f"Duplicate timestamps: {duplicates:,}")

        # Check for out-of-order timestamps
        if not df.index.is_monotonic_increasing:
            result["issues"].append("Timestamps not monotonically increasing")

        # Check timestamp range
        if len(df) > 0:
            time_span = df.index[-1] - df.index[0]
            result["time_span"] = str(time_span)
            result["first_timestamp"] = str(df.index[0])
            result["last_timestamp"] = str(df.index[-1])

    # Check for LOB-specific issues
    lob_columns = [col for col in df.columns if any(x in col.lower() for x in ['px', 'sz', 'ct', 'price', 'size', 'count', 'depth'])]

    for col in lob_columns:
        if col not in df.columns:
            continue

        series = df[col].dropna()

        if len(series) == 0:
            result["issues"].append(f"Column '{col}' has no valid data")
            continue

        try:
            is_numeric = np.issubdtype(series.dtype, np.number)
        except (TypeError, AttributeError):
            is_numeric = False

        if is_numeric:
            # Check for negative prices
            if 'px' in col.lower() or 'price' in col.lower():
                negative_prices = (series < 0).sum()
                if negative_prices > 0:
                    result["issues"].append(f"Negative prices in '{col}': {negative_prices:,}")

                # Check for unreasonably large prices
                if 'bid_px' in col or 'ask_px' in col:
                    if series.max() > 10000:  # Unlikely for most stocks
                        result["issues"].append(f"Suspiciously high prices in '{col}': max ${series.max():.2f}")

            # Check for negative sizes
            if 'sz' in col.lower() or 'size' in col.lower():
                negative_sizes = (series < 0).sum()
                if negative_sizes > 0:
                    result["issues"].append(f"Negative sizes in '{col}': {negative_sizes:,}")

            # Check for unreasonably large sizes
            if series.max() > 1_000_000:  # Unlikely for LOB data
                result["issues"].append(f"Suspiciously large sizes in '{col}': max {series.max():,.0f}")

            # Check for zero variance columns
            if series.std() < 1e-10:
                result["issues"].append(f"Zero variance in '{col}': all values are {series.iloc[0]}")

            # Outlier detection
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 3.0 * IQR  # More conservative for LOB data
            upper_bound = Q3 + 3.0 * IQR

            outliers = (series < lower_bound) | (series > upper_bound)
            outlier_count = outliers.sum()
            outlier_pct = float(outlier_count / len(series) * 100)

            if outlier_pct > 20.0:  # High threshold for LOB data
                result["issues"].append(f"High outlier rate in '{col}': {outlier_pct:.2f}% (>20%)")

    return result


def format_lob_analysis(result: dict, file_path: str) -> str:
    """Format LOB analysis results."""
    lines = []
    lines.append(f"{'=' * 70}")
    lines.append(f"🔍 RAW LOB DATA VALIDATION")
    lines.append(f"{'=' * 70}")
    lines.append(f"File: {Path(file_path).name}")
    lines.append(f"Shape: {result['shape'][0]:,} rows × {result['shape'][1]:,} columns")
    lines.append(f"Index: {result['index_type']}")

    if 'time_span' in result:
        lines.append(f"Time range: {result['first_timestamp']} to {result['last_timestamp']}")
        lines.append(f"Time span: {result['time_span']}")

    # Null data check
    if result['total_null_count'] > 0:
        lines.append(f"\n⚠️  NULL VALUES:")
        lines.append(f"  Total null cells: {result['total_null_count']:,}")
        lines.append(f"  Null percentage: {result['total_null_percentage']:.4f}%")

        high_null_cols = [col for col, count in result['null_counts'].items() if count > 0]
        if high_null_cols:
            lines.append(f"  Columns with nulls: {len(high_null_cols)}")
            for col in high_null_cols[:5]:  # Show first 5
                null_pct = result['null_counts'][col] / result['shape'][0] * 100
                lines.append(f"    - {col}: {null_pct:.2f}%")
            if len(high_null_cols) > 5:
                lines.append(f"    ... and {len(high_null_cols) - 5} more")
    else:
        lines.append(f"\n✅ NO NULL VALUES - All data complete")

    # Data type check
    lines.append(f"\n📊 DATA TYPES:")
    type_counts = {}
    for dtype in result['dtypes'].values():
        type_counts[dtype] = type_counts.get(dtype, 0) + 1

    for dtype, count in sorted(type_counts.items()):
        lines.append(f"  {dtype}: {count} columns")

    # Issues found
    if result['issues']:
        lines.append(f"\n🚨 ISSUES FOUND ({len(result['issues'])}):")
        for i, issue in enumerate(result['issues'], 1):
            lines.append(f"  {i}. {issue}")
    else:
        lines.append(f"\n✅ NO ISSUES FOUND - Raw LOB data looks clean")

    lines.append(f"{'=' * 70}")
    return "\n".join(lines)


def analyze_specific_columns(df: pd.DataFrame, columns: list[str]) -> None:
    """Analyze specific LOB columns in detail."""
    print(f"\n{'#' * 70}")
    print(f"📊 DETAILED COLUMN ANALYSIS")
    print(f"{'#' * 70}")

    for col in columns:
        if col not in df.columns:
            print(f"\n⚠️  Column '{col}' not found in data")
            continue

        series = df[col].dropna()
        if len(series) == 0:
            print(f"\n🚨 Column '{col}': NO VALID DATA")
            continue

        try:
            is_numeric = np.issubdtype(series.dtype, np.number)
        except (TypeError, AttributeError):
            is_numeric = False

        if is_numeric:
            print(f"\n📊 COLUMN: {col}")
            print(f"  Type: {series.dtype}")
            print(f"  Non-null: {len(series):,} / {len(df):,} ({len(series)/len(df)*100:.1f}%)")

            if len(series) > 0:
                print(f"  Min: {series.min():.4f}")
                print(f"  Max: {series.max():.4f}")
                print(f"  Mean: {series.mean():.4f}")
                print(f"  Median: {series.median():.4f}")
                print(f"  Std: {series.std():.4f}")

                # Outlier analysis
                Q1 = series.quantile(0.25)
                Q3 = series.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 3.0 * IQR
                upper_bound = Q3 + 3.0 * IQR

                outliers = (series < lower_bound) | (series > upper_bound)
                outlier_pct = float(outliers.sum() / len(series) * 100)

                print(f"  🚨 Outliers: {outliers.sum():,} ({outlier_pct:.2f}%)")
                print(f"     IQR bounds: [{lower_bound:.4f}, {upper_bound:.4f}]")

                if outlier_pct > 20.0:
                    print(f"     ⚠️  HIGH OUTLIER RATE for LOB data")
                elif outlier_pct > 10.0:
                    print(f"     ⚠️  MODERATE OUTLIER RATE")
                else:
                    print(f"     ✅ NORMAL OUTLIER RATE for LOB data")


def main():
    """Main function."""
    if len(sys.argv) < 2:
        print(__doc__)
        print("\nExample files:")
        print("  data/raw/stocks/AAPL_2026-02-25_2026-03-03_raw_mbp-10_us_hours.parquet")
        sys.exit(1)

    file_path = Path(sys.argv[1])

    if not file_path.exists():
        print(f"Error: File not found: {file_path}")
        sys.exit(1)

    print(f"Loading raw LOB data from: {file_path}")
    df = pd.read_parquet(file_path)

    # Basic validation
    analysis = analyze_lob_data(df)
    print(format_lob_analysis(analysis, str(file_path)))

    # Detailed analysis of key LOB columns
    key_columns = [
        'bid_px_00', 'ask_px_00', 'bid_sz_00', 'ask_sz_00',  # Level 0
        'bid_px_01', 'ask_px_01', 'bid_sz_01', 'ask_sz_01',  # Level 1
        'bid_px_02', 'ask_px_02', 'bid_sz_02', 'ask_sz_02',  # Level 2
        'depth', 'size', 'price'  # Order metadata
    ]

    # Filter to columns that exist in the data
    existing_columns = [col for col in key_columns if col in df.columns]

    if existing_columns:
        analyze_specific_columns(df, existing_columns)
    else:
        print(f"\n⚠️  No key LOB columns found for detailed analysis")

    print("\n✅ Raw LOB data validation complete!")


if __name__ == "__main__":
    main()
