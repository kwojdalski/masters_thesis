#!/usr/bin/env python3
"""
Validate prepared trading data by analyzing variable distributions and detecting outliers.

Usage:
    python scripts/validate_prepared_data.py <prepared_data_dir>

Example:
    python scripts/validate_prepared_data.py data/prepared/pooled_aapl_msft_tsla_nvda
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd


def detect_outliers_iqr(series: pd.Series) -> dict:
    """Detect outliers using IQR method."""
    if len(series) == 0 or not np.issubdtype(series.dtype, np.number):
        return {"count": 0, "percentage": 0.0}

    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = (series < lower_bound) | (series > upper_bound)
    outlier_count = outliers.sum()

    return {
        "count": int(outlier_count),
        "percentage": float(outlier_count / len(series) * 100),
        "lower_bound": float(lower_bound),
        "upper_bound": float(upper_bound),
        "method": "IQR (1.5 * IQR)"
    }


def detect_outliers_zscore(series: pd.Series, threshold: float = 3.0) -> dict:
    """Detect outliers using z-score method."""
    if len(series) == 0 or not np.issubdtype(series.dtype, np.number):
        return {"count": 0, "percentage": 0.0}

    mean = series.mean()
    std = series.std()

    if std == 0:
        return {"count": 0, "percentage": 0.0}

    z_scores = np.abs((series - mean) / std)
    outliers = z_scores > threshold

    outlier_count = outliers.sum()

    return {
        "count": int(outlier_count),
        "percentage": float(outlier_count / len(series) * 100),
        "threshold": threshold,
        "method": f"Z-score (> {threshold})"
    }


def analyze_column(series: pd.Series, column_name: str) -> dict:
    """Analyze a single column for distribution and outliers."""
    result = {
        "column": column_name,
        "dtype": str(series.dtype),
        "non_null_count": series.notna().sum(),
        "null_count": series.isna().sum(),
        "null_percentage": float(series.isna().sum() / len(series) * 100),
    }

    # Numeric analysis
    try:
        is_numeric = np.issubdtype(series.dtype, np.number)
    except (TypeError, AttributeError):
        is_numeric = False

    if is_numeric:
        result.update({
            "count": len(series),
            "mean": float(series.mean()) if not series.isna().all() else None,
            "std": float(series.std()) if not series.isna().all() else None,
            "min": float(series.min()) if not series.isna().all() else None,
            "max": float(series.max()) if not series.isna().all() else None,
            "median": float(series.median()) if not series.isna().all() else None,
            "q25": float(series.quantile(0.25)) if not series.isna().all() else None,
            "q75": float(series.quantile(0.75)) if not series.isna().all() else None,
        })

        # Outlier detection
        result["outliers_iqr"] = detect_outliers_iqr(series.dropna())
        result["outliers_zscore"] = detect_outliers_zscore(series.dropna())

    # Categorical/text analysis
    else:
        result.update({
            "count": len(series),
            "unique_count": series.nunique(),
            "most_common": series.mode().iloc[0] if not series.isna().all() and len(series.mode()) > 0 else None,
            "unique_percentage": float(series.nunique() / len(series) * 100) if len(series) > 0 else 0.0
        })

    return result


def format_analysis_result(result: dict, max_col_width: int = 30) -> str:
    """Format analysis result for display."""
    lines = []
    col_name = result["column"]

    lines.append(f"{'=' * max_col_width}")
    lines.append(f"📊 COLUMN: {col_name}")
    lines.append(f"{'=' * max_col_width}")
    lines.append(f"Type: {result['dtype']}")
    lines.append(f"Total rows: {result['count']:,}")
    lines.append(f"Non-null: {result['non_null_count']:,} ({100 - result['null_percentage']:.2f}%)")

    if result['null_count'] > 0:
        lines.append(f"⚠️  NULL VALUES: {result['null_count']:,} ({result['null_percentage']:.2f}%)")

    # Numeric results
    if 'mean' in result:
        lines.append(f"\n📈 STATISTICS:")
        lines.append(f"  Mean:   {result['mean']:.4f}")
        lines.append(f"  Median: {result['median']:.4f}")
        lines.append(f"  Std:    {result['std']:.4f}")
        lines.append(f"  Min:    {result['min']:.4f}")
        lines.append(f"  Max:    {result['max']:.4f}")
        lines.append(f"  Q25:    {result['q25']:.4f}")
        lines.append(f"  Q75:    {result['q75']:.4f}")

        # Outlier results
        outliers_iqr = result.get('outliers_iqr', {})
        outliers_zscore = result.get('outliers_zscore', {})

        if outliers_iqr and outliers_zscore:
            lines.append(f"\n🚨 OUTLIERS:")
            lines.append(f"  IQR Method: {outliers_iqr.get('count', 0):,} ({outliers_iqr.get('percentage', 0):.2f}%)")
            if 'lower_bound' in outliers_iqr and 'upper_bound' in outliers_iqr:
                lines.append(f"    Bounds: [{outliers_iqr['lower_bound']:.4f}, {outliers_iqr['upper_bound']:.4f}]")
            lines.append(f"  Z-Score Method: {outliers_zscore.get('count', 0):,} ({outliers_zscore.get('percentage', 0):.2f}%)")
            if 'threshold' in outliers_zscore:
                lines.append(f"    Threshold: > {outliers_zscore['threshold']}")

            # Warning for high outlier percentages
            if outliers_iqr.get('percentage', 0) > 10.0 or outliers_zscore.get('percentage', 0) > 10.0:
                lines.append(f"⚠️  WARNING: High outlier percentage detected!")

    # Categorical results
    else:
        lines.append(f"\n📊 CATEGORICAL:")
        lines.append(f"  Unique values: {result['unique_count']:,} ({result['unique_percentage']:.2f}%)")
        if result['most_common'] is not None:
            lines.append(f"  Most common: {result['most_common']}")

    lines.append("")
    return "\n".join(lines)


def generate_summary_report(results: list[dict]) -> str:
    """Generate summary report across all columns."""
    lines = []
    lines.append(f"\n{'=' * 60}")
    lines.append("📋 SUMMARY REPORT")
    lines.append(f"{'=' * 60}")

    total_columns = len(results)
    numeric_columns = sum(1 for r in results if 'mean' in r)
    categorical_columns = total_columns - numeric_columns

    lines.append(f"Total columns analyzed: {total_columns}")
    lines.append(f"  Numeric: {numeric_columns}")
    lines.append(f"  Categorical: {categorical_columns}")

    # Columns with high null percentages
    high_null_cols = [r['column'] for r in results if r['null_percentage'] > 5.0]
    if high_null_cols:
        lines.append(f"\n⚠️  COLUMNS WITH HIGH NULL VALUES (>5%):")
        for col in high_null_cols:
            null_pct = next(r['null_percentage'] for r in results if r['column'] == col)
            lines.append(f"  - {col}: {null_pct:.2f}%")

    # Columns with high outlier percentages
    high_outlier_cols = [
        r['column'] for r in results
        if 'outliers_iqr' in r and r['outliers_iqr']['percentage'] > 10.0
    ]
    if high_outlier_cols:
        lines.append(f"\n🚨 COLUMNS WITH HIGH OUTLIER PERCENTAGES (>10%):")
        for col in high_outlier_cols:
            outlier_pct = next(r['outliers_iqr']['percentage'] for r in results if r['column'] == col)
            lines.append(f"  - {col}: {outlier_pct:.2f}%")

    # Potential issues
    potential_issues = []

    # Check for constant columns
    constant_cols = [
        r['column'] for r in results
        if 'std' in r and r['std'] is not None and r['std'] < 1e-10
    ]
    if constant_cols:
        potential_issues.append(f"Constant columns (zero variance): {len(constant_cols)}")

    # Check for columns with extreme ranges
    extreme_range_cols = [
        r['column'] for r in results
        if 'mean' in r and r['std'] is not None and
        abs(r['max'] - r['min']) > 1e6  # Very large range
    ]
    if extreme_range_cols:
        potential_issues.append(f"Columns with extreme ranges: {len(extreme_range_cols)}")

    if potential_issues:
        lines.append(f"\n🔍 POTENTIAL ISSUES:")
        for issue in potential_issues:
            lines.append(f"  - {issue}")
    else:
        lines.append(f"\n✅ No obvious structural issues detected")

    lines.append(f"{'=' * 60}\n")
    return "\n".join(lines)


def validate_split(df: pd.DataFrame, split_name: str, max_columns: int = 20) -> None:
    """Validate a single data split."""
    print(f"\n{'#' * 60}")
    print(f"🔍 VALIDATING {split_name.upper()} SPLIT")
    print(f"{'#' * 60}")
    print(f"Shape: {df.shape}")
    print(f"Index: {type(df.index).__name__}")
    if hasattr(df.index, 'min'):
        print(f"Time range: {df.index.min()} to {df.index.max()}")

    # Analyze first N columns
    if max_columns is None:
        columns_to_analyze = df.columns
    else:
        columns_to_analyze = df.columns[:max_columns]
        if len(df.columns) > max_columns:
            print(f"\n⚠️  Analyzing first {max_columns} of {len(df.columns)} columns")
            print(f"    Use --all-columns to analyze all columns\n")

    results = []
    for col in columns_to_analyze:
        result = analyze_column(df[col], col)
        results.append(result)
        print(format_analysis_result(result))

    # Generate summary
    print(generate_summary_report(results))

    return results


def main():
    """Main function."""
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    data_dir = Path(sys.argv[1])

    if not data_dir.exists():
        print(f"Error: Data directory not found: {data_dir}")
        sys.exit(1)

    max_columns = 20
    if "--all-columns" in sys.argv:
        max_columns = None
    elif len(sys.argv) >= 3 and sys.argv[2].isdigit():
        max_columns = int(sys.argv[2])

    # Load splits
    splits = {}
    for split_name in ["train", "val", "test"]:
        file_path = data_dir / f"{split_name}_prepared.parquet"
        if file_path.exists():
            print(f"Loading {split_name} split from: {file_path}")
            splits[split_name] = pd.read_parquet(file_path)
        else:
            print(f"⚠️  {split_name} split not found: {file_path}")

    if not splits:
        print("Error: No prepared split files found!")
        sys.exit(1)

    # Validate each split
    all_results = {}
    for split_name, df in splits.items():
        results = validate_split(df, split_name, max_columns)
        all_results[split_name] = results

    print("\n✅ Validation complete!")
    print(f"Analyzed splits: {list(splits.keys())}")


if __name__ == "__main__":
    main()
