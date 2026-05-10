#!/usr/bin/env python3
"""
Validate all raw LOB data files in a directory.

Usage:
    python scripts/validate_all_raw_lob.py <raw_data_dir>

Example:
    python scripts/validate_all_raw_lob.py data/raw/stocks/
"""

import sys
from pathlib import Path

import pandas as pd


def validate_raw_file(file_path: Path) -> dict:
    """Quick validation of a single raw LOB file."""
    try:
        df = pd.read_parquet(file_path)
    except Exception as e:
        return {"error": str(e), "file": str(file_path)}

    result = {
        "file": str(file_path.name),
        "rows": len(df),
        "columns": len(df.columns),
        "null_count": int(df.isnull().sum().sum()),
        "null_pct": float(df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100),
    }

    # Check for duplicate timestamps
    if isinstance(df.index, pd.DatetimeIndex):
        duplicates = df.index.duplicated().sum()
        result["duplicates"] = int(duplicates)
        result["dup_pct"] = float(duplicates / len(df) * 100)
        result["time_span"] = str(df.index[-1] - df.index[0])
    else:
        result["duplicates"] = None
        result["dup_pct"] = None
        result["time_span"] = None

    # Check key LOB columns
    key_columns = ['bid_px_00', 'ask_px_00', 'bid_sz_00', 'ask_sz_00']
    result["key_columns"] = [col for col in key_columns if col in df.columns]

    return result


def main():
    """Main function."""
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    data_dir = Path(sys.argv[1])

    if not data_dir.exists():
        print(f"Error: Directory not found: {data_dir}")
        sys.exit(1)

    # Find all raw LOB files
    raw_files = sorted(data_dir.glob("*_raw_mbp-10_us_hours.parquet"))

    if not raw_files:
        print("No raw LOB files found with pattern '*_raw_mbp-10_us_hours.parquet'")
        sys.exit(1)

    print(f"{'=' * 70}")
    print(f"🔍 VALIDATING {len(raw_files)} RAW LOB FILES")
    print(f"{'=' * 70}")

    results = []
    for file_path in raw_files:
        result = validate_raw_file(file_path)
        results.append(result)

    # Sort by file name
    results.sort(key=lambda x: x['file'])

    # Display results
    for i, result in enumerate(results, 1):
        if 'error' in result:
            print(f"\n❌ {i}. {result['file']}")
            print(f"   ERROR: {result['error']}")
            continue

        print(f"\n📊 {i}. {result['file']}")
        print(f"   Rows: {result['rows']:,} × Columns: {result['columns']:,}")
        print(f"   Null: {result['null_count']:,} ({result['null_pct']:.2f}%)")

        if result['duplicates'] is not None:
            print(f"   Duplicates: {result['duplicates']:,} ({result['dup_pct']:.2f}%)")
            print(f"   Time span: {result['time_span']}")

        key_cols = result['key_columns']
        if len(key_cols) == 4:
            print(f"   ✅ All key LOB columns present")
        else:
            print(f"   ⚠️  Missing LOB columns: {4 - len(key_cols)}")

    # Summary
    print(f"\n{'=' * 70}")
    print(f"📋 SUMMARY")
    print(f"{'=' * 70}")

    valid_files = [r for r in results if 'error' not in r]
    if valid_files:
        total_rows = sum(r['rows'] for r in valid_files)
        total_nulls = sum(r['null_count'] for r in valid_files)
        total_dups = sum(r['duplicates'] for r in valid_files if r['duplicates'] is not None)

        print(f"Total files: {len(results)}")
        print(f"Valid files: {len(valid_files)}")
        print(f"Total rows: {total_rows:,}")
        print(f"Total nulls: {total_nulls:,}")
        print(f"Total duplicates: {total_dups:,}")

        # Check for issues
        issues = []

        high_null_files = [r['file'] for r in valid_files if r['null_pct'] > 1.0]
        if high_null_files:
            issues.append(f"High null rates in: {len(high_null_files)} files")

        high_dup_files = [r['file'] for r in valid_files if r['dup_pct'] and r['dup_pct'] > 10.0]
        if high_dup_files:
            issues.append(f"High duplicate rates in: {len(high_dup_files)} files")

        if issues:
            print(f"\n⚠️  ISSUES:")
            for issue in issues:
                print(f"   - {issue}")
        else:
            print(f"\n✅ NO SYSTEMATIC ISSUES - Raw LOB data quality is good")

        print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
