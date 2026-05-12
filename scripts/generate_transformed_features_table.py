"""Generate a sample table showing raw order book data and transformed features for the thesis."""

import pandas as pd
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def generate_transformed_features_table():
    """Generate a markdown table showing raw and transformed features."""

    # Load the prepared training data
    data_path = project_root / "data" / "prepared" / "pooled_daily_6sym_selected" / "train_prepared.parquet"
    df = pd.read_parquet(data_path)

    # Select key raw columns and a subset of feature columns for readability
    raw_cols = ['ts_event', 'bid_px_00', 'ask_px_00', 'bid_sz_00', 'ask_sz_00']
    feature_cols = [
        'feature_hft_book_pressure_l0',
        'feature_hft_order_book_imbalance_3l',
        'feature_hft_microprice',
        'feature_hft_ofi'
    ]

    # Get a sample of events that shows variety
    sample_df = df[raw_cols + feature_cols].head(12).copy()

    # Format timestamps nicely
    sample_df['Time (UTC)'] = sample_df['ts_event'].dt.strftime('%H:%M:%S.%f').str[:-3]

    # Create a more readable table with renamed columns
    table_df = pd.DataFrame()
    table_df['Time (UTC)'] = sample_df['Time (UTC)']
    table_df['Best Bid ($)'] = sample_df['bid_px_00'].round(2)
    table_df['Best Ask ($)'] = sample_df['ask_px_00'].round(2)
    table_df['Bid Size'] = sample_df['bid_sz_00'].astype(int)
    table_df['Ask Size'] = sample_df['ask_sz_00'].astype(int)

    # Add transformed features with descriptive names
    table_df['Book Pressure'] = sample_df['feature_hft_book_pressure_l0'].round(2)
    table_df['Order Imbalance'] = sample_df['feature_hft_order_book_imbalance_3l'].round(2)
    table_df['Microprice Deviation'] = sample_df['feature_hft_microprice'].round(3)
    table_df['Order Flow Imbalance'] = sample_df['feature_hft_ofi'].round(2)

    # Generate markdown table
    markdown_table = []
    markdown_table.append("| Time (UTC) | Best Bid ($) | Best Ask ($) | Bid Size | Ask Size | Book Pressure | Order Imbalance | Microprice Deviation | Order Flow Imbalance |")
    markdown_table.append("|------------|--------------|--------------|----------|----------|----------------|-----------------|----------------------|----------------------|")

    for _, row in table_df.iterrows():
        markdown_table.append(
            f"| {row['Time (UTC)']} | {row['Best Bid ($)']} | {row['Best Ask ($)']} | "
            f"{row['Bid Size']} | {row['Ask Size']} | {row['Book Pressure']} | "
            f"{row['Order Imbalance']} | {row['Microprice Deviation']} | {row['Order Flow Imbalance']} |"
        )

    return "\n".join(markdown_table), table_df

def main():
    """Main function to generate and display the table."""
    print("Generating transformed features table for thesis...")

    markdown_table, table_df = generate_transformed_features_table()

    print("\n" + "=" * 120)
    print("MARKDOWN TABLE FOR THESIS")
    print("=" * 120)
    print(markdown_table)
    print("\n" + "=" * 120)
    print("CAPTION: Sample of raw order book data and corresponding transformed microstructure features.")
    print("The transformed features are normalized and capture the relevant signals for the RL agent.")
    print("=" * 120)

    print("\n" + "=" * 120)
    print("DATA SUMMARY")
    print("=" * 120)
    print(f"Total rows in sample: {len(table_df)}")
    print(f"Feature columns shown: 4 (out of 10 total features)")
    print("\nFeature ranges in sample:")
    for col in ['Book Pressure', 'Order Imbalance', 'Microprice Deviation', 'Order Flow Imbalance']:
        print(f"  {col}: [{table_df[col].min():.2f}, {table_df[col].max():.2f}]")

    print("\n" + "=" * 120)
    print("PYTHON CODE TO REPRODUCE")
    print("=" * 120)
    print("""
```python
import pandas as pd

# Load prepared training data
df = pd.read_parquet('data/prepared/pooled_daily_6sym_selected/train_prepared.parquet')

# Select raw and feature columns
raw_cols = ['ts_event', 'bid_px_00', 'ask_px_00', 'bid_sz_00', 'ask_sz_00']
feature_cols = [
    'feature_hft_book_pressure_l0',
    'feature_hft_order_book_imbalance_3l',
    'feature_hft_microprice',
    'feature_hft_ofi'
]

# Get sample data
sample = df[raw_cols + feature_cols].head(12)
sample['Time (UTC)'] = sample['ts_event'].dt.strftime('%H:%M:%S.%f').str[:-3]

# Display key columns
display_cols = ['Time (UTC)'] + raw_cols[1:] + feature_cols
print(sample[display_cols].round(2))
```
""")

    return markdown_table

if __name__ == "__main__":
    main()