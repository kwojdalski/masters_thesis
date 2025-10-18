"""
Command-line interface for data generation and management tools.
"""

import argparse
import sys

from data_generator import PriceDataGenerator


def setup_data_generator_parser(subparsers):
    """Setup argument parser for data generator commands."""
    parser = subparsers.add_parser(
        "generate-data",
        help="Generate synthetic price data from existing parquet files",
        description="Generate synthetic price data by sampling or filtering existing data",
    )

    parser.add_argument(
        "--source-dir",
        type=str,
        default="data/raw/binance",
        help="Source directory containing parquet files (default: data/raw/binance)",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/raw/synthetic",
        help="Output directory for synthetic data (default: data/raw/synthetic)",
    )

    parser.add_argument(
        "--list",
        action="store_true",
        help="List available source files",
    )

    parser.add_argument(
        "--source-file",
        type=str,
        help="Source parquet file name (e.g., binance-BTCUSDT-1h.parquet)",
    )

    parser.add_argument(
        "--output-file",
        type=str,
        help="Output file name (default: source file with _synthetic suffix)",
    )

    parser.add_argument(
        "--start-date",
        type=str,
        help="Start date for filtering (format: YYYY-MM-DD)",
    )

    parser.add_argument(
        "--end-date",
        type=str,
        help="End date for filtering (format: YYYY-MM-DD)",
    )

    parser.add_argument(
        "--sample-size",
        type=int,
        help="Number of rows to sample randomly",
    )

    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy source file without modifications",
    )

    parser.set_defaults(func=handle_data_generator)


def handle_data_generator(args):
    """Handle data generator commands."""
    generator = PriceDataGenerator(
        source_dir=args.source_dir, output_dir=args.output_dir
    )

    # List available files
    if args.list:
        print("Available source files:")
        source_files = generator.list_source_files()
        if not source_files:
            print("  No parquet files found in source directory")
        else:
            for f in source_files:
                print(f"  - {f}")
        return

    # Require source file for other operations
    if not args.source_file:
        print("Error: --source-file is required (or use --list to see available files)")
        sys.exit(1)

    # Copy operation
    if args.copy:
        generator.copy_data(args.source_file, args.output_file)
        return

    # Generate synthetic data
    try:
        df = generator.generate_synthetic_sample(
            source_file=args.source_file,
            output_file=args.output_file,
            start_date=args.start_date,
            end_date=args.end_date,
            sample_size=args.sample_size,
        )
        print(f"\nSuccessfully generated synthetic data with {len(df)} rows")
    except Exception as e:
        print(f"Error generating data: {e}")
        sys.exit(1)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="CLI tools for trading data science project",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(
        title="commands", description="Available commands", dest="command"
    )

    # Setup command parsers
    setup_data_generator_parser(subparsers)

    # Parse arguments
    args = parser.parse_args()

    # Show help if no command specified
    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Execute command
    args.func(args)


if __name__ == "__main__":
    main()
