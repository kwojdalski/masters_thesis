#!/usr/bin/env python3
import logging
import os
import sys
from pathlib import Path


def setup_logging():
    # Create logs directory if it doesn't exist
    log_dir = Path.cwd().parent.parent / "logs"
    log_dir.mkdir(exist_ok=True)

    # Configure logging
    log_file = log_dir / "render.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)],
    )


def main():
    try:
        setup_logging()
        logging.info("Starting post-render processing")

        # Get current working directory
        current_dir = Path.cwd()
        logging.info(f"Current working directory: {current_dir}")

        # Get output files from environment
        output_files = os.environ.get("QUARTO_PROJECT_OUTPUT_FILES", "")
        if output_files:
            logging.info("Output files generated:")
            for file in output_files.split():
                logging.info(f"  - {file}")
        else:
            logging.warning("No output files found in environment")

        # Log build directory information
        build_dir = current_dir.parent.parent / "build"
        logging.info(f"Build directory: {build_dir}")

        # Log completion
        logging.info("Post-render processing completed successfully")

    except Exception as e:
        logging.error(f"Error during post-render processing: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
