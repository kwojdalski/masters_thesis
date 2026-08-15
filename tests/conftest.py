import os
import sys
from pathlib import Path

# Add src directory to Python path for imports
# This ensures tests work both from command line and VS Code test runner
src_path = Path(__file__).parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# Newer MLflow versions raise instead of warning when using the filesystem
# tracking backend (the project's local default, e.g. "./mlruns"). Tests
# that exercise the training/evaluation pipeline set up an MLflow run as a
# side effect and don't care which backend is used, so opt out of the
# database-migration requirement for the test suite specifically rather
# than forcing every local dev run to configure a SQL backend.
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
