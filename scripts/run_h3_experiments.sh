#!/usr/bin/env bash
# Thin wrapper — delegates to run_experiments.py h3.
# Kept for backward compatibility; prefer calling run_experiments.py directly.
#
# Usage:
#   bash scripts/run_h3_experiments.sh [args...]
#   uv run python scripts/run_experiments.py h3 [args...]
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec uv run python "$REPO_ROOT/scripts/run_experiments.py" h3 "$@"
