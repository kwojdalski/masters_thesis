#!/usr/bin/env bash
# Render the thesis PDF locally, commit, and push.
# Usage: ./scripts/publish_thesis.sh [git push args]
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

PDF_PATH="thesis/qmd/src/masters-thesis.pdf"
QMD_PATH="thesis/qmd/src/masters-thesis.qmd"

echo "Rendering thesis PDF..."
quarto render "$QMD_PATH" --to pdf

if git diff --quiet "$PDF_PATH"; then
    echo "PDF unchanged — nothing to commit."
else
    echo "PDF changed — committing."
    git add "$PDF_PATH"
    git commit -m "Refresh rendered thesis PDF"
fi

echo "Pushing..."
git push master master "$@"