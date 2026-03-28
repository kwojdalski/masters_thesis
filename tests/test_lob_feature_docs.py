"""Ensure every registered LOB feature type is documented in docs/hft_features.md.

The test reads FeatureRegistry to find all feature types whose implementation
lives in trading_rl.features.lob_features, then checks that each type name
appears as a backtick-quoted identifier (e.g. `book_pressure`) in the docs file.
This makes the documentation a hard CI requirement.
"""

import importlib
from pathlib import Path

import pytest


DOC_PATH = Path(__file__).parent.parent / "docs" / "hft_features.md"


def _lob_feature_types() -> list[str]:
    """Return all feature type names registered from lob_features module."""
    importlib.import_module("trading_rl.features.lob_features")
    from trading_rl.features.registry import FeatureRegistry

    registry = FeatureRegistry()
    return sorted(
        key
        for key, cls in registry._registry.items()
        if "lob_features" in cls.__module__
    )


def test_docs_file_exists():
    assert DOC_PATH.exists(), (
        f"{DOC_PATH} not found. "
        "Run 'git add -f docs/hft_features.md' if the file exists but is gitignored."
    )


@pytest.mark.parametrize("feature_type", _lob_feature_types())
def test_feature_type_documented(feature_type: str):
    doc = DOC_PATH.read_text()
    assert f"`{feature_type}`" in doc, (
        f"Feature type '{feature_type}' is not documented in {DOC_PATH}. "
        f"Add a section with `{feature_type}` to the docs file."
    )
