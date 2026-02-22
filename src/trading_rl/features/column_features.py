"""Generic pass-through column features."""

from __future__ import annotations

import pandas as pd

from trading_rl.features.base import Feature
from trading_rl.features.registry import register_feature


@register_feature("column_value")
class ColumnValueFeature(Feature):
    """Pass through a raw input column as a feature.

    This is useful when observations must be feature_* columns only, but the
    researcher still wants direct access to raw OHLCV values (optionally
    normalized via the standard feature pipeline fit/transform workflow).

    Params:
        column: Input dataframe column to copy (required)
    """

    def _column_name(self) -> str:
        column = self.config.params.get("column")
        if not column:
            raise ValueError(
                "column_value feature requires params.column (e.g. column: close)"
            )
        return str(column)

    def required_columns(self) -> list[str]:
        return [self._column_name()]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        column = self._column_name()
        return df[column].copy()

