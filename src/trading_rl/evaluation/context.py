"""Evaluation context utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import pandas as pd


@dataclass(frozen=True)
class EvaluationContext:
    """Couples evaluation dataframe with the environment built from it."""

    split: Literal["train", "val", "test"]
    df: pd.DataFrame
    env: Any
    max_steps: int

