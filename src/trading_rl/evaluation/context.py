"""Evaluation context utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from trading_rl.constants import SplitName


@dataclass(frozen=True)
class EvaluationContext:
    """Couples evaluation dataframe with the environment built from it."""

    split: SplitName | str
    df: pd.DataFrame
    env: Any
    max_steps: int
