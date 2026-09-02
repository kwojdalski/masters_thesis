from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "thesis" / "qmd" / "src"))

from thesis_mlflow_results import format_statistical_significance_summary


def test_statistical_summary_preserves_symbols_and_flags_invalid_bootstrap() -> None:
    payload = {
        "baselines": [
            {
                "symbol": "AAPL",
                "baseline": "buy_and_hold",
                "t_test": {"p_value": 0.01234, "significant": True},
                "sortino_bootstrap": {
                    "p_value": None,
                    "significant": False,
                    "bootstrap_inference_valid": False,
                    "bootstrap_note": "250 of 2000 resamples were undefined.",
                },
            },
            {
                "symbol": "MSFT",
                "baseline": "buy_and_hold",
                "t_test": {"p_value": 0.2, "significant": False},
            },
        ]
    }

    result = format_statistical_significance_summary(payload)

    assert result["Symbol"].tolist() == ["AAPL", "AAPL", "MSFT"]
    assert result["p-value"].tolist() == ["N/A", "0.0123", "0.2000"]
    assert result["Significant (p < 0.05)"].tolist() == [
        "Not reported",
        "Yes",
        "No",
    ]
    assert result.loc[0, "Note"] == "250 of 2000 resamples were undefined."
