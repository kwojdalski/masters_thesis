"""Tests for the standalone evaluate command helpers."""

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
from rich.console import Console

from cli.commands.evaluate_command import EvaluateCommand, EvaluateParams, _json_default


def test_json_default_serializes_numpy_bool() -> None:
    payload = {"ok": np.bool_(True), "nested": [{"flag": np.bool_(False)}]}

    encoded = json.dumps(payload, default=_json_default)

    assert json.loads(encoded) == {"ok": True, "nested": [{"flag": False}]}


def _command() -> EvaluateCommand:
    return EvaluateCommand(Console(record=True))


def _df(start: float, n_rows: int = 6) -> pd.DataFrame:
    index = pd.date_range("2024-01-01 09:30:00", periods=n_rows, freq="1s")
    close = start + np.arange(n_rows, dtype=float)
    return pd.DataFrame({"close": close, "feature_signal": close / close[0]}, index=index)


def test_resolve_per_symbol_splits_keeps_each_symbol_separate(monkeypatch) -> None:
    command = _command()
    frames = {
        "AAPL_2026-03-02_raw.parquet": _df(100.0),
        "MSFT_2026-03-02_raw.parquet": _df(200.0),
    }

    def fake_prepare(path, config):
        return frames[path.name]

    monkeypatch.setattr(command, "_prepare_arbitrary_df", fake_prepare)

    splits, split_dfs = command._resolve_per_symbol_splits(
        config=SimpleNamespace(),
        params=EvaluateParams(split="all"),
        val_data_paths=list(frames),
    )

    assert splits == ["val_AAPL", "test_AAPL", "val_MSFT", "test_MSFT"]
    np.testing.assert_allclose(split_dfs["val_AAPL"]["close"], [100.0, 101.0, 102.0])
    np.testing.assert_allclose(split_dfs["test_AAPL"]["close"], [103.0, 104.0, 105.0])
    np.testing.assert_allclose(split_dfs["val_MSFT"]["close"], [200.0, 201.0, 202.0])
    np.testing.assert_allclose(split_dfs["test_MSFT"]["close"], [203.0, 204.0, 205.0])


def test_resolve_per_symbol_splits_can_select_test_only(monkeypatch) -> None:
    command = _command()
    frames = {
        "AAPL_2026-03-02_raw.parquet": _df(100.0),
        "MSFT_2026-03-02_raw.parquet": _df(200.0),
    }

    monkeypatch.setattr(
        command,
        "_prepare_arbitrary_df",
        lambda path, config: frames[path.name],
    )

    splits, split_dfs = command._resolve_per_symbol_splits(
        config=SimpleNamespace(),
        params=EvaluateParams(split="test"),
        val_data_paths=list(frames),
    )

    assert splits == ["test_AAPL", "test_MSFT"]
    assert set(split_dfs) == {"test_AAPL", "test_MSFT"}


def test_save_rollout_data_aligns_lengths_and_drops_initial_cumulative_value(tmp_path) -> None:
    command = _command()
    split_df = _df(100.0, n_rows=5)
    result = SimpleNamespace(
        last_positions=[1.0, 0.5, -1.0, 1.0],
        simple_returns=np.array([0.0, 0.01, -0.02]),
        cumulative_returns=np.array([0.0, 0.0, np.log1p(0.01), np.log1p(0.01) + np.log1p(-0.02)]),
    )

    command._save_rollout_data(result, "test_AAPL", split_df, tmp_path)

    saved = pd.read_parquet(tmp_path / "test_AAPL_rollout.parquet")
    assert list(saved.columns) == ["action", "simple_return", "cumulative_log_return"]
    assert len(saved) == 3
    np.testing.assert_allclose(saved["action"], [1.0, 0.5, -1.0])
    np.testing.assert_allclose(saved["simple_return"], [0.0, 0.01, -0.02])
    np.testing.assert_allclose(
        saved["cumulative_log_return"],
        [0.0, np.log1p(0.01), np.log1p(0.01) + np.log1p(-0.02)],
    )
