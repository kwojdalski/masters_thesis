"""Unit tests for trading_rl.evaluation.explainability.

Prior test references to "explainability" elsewhere in the suite only
stub or disable RLInterpretabilityAnalyzer for pipeline smoke tests --
none exercise the actual permutation/integrated-gradients computation.
These tests use synthetic actors with a known dominant input feature
(all other feature weights zeroed) so a broken attribution pipeline
(wrong feature shuffled, wrong observation shape fed to Captum) fails
loudly instead of silently mis-ranking features that could land in the
thesis.
"""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest
import torch
import torch.nn as nn
from tensordict import TensorDict
from tensordict.nn import TensorDictModule

from trading_rl.evaluation.explainability import (
    ActorCaptumProxy,
    RLInterpretabilityAnalyzer,
)

FEATURE_NAMES = ["f0", "f1", "f2", "f3"]


class _LinearNet(nn.Module):
    """Deterministic linear map with a known per-feature weight vector, so
    the "correct" most-important feature is known ahead of time."""

    def __init__(self, weight: list[float]) -> None:
        super().__init__()
        self.linear = nn.Linear(len(weight), 1, bias=False)
        with torch.no_grad():
            self.linear.weight.copy_(torch.tensor([weight]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def _dominant_feature_actor(
    dominant_idx: int, n_features: int = 4, weight_val: float = 5.0
):
    """A TensorDictModule actor whose output depends only on feature `dominant_idx`."""
    weight = [0.0] * n_features
    weight[dominant_idx] = weight_val
    net = _LinearNet(weight)
    return TensorDictModule(net, in_keys=["observation"], out_keys=["loc"])


def _analyzer_for(dominant_idx: int, n_features: int = 4) -> RLInterpretabilityAnalyzer:
    actor = _dominant_feature_actor(dominant_idx, n_features=n_features)
    trainer = SimpleNamespace(actor=actor)
    names = [f"f{i}" for i in range(n_features)]
    return RLInterpretabilityAnalyzer(trainer, feature_names=names)


# ---------------------------------------------------------------------------
# ActorCaptumProxy
# ---------------------------------------------------------------------------


def test_proxy_extracts_loc_when_present() -> None:
    net = _LinearNet([1.0, 2.0])
    actor = TensorDictModule(net, in_keys=["observation"], out_keys=["loc"])
    proxy = ActorCaptumProxy(actor)

    out = proxy(torch.tensor([[1.0, 1.0]]))

    assert torch.allclose(out, torch.tensor([[3.0]]))


def test_proxy_extracts_action_when_loc_absent() -> None:
    net = _LinearNet([1.0, 2.0])
    actor = TensorDictModule(net, in_keys=["observation"], out_keys=["action"])
    proxy = ActorCaptumProxy(actor)

    out = proxy(torch.tensor([[1.0, 1.0]]))

    assert torch.allclose(out, torch.tensor([[3.0]]))


def test_proxy_raises_when_neither_loc_nor_action_present() -> None:
    class _NoOutputActor(nn.Module):
        def forward(self, td: TensorDict) -> TensorDict:
            return td  # echoes input, contributes neither "loc" nor "action"

    proxy = ActorCaptumProxy(_NoOutputActor())

    with pytest.raises(ValueError, match=r"loc.*action"):
        proxy(torch.randn(3, 4))


# ---------------------------------------------------------------------------
# compute_global_importance (FeaturePermutation)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dominant_idx", [0, 1, 2, 3])
def test_compute_global_importance_ranks_dominant_feature_first(
    dominant_idx: int,
) -> None:
    torch.manual_seed(0)
    analyzer = _analyzer_for(dominant_idx)
    obs_batch = torch.randn(64, 4)

    df = analyzer.compute_global_importance(obs_batch)

    assert df.iloc[0]["feature"] == f"f{dominant_idx}"
    # every non-dominant feature has zero effect on a linear model with
    # zero weight on it, so it should be far below the dominant feature
    other_importance = df[df["feature"] != f"f{dominant_idx}"]["importance"]
    assert (other_importance < df.iloc[0]["importance"]).all()
    assert other_importance.abs().max() < 1e-6


def test_compute_global_importance_returns_sorted_descending() -> None:
    torch.manual_seed(0)
    analyzer = _analyzer_for(dominant_idx=2)
    obs_batch = torch.randn(32, 4)

    df = analyzer.compute_global_importance(obs_batch)

    assert list(df["importance"]) == sorted(df["importance"], reverse=True)
    assert set(df["feature"]) == set(FEATURE_NAMES)


# ---------------------------------------------------------------------------
# compute_global_ig (IntegratedGradients)
# ---------------------------------------------------------------------------


def test_compute_global_ig_ranks_dominant_feature_first() -> None:
    torch.manual_seed(0)
    analyzer = _analyzer_for(dominant_idx=1)
    # offset from zero baseline so IG has a non-trivial path integral
    obs_batch = torch.randn(64, 4) + 2.0

    df = analyzer.compute_global_ig(obs_batch)

    assert df.iloc[0]["feature"] == "f1"
    other_importance = df[df["feature"] != "f1"]["importance"]
    assert other_importance.abs().max() < 1e-6


# ---------------------------------------------------------------------------
# compute_local_attribution
# ---------------------------------------------------------------------------


def test_compute_local_attribution_matches_feature_names_and_dominant_feature() -> None:
    torch.manual_seed(0)
    analyzer = _analyzer_for(dominant_idx=3)
    obs_sample = torch.randn(1, 4) + 2.0

    df = analyzer.compute_local_attribution(obs_sample)

    assert list(df["feature"]) == FEATURE_NAMES
    dominant_row = df[df["feature"] == "f3"]
    other_rows = df[df["feature"] != "f3"]
    assert abs(dominant_row["attribution"].iloc[0]) > 0
    assert other_rows["attribution"].abs().max() < 1e-6


# ---------------------------------------------------------------------------
# quantify_interpretability
# ---------------------------------------------------------------------------


def test_quantify_interpretability_all_importance_in_one_feature() -> None:
    analyzer = _analyzer_for(dominant_idx=0)
    df = pd.DataFrame({"feature": FEATURE_NAMES, "importance": [100.0, 0.0, 0.0, 0.0]})

    metrics = analyzer.quantify_interpretability(df)

    assert metrics["interpretability_top_feature_count"] == 1.0
    assert metrics["interpretability_top_80_feature_ratio"] == pytest.approx(0.25)
    assert metrics["interpretability_sparsity"] == pytest.approx(0.75)


def test_quantify_interpretability_uniform_importance() -> None:
    analyzer = _analyzer_for(dominant_idx=0)
    df = pd.DataFrame(
        {"feature": FEATURE_NAMES, "importance": [25.0, 25.0, 25.0, 25.0]}
    )

    metrics = analyzer.quantify_interpretability(df)

    # uniform importance needs all 4 features to reach 80% cumulative share
    assert metrics["interpretability_top_feature_count"] == 4.0
    assert metrics["interpretability_top_80_feature_ratio"] == pytest.approx(1.0)
    assert metrics["interpretability_sparsity"] == pytest.approx(0.0)


def test_quantify_interpretability_concentrated_ranks_lower_top_feature_count_than_uniform() -> (
    None
):
    """A regression here (e.g. wrong cumulative-sum axis) would flip this
    monotonic relationship between concentration and top-feature count."""
    analyzer = _analyzer_for(dominant_idx=0)
    concentrated = pd.DataFrame(
        {"feature": FEATURE_NAMES, "importance": [100.0, 0.0, 0.0, 0.0]}
    )
    uniform = pd.DataFrame(
        {"feature": FEATURE_NAMES, "importance": [25.0, 25.0, 25.0, 25.0]}
    )

    concentrated_metrics = analyzer.quantify_interpretability(concentrated)
    uniform_metrics = analyzer.quantify_interpretability(uniform)

    assert (
        concentrated_metrics["interpretability_top_feature_count"]
        < uniform_metrics["interpretability_top_feature_count"]
    )


# ---------------------------------------------------------------------------
# plot_importance / plot_importance_merged (smoke tests)
# ---------------------------------------------------------------------------


def test_plot_importance_returns_ggplot_for_importance_and_attribution_columns() -> (
    None
):
    analyzer = _analyzer_for(dominant_idx=0)
    from plotnine import ggplot

    importance_df = pd.DataFrame(
        {"feature": FEATURE_NAMES, "importance": [4.0, 3.0, 2.0, 1.0]}
    )
    attribution_df = pd.DataFrame(
        {"feature": FEATURE_NAMES, "attribution": [4.0, 3.0, 2.0, 1.0]}
    )

    assert isinstance(analyzer.plot_importance(importance_df), ggplot)
    assert isinstance(analyzer.plot_importance(attribution_df), ggplot)


def test_plot_importance_merged_stacks_two_plots() -> None:
    analyzer = _analyzer_for(dominant_idx=0)
    df_permutation = pd.DataFrame(
        {"feature": FEATURE_NAMES, "importance": [4.0, 3.0, 2.0, 1.0]}
    )
    df_ig = pd.DataFrame({"feature": FEATURE_NAMES, "importance": [1.0, 2.0, 3.0, 4.0]})

    merged = analyzer.plot_importance_merged(df_permutation, df_ig)

    # plotnine's "/" (patchworklib-style) composition returns a stacked object,
    # not a plain ggplot -- just verify it builds without raising.
    assert merged is not None
