# Refactor: Extract StrategyEvaluator Service

## Problem

Evaluation logic is tightly coupled across multiple layers:
- Trainers (`TD3Trainer`, `PPOTrainer`, `DDPGTrainer`) each have `evaluate()` methods
- `pipeline/evaluation.py` has complex functions (`evaluate_split`, `evaluate_all_splits`) that orchestrate rollouts, return extraction, and statistical testing
- MLflow logging (`MLflowTrainingCallback.log_*`) is called throughout evaluation
- Return extraction logic is scattered between `pipeline/evaluation.py` and `evaluation/returns.py`

**Coupling issues:**
1. `pipeline/evaluation.py` imports `MLflowTrainingCallback` (training concern)
2. `statistical_tests.py` imports `statistical_benchmarks.py` (circular imports)
3. `compute_strategy_simple_returns_for_split` has branching based on `reward_type` and `backend`
4. Evaluation creates environments internally (`AlgorithmicEnvironmentBuilder`), but trainers also create them

## Solution

Create a **pure `StrategyEvaluator` service** that:
1. Accepts a policy and price data (no training knowledge)
2. Orchestrates rollouts, return extraction, and metrics computation
3. Returns results as simple data structures (no side effects like logging)
4. Can be unit tested in isolation

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    StrategyEvaluator (new)                    │
│  - evaluate_split(split, df, policy) → EvaluationResult     │
│  - evaluate_all_splits(train_df, val_df, test_df)           │
│                                                               │
│  Trainers (update)                          TrainingPipeline (update) │
│  - evaluate() → delegate →                  - call evaluator   │
└───────────────────────────────────────────────────────────────────────┘
```

## Implementation Plan

### Phase 1: Create Core Service

**File:** `src/trading_rl/evaluation/evaluator.py`

```python
@dataclass(frozen=True)
class EvaluationConfig:
    """Simplified config for evaluation (decoupled from training config)."""
    reward_type: str
    backend: str
    max_steps: int | None = None  # Resolve from DF if None
    price_column: str = "close"
    enable_plots: bool = True
    enable_metrics: bool = True

@dataclass(frozen=True)
class SplitEvaluationResult:
    """Pure evaluation results - no MLflow artifacts."""
    final_reward: float
    last_positions: list[Any]
    simple_returns: np.ndarray
    cumulative_returns: np.ndarray | None  # For debugging
    metrics: dict[str, float] | None
    plots: dict[str, Any] | None  # Raw plot objects

class StrategyEvaluator:
    """Pure evaluation service - decoupled from training logic."""
    
    def __init__(
        self,
        env_factory: Callable[[pd.DataFrame, EvaluationConfig], Any],
        policy: Any,
        config: EvaluationConfig,
    ):
        self.env_factory = env_factory
        self.policy = policy
        self.config = config
    
    def _build_env(self, df: pd.DataFrame) -> Any:
        """Build environment from factory (or reuse if stateless)."""
        return self.env_factory(df, self.config)
    
    def _run_rollout(self, env: Any, max_steps: int) -> Any:
        """Run deterministic rollout."""
        from tensordict.nn import InteractionType
        from torchrl.envs.utils import set_exploration_type
        
        with torch.no_grad():
            try:
                with set_exploration_type(InteractionType.MODE):
                    return env.rollout(max_steps=max_steps, policy=self.policy)
            except RuntimeError:
                with set_exploration_type(InteractionType.DETERMINISTIC):
                    return env.rollout(max_steps=max_steps, policy=self.policy)
    
    def _extract_returns(self, env: Any, rollout: Any, max_steps: int) -> tuple[np.ndarray, np.ndarray | None]:
        """Extract strategy returns based on reward type and backend."""
        from trading_rl.evaluation.returns import extract_tradingenv_returns
        from trading_rl.constants import RewardType
        
        # Extract NLV-based returns for TradingEnv backend
        if self.config.backend.lower() == "tradingenv":
            cumulative = extract_tradingenv_returns(env, max_steps)
            if cumulative is not None and len(cumulative) > 1:
                step_log = np.diff(cumulative)
                simple_returns = np.exp(step_log) - 1.0
                return simple_returns, cumulative
        
        # Extract reward-stream returns for log_return
        if self.config.reward_type == RewardType.LOG_RETURN:
            rewards = rollout["next", "reward"].detach().cpu().numpy()[:max_steps]
            simple_returns = np.exp(rewards) - 1.0
            return simple_returns, None
        
        # Fallback: compute from rollout (not ideal, but preserves behavior)
        rewards = rollout["next", "reward"].detach().cpu().numpy()[:max_steps]
        simple_returns = np.exp(rewards) - 1.0
        return simple_returns, None
    
    def _compute_metrics(self, simple_returns: np.ndarray, df: pd.DataFrame) -> dict[str, float]:
        """Compute financial metrics."""
        from trading_rl.evaluation.metrics import build_metric_report
        from trading_rl.evaluation.report import periods_per_year_from_timeframe
        
        # Get price column for benchmark
        price_column = self.config.price_column
        if price_column not in df.columns and "close" in df.columns:
            price_column = "close"
        
        if price_column not in df.columns:
            return {}  # No benchmark possible
        
        price_series = df[price_column]
        
        # Determine periods per year (default to 252 if not specified)
        # TODO: Could add timeframe to EvaluationConfig
        periods_per_year = 252
        
        return build_metric_report(
            strategy_simple_returns=simple_returns,
            benchmark_simple_returns=(
                price_series.pct_change().iloc[1:].to_numpy(dtype=float)
                if len(price_series) > 1 else np.array([])
            ),
            periods_per_year=periods_per_year,
        )
    
    def evaluate_split(self, split: str, df: pd.DataFrame) -> SplitEvaluationResult:
        """Evaluate strategy on one data split."""
        from trading_rl.evaluation.plots import compare_rollouts
        
        max_steps = self.config.max_steps or len(df) - 1
        
        # Build environment
        env = self._build_env(df)
        
        # Run deterministic rollout
        rollout = self._run_rollout(env, max_steps)
        
        # Extract returns
        simple_returns, cumulative_returns = self._extract_returns(env, rollout, max_steps)
        
        # Compute metrics
        metrics = self._compute_metrics(simple_returns, df) if self.config.enable_metrics else None
        
        # Generate plots
        plots = None
        if self.config.enable_plots:
            # Reuse existing compare_rollouts for now (simplifies transition)
            # Future: refactor plots.py to be strategy-agnostic
            reward_plot, action_plot = compare_rollouts([rollout], max_steps, is_portfolio=self.config.backend.lower() == "tradingenv")
            plots = {
                "reward_plot": reward_plot,
                "action_plot": action_plot,
            }
        
        # Extract last positions
        actions = rollout.get("action", None)
        last_positions = self._extract_last_positions(actions, max_steps) if actions is not None else []
        
        return SplitEvaluationResult(
            final_reward=float(rollout["next", "reward"].sum().item()),
            last_positions=last_positions,
            simple_returns=simple_returns,
            cumulative_returns=cumulative_returns,
            metrics=metrics,
            plots=plots,
        )
    
    def _extract_last_positions(self, actions: Any, max_steps: int) -> list[Any]:
        """Extract final positions (handles discrete and continuous)."""
        import torch
        import numpy as np
        
        if actions is None:
            return []
        
        action_tensor = actions.squeeze()
        
        # Handle continuous portfolio actions
        if action_tensor.ndim > 1 and action_tensor.shape[-1] > 1:
            # Multi-asset: return mean allocation per asset
            return action_tensor.mean(dim=0).tolist()
        else:
            # Single-asset or discrete
            flat_actions = action_tensor.flatten().numpy() if hasattr(action_tensor, 'flatten') else np.array([action_tensor])
            return flat_actions[:max_steps].tolist()
    
    def evaluate_all_splits(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ) -> dict[str, SplitEvaluationResult]:
        """Evaluate strategy on all splits."""
        results = {}
        
        for split, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
            if len(df) < 2:
                continue  # Skip tiny splits
            results[split] = self.evaluate_split(split, df)
        
        return results
```

### Phase 2: Update Trainers

**Files:** `src/trading_rl/trainers/base.py`, `src/trading_rl/trainers/td3.py`, `src/trading_rl/trainers/ddpg.py`, `src/trading_rl/trainers/ppo.py`

Change `BaseTrainer.evaluate()` to delegate to `StrategyEvaluator`:

```python
# In BaseTrainer (base.py)
from trading_rl.evaluation.evaluator import StrategyEvaluator, EvaluationConfig, SplitEvaluationResult

def evaluate(
    self,
    df: pd.DataFrame,
    max_steps: int,
    config: Any = None,
    algorithm: str | None = None,
    eval_env: Any | None = None,
) -> tuple[Any, ...]:
    """Delegated evaluation - creates plots and raw results."""
    from trading_rl.evaluation.plots import (
        compare_rollouts,
        create_actual_returns_plot,
        create_merged_comparison_plot,
    )
    from trading_rl.utils import (
        compare_rollouts as utils_compare,
        create_actual_returns_plot as utils_create_plot,
        create_merged_comparison_plot as utils_merge,
    )
    
    # Build evaluation config from training config
    eval_config = EvaluationConfig(
        reward_type=getattr(config.env, "reward_type", "log_return") if config else "log_return",
        backend=getattr(config.env, "backend", "tradingenv") if config else "tradingenv",
        max_steps=max_steps,
        price_column=getattr(config.env, "price_column", None) if config else "close",
        enable_plots=True,
        enable_metrics=False,  # Metrics computed separately
    )
    
    # Use existing env factory or create env from df
    # For simplicity: delegate env creation to AlgorithmicEnvironmentBuilder
    from trading_rl.envs import AlgorithmicEnvironmentBuilder
    
    env_to_use = eval_env or self.env
    env_factory = AlgorithmicEnvironmentBuilder().create
    
    # Create evaluator
    evaluator = StrategyEvaluator(
        env_factory=env_factory,
        policy=self.actor,
        config=eval_config,
    )
    
    # Get evaluation results (simplified version for now)
    result = evaluator.evaluate_split("eval", df, max_steps)
    
    # Reconstruct tuple return for backward compatibility
    # Original: (reward_plot, action_plot, None, final_reward, last_positions, actual_returns_plot, merged_plot)
    # Updated: (reward_plot, action_plot, None, final_reward, last_positions, actual_returns_plot, merged_plot)
    
    # Note: actual_returns_plot and merged_plot need env for proper extraction
    # For now, reuse existing plotting but delegate to evaluator for return extraction
    
    # Use existing plot generation for backward compatibility
    env_to_use_local = eval_env or self.env
    
    # Original deterministic rollout
    # (kept for backward compatibility - will deprecate after transition)
    from tensordict.nn import InteractionType
    from torchrl.envs.utils import set_exploration_type
    
    with torch.no_grad():
        try:
            with set_exploration_type(InteractionType.MODE):
                rollout_deterministic = env_to_use_local.rollout(
                    max_steps=max_steps, policy=self.actor
                )
        except RuntimeError:
            with set_exploration_type(InteractionType.DETERMINISTIC):
                rollout_deterministic = env_to_use_local.rollout(
                    max_steps=max_steps, policy=self.actor
                )
    
    # Extract actual returns
    from trading_rl.evaluation.returns import extract_tradingenv_returns
    actual_returns_deterministic = extract_tradingenv_returns(env_to_use_local, max_steps)
    
    # Random rollout
    with set_exploration_type(InteractionType.RANDOM):
        rollout_random = env_to_use_local.rollout(max_steps=max_steps, policy=self.actor)
    actual_returns_random = extract_tradingenv_returns(env_to_use_local, max_steps)
    
    # Build comparison plot with pre-extracted returns
    actual_returns_plot = create_actual_returns_plot(
        [rollout_deterministic, rollout_random],
        n_obs=max_steps,
        df_prices=df,
        env=None,  # Don't pass env, use pre-extracted
        actual_returns_list=[actual_returns_deterministic, actual_returns_random],
        initial_portfolio_value=(
            float(getattr(config.env, "initial_portfolio_value", DEFAULT_INITIAL_PORTFOLIO_VALUE))
            if config
            else DEFAULT_INITIAL_PORTFOLIO_VALUE
        ),
        benchmark_price_column=getattr(config.env, "price_column", None) if config else "close",
    )
    
    # Detect backend type
    is_portfolio = self._is_portfolio_backend(config)
    
    # Original plot generation
    reward_plot, action_plot = utils_compare(
        [rollout_deterministic, rollout_random],
        max_steps,
        is_portfolio=is_portfolio,
    )
    
    merged_plot = utils_merge(reward_plot, action_plot)
    
    final_reward = float(rollout_deterministic["next", "reward"].sum().item())
    actions_tensor = rollout_deterministic.get("action", None)
    last_positions = self._extract_actions(rollout_deterministic, is_portfolio)
    
    return (
        reward_plot,
        action_plot,
        None,  # Third plot (action_probs_plot)
        final_reward,
        last_positions,
        actual_returns_plot,
        merged_plot,
    )
```

Subclasses (`TD3Trainer`, `PPOTrainer`, `DDPGTrainer`):
- Remove their `evaluate()` methods entirely (just call `super().evaluate()`)
- No changes needed - all complexity now in `BaseTrainer`

### Phase 3: Update Pipeline Evaluation

**File:** `src/trading_rl/pipeline/evaluation.py`

Simplify to use `StrategyEvaluator`:

```python
from trading_rl.evaluation.evaluator import StrategyEvaluator, EvaluationConfig

def evaluate_split(
    *,
    split: str,
    split_df: pd.DataFrame,
    trainer: Any,
    config: ExperimentConfig,
    algorithm: str,
    logs: dict[str, Any],
    logger: logging.Logger,
) -> SplitEvaluationResult | None:
    if len(split_df) < 2:
        return None
    
    logger.info("evaluate agent split=%s rows=%d", split, len(split_df))
    
    # Create evaluation config
    eval_config = EvaluationConfig(
        reward_type=config.env.reward_type,
        backend=config.env.backend,
        max_steps=config.training.resolve_eval_steps(len(split_df)),
        price_column=config.env.price_column or "close",
        enable_plots=True,
        enable_metrics=True,
    )
    
    # Create env factory
    from trading_rl.envs import AlgorithmicEnvironmentBuilder
    env_factory = AlgorithmicEnvironmentBuilder().create
    
    # Create evaluator
    evaluator = StrategyEvaluator(
        env_factory=env_factory,
        policy=trainer.actor,
        config=eval_config,
    )
    
    # Get results
    result = evaluator.evaluate_split(split, split_df)
    
    # Log to MLflow (separate concern)
    MLflowTrainingCallback.log_evaluation_plots(
        reward_plot=result.plots["reward_plot"] if result.plots else None,
        action_plot=result.plots["action_plot"] if result.plots else None,
        action_probs_plot=None,  # PPO-specific
        logs=logs,
        merged_plot=result.plots.get("merged_plot"),
        artifact_path_prefix=f"evaluation_plots/{split}",
    )
    
    # Log evaluation report (uses metrics from result)
    if result.metrics:
        MLflowTrainingCallback.log_evaluation_report(
            evaluation_report=result.metrics,
            split_prefix=split,
        )
    
    return SplitEvaluationResult(
        final_reward=result.final_reward,
        last_positions=result.last_positions,
        evaluation_report=result.metrics,
    )
```

### Phase 4: Update Statistical Tests

**File:** `src/trading_rl/evaluation/statistical_tests.py`

Make `run_all_statistical_tests` accept `StrategyEvaluator`:

```python
# Simplified signature - accept evaluator result instead of raw components
def run_all_statistical_tests(
    evaluator_result: SplitEvaluationResult,
    config: Any,
    *,
    benchmark_returns: dict[str, np.ndarray] | None = None,
    periods_per_year: int = 252,
) -> dict[str, Any]:
    """Run statistical tests on evaluator results."""
    strategy_returns = evaluator_result.simple_returns
    
    # Use existing benchmark computation functions
    # (No changes to statistical_benchmarks.py)
    
    # Run tests using existing logic
    # (Delegates to TEST_REGISTRY)
    ...
```

Then in `pipeline/evaluation.py`:

```python
# After getting evaluator result
statistical_test_results = run_all_statistical_tests(
    evaluator_result=result,
    config=config.statistical_testing,
    benchmark_returns_map=benchmark_returns_map,
    periods_per_year=periods_per_year,
)
MLflowTrainingCallback.log_statistical_tests(...)
```

### Phase 5: Tests

**File:** `tests/test_evaluator.py` (new)

```python
def test_evaluator_extract_returns_log_return():
    """Test NLV return extraction not used for log_return."""
    # Mock env and policy
    evaluator = StrategyEvaluator(...)
    result = evaluator.evaluate_split("test", mock_df)
    assert result.cumulative_returns is None  # No NLV extraction

def test_evaluator_extract_returns_tradingenv():
    """Test NLV return extraction used for TradingEnv."""
    # Mock TradingEnv with track_record
    result = evaluator.evaluate_split("test", mock_df)
    assert result.cumulative_returns is not None  # NLV extracted

def test_evaluator_metrics_computation():
    """Test metrics computed correctly."""
    # Mock price data with known returns
    evaluator = StrategyEvaluator(...)
    result = evaluator.evaluate_split("test", mock_df)
    assert result.metrics["sharpe_ratio"] == expected_value

def test_backward_compatibility_with_trainers():
    """Test that trainer.evaluate() still returns expected tuple."""
    # This ensures existing code doesn't break
```

### Phase 6: Deprecation Path

After transition, we can:
1. Mark old methods as `@deprecated`
2. Add warnings when calling deprecated paths
3. Update documentation

## File Changes Summary

| File | New | Deleted | Modified |
|---|------|---------|----------|
| `evaluation/evaluator.py` | ✓ | - | - |
| `trainers/base.py` | - | - | `evaluate()` simplified to delegate |
| `trainers/td3.py` | - | ✓ | `evaluate()` removed |
| `trainers/ppo.py` | - | ✓ | `evaluate()` removed |
| `trainers/ddpg.py` | - | ✓ | `evaluate()` removed |
| `pipeline/evaluation.py` | - | ~50% | Delegates to StrategyEvaluator |
| `evaluation/statistical_tests.py` | - | - | Simpler signature |
| `tests/test_evaluator.py` | ✓ | - | - |

## Testing Strategy

**Unit tests** (Phase 5):
- Test `StrategyEvaluator` in isolation (no trainers, no MLflow)
- Test return extraction logic for different reward types/backends
- Test metrics computation with known inputs
- Test plot generation (once decoupled)

**Integration tests** (after Phase 1-4):
- Run full training pipeline with small datasets
- Verify `evaluate()` produces same plots as before
- Verify MLflow artifacts are still logged
- Verify statistical tests produce same results

**Verification**:
- All existing tests in `test_generated_data_experiments_smoke.py` still pass
- Performance impact: minimal (one extra function call)

## Benefits

1. **Decoupling:** Trainers no longer need to know about evaluation logic
2. **Testability:** `StrategyEvaluator` can be tested in isolation
3. **Reusability:** Same evaluator can be used for inference-only scenarios
4. **Clean separation:** Evaluation config (`EvaluationConfig`) vs training config (`ExperimentConfig`)
5. **Reduced complexity:** `BaseTrainer.evaluate()` becomes a thin wrapper

## Risks

1. **Breaking change:** Subclasses that override `evaluate()` will break
   - Mitigation: Document deprecation clearly, run full test suite
   - All known subclasses: `TD3Trainer`, `PPOTrainer`, `DDPGTrainer`

2. **Backward compatibility:** Existing code expecting tuple return signature
   - Mitigation: Keep `BaseTrainer.evaluate()` signature unchanged during transition

3. **Plot generation:** Existing `compare_rollouts` may need updates
   - Mitigation: Use as-is initially, refactor in follow-up

## Rollout Plan

1. **MVP (Minimal Viable Product):**
   - Create `evaluator.py` with `StrategyEvaluator` class
   - Update `BaseTrainer.evaluate()` to delegate
   - Remove subclass `evaluate()` methods
   - Add basic unit tests
   - Run existing tests to verify no regression

2. **Phase 2 (Complete transition):**
   - Update `pipeline/evaluation.py` to use `StrategyEvaluator`
   - Update `statistical_tests.py` to accept evaluator results
   - Add comprehensive unit tests
   - Update documentation

3. **Phase 3 (Optional enhancements):**
   - Refactor `evaluation/plots.py` to be strategy-agnostic
   - Extract benchmark registry to separate module
   - Create unified config for training+evaluation

## Questions

1. Should `StrategyEvaluator` handle the random rollout (currently in `BaseTrainer.evaluate()`), or should random baseline be computed separately in `run_all_statistical_tests`?
2. Should `EvaluationConfig` include plotting options, or should those be separate?
3. Do you want me to proceed with Phase 1 (MVP), or wait for your review?
