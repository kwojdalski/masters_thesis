# Statistical Tests Architecture

This module uses the **Strategy Pattern** for extensible, type-safe statistical testing.

## Architecture Overview

```
StatisticalTest (ABC)
├── BootstrapTest (ABC) - Template for ratio tests
│   ├── SharpeBootstrapTest
│   └── SortinoBootstrapTest
├── PermutationTest (ABC) - Template for permutation tests
│   └── PermutationMeanTest
├── TTest
└── MannWhitneyTest

Registry: TEST_REGISTRY maps test names → classes
Factory: get_test(name) creates instances
```

## Benefits

- **Extensible**: Add new tests without modifying existing code
- **DRY**: Shared infrastructure for bootstrap/permutation logic
- **Type-safe**: No string matching in orchestration logic
- **Testable**: Each test is isolated and mockable
- **Clear contracts**: Abstract base classes document requirements

## Adding a New Test

### Example 1: Simple Test

```python
from trading_rl.evaluation.statistical_tests import StatisticalTest, register_test
from scipy import stats

class KolmogorovSmirnovTest(StatisticalTest):
    @property
    def name(self) -> str:
        return "ks_test"

    def run(self, strategy_returns, baseline_returns, **params):
        stat, p_value = stats.ks_2samp(strategy_returns, baseline_returns)
        return {
            "test_name": self.name,
            "ks_statistic": float(stat),
            "p_value": float(p_value),
            "significant": p_value < 0.05,
        }

# Register the test
register_test(KolmogorovSmirnovTest)
```

### Example 2: Bootstrap Test (Leverages Template)

```python
from trading_rl.evaluation.statistical_tests import BootstrapTest, register_test

class CalmarBootstrapTest(BootstrapTest):
    @property
    def name(self) -> str:
        return "calmar_bootstrap"

    def compute_metric(self, returns):
        # Compute Calmar ratio: CAGR / Max Drawdown
        equity = np.cumprod(1.0 + returns)
        cagr = equity[-1] ** (252 / len(returns)) - 1.0
        drawdowns = equity / np.maximum.accumulate(equity) - 1.0
        max_dd = abs(np.min(drawdowns))
        return cagr / max_dd if max_dd > 0 else np.nan

# Register the test
register_test(CalmarBootstrapTest)
```

### Example 3: Permutation Test (Leverages Template)

```python
from trading_rl.evaluation.statistical_tests import PermutationTest, register_test

class PermutationMedianTest(PermutationTest):
    @property
    def name(self) -> str:
        return "permutation_median"

    def compute_test_statistic(self, strategy_returns, baseline_returns):
        return np.median(strategy_returns) - np.median(baseline_returns)

# Register the test
register_test(PermutationMedianTest)
```

## Usage in Config

After registering, add to your YAML:

```yaml
statistical_testing:
  enabled: true
  tests:
    - "ks_test"
    - "calmar_bootstrap"
    - "permutation_median"
```

## Programmatic Usage

```python
from trading_rl.evaluation import get_test, list_available_tests

# List all available tests
print(list_available_tests())
# ['t_test', 'mann_whitney', 'permutation_test', 'sharpe_bootstrap', 'sortino_bootstrap', ...]

# Get a specific test
test = get_test("sharpe_bootstrap")
results = test.run(
    strategy_returns=strategy_returns,
    baseline_returns=baseline_returns,
    n_bootstrap=10000,
    confidence_level=0.95,
)
```

## Backward Compatibility

All existing function-based APIs still work:

```python
from trading_rl.evaluation import sharpe_ratio_bootstrap_test

results = sharpe_ratio_bootstrap_test(
    strategy_returns, baseline_returns, n_bootstrap=10000
)
```

## Design Rationale

### Why Strategy Pattern?

- **Before**: `if test_name == "sharpe_bootstrap": ...` (fragile, string-based)
- **After**: `test = get_test("sharpe_bootstrap"); test.run(...)` (type-safe, registry-based)

### Why Template Methods (BootstrapTest, PermutationTest)?

- **Before**: 50 lines of bootstrap code duplicated in Sharpe and Sortino
- **After**: Implement `compute_metric()`, inherit all bootstrap logic

### Why Registry?

- Adding a test requires ONE place (register_test call), not N places
- Easy to list/introspect available tests
- Supports dynamic extension at runtime
