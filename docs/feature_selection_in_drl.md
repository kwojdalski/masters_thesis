# Feature Selection in DRL Trading (Literature Review Notes)

## Scope

This note summarizes how the DRL trading literature typically chooses input variables (state features), with emphasis on:

- what variables are included,
- how authors justify them,
- whether formal feature selection is used,
- and what process is most defensible for this thesis project.

In most papers, "feature selection" is really "state design" for an MDP/POMDP, not a classical supervised-learning feature-selection pipeline.

## Short Answer

Most DRL trading papers do not perform strict statistical feature selection (e.g., wrapper/filter methods with nested validation). Instead, they usually use:

- task-driven state design (what the agent must know to act),
- finance priors (OHLCV, indicators, volatility, costs),
- agent-state variables (position, cash, inventory, previous weights),
- normalization and lookback windows,
- and occasionally ablation studies over a few feature combinations.

A smaller subset uses learned/adaptive feature selection (gates, attention, autoencoders).

## How Variables Are Usually Chosen (Observed Process)

### 1. Start from the task and action space

Authors first define the problem type, which strongly determines variables:

- Single-asset trading:
  - price/returns, current position, cash/holdings
- Portfolio allocation:
  - multi-asset price tensor, portfolio weights, cash, previous weights
- Optimal execution / market making:
  - order book state, spread/imbalance, inventory, time remaining

This is usually framed as "state sufficiency" for the MDP.

### 2. Add internal agent state (not only market state)

Stronger papers include variables representing the agent's own constraints/state:

- current holdings / inventory,
- cash balance / net asset value,
- previous action or previous portfolio weights,
- remaining budget or horizon (especially execution tasks).

This is important when transaction costs or rebalancing penalties matter.

### 3. Add friction and risk-aware variables

Feature sets are often expanded to reflect real trading constraints:

- transaction costs (implicitly via previous weights/positions),
- volatility or turbulence proxies,
- regime/context indicators,
- volume/liquidity features.

### 4. Normalize and use a rolling window

The literature often treats preprocessing as part of feature design:

- normalized prices/returns instead of raw price levels,
- volume normalization,
- fixed lookback windows (time stacks),
- cross-sectional normalization for portfolio settings.

In practice, this is often more impactful than adding many indicators.

### 5. Run limited ablations (if any)

Many papers evaluate a few predefined feature sets, for example:

- close only vs OHLC,
- OHLC vs OHLCV,
- prices only vs prices + technical indicators,
- with vs without risk variables.

This is not usually exhaustive feature selection, but it is the most common validation step.

### 6. (Less common) Learn feature relevance inside the model

A smaller group of papers adds model components that implicitly or explicitly reweight/select variables:

- attention mechanisms,
- gating layers,
- temporal feature selection modules,
- autoencoder-based latent representations.

These approaches reduce manual feature engineering, but are less common in applied trading papers than domain-driven state design.

## Literature Snapshots: How Authors Chose Variables

### Moody and Saffell (2001): Direct reinforcement in trading

Key pattern:

- Strong emphasis on reward design (including risk-adjusted objectives and transaction costs).
- Feature engineering is relatively compact compared with later deep learning papers.
- Variable choice is driven more by trading objective and feedback signal than by large indicator inventories.

Interpretation:

- Early RL trading work focused on "what should be optimized" and market frictions, not feature proliferation.

### Deng et al. (2017): Deep direct reinforcement learning for signal representation

Key pattern:

- Shift toward representation learning.
- Less reliance on hand-crafted indicators than classical systems.
- Variables are still task-structured, but deep models are expected to learn a useful representation.

Interpretation:

- This is a transition point from manual feature engineering toward learned features.

### Jiang et al. (2017): Deep RL for portfolio management

Key variable-selection decisions:

- Use a fixed-window normalized price tensor (multi-asset, multi-period input).
- Include previous portfolio weights in the state.
- Asset universe is preselected (e.g., by liquidity/volume ranking).

Why this matters:

- Previous weights are included specifically because transaction costs depend on rebalancing.
- This is a good example of architecture-aware, finance-aware state design.

Selection style:

- Domain-driven, constrained, and explicitly tied to the portfolio-rebalancing objective.

### Liu et al. (2018): Practical DDPG for stock trading

Key variable-selection decisions:

- Compact state containing market prices and account state (holdings/cash/balance).
- State variables are chosen to make the MDP operational rather than indicator-rich.

Why this matters:

- Demonstrates a minimal and interpretable state design that is easy to audit.

Selection style:

- MDP sufficiency first, feature mining second.

### Liang et al. (2018): Adversarial DRL in portfolio management

Key variable-selection decisions:

- Uses rolling-window market features for portfolio decisions.
- Evaluates multiple feature combinations (ablation/comparison style).

Why this matters:

- One of the clearer examples in DRL trading where authors explicitly test different feature subsets and show performance sensitivity.

Selection style:

- Domain-driven features plus empirical feature-combination testing.

### FinRL (2020+): Framework-oriented DRL for finance

Key variable-selection decisions:

- Exposes a configurable menu of state variables instead of a single canonical state.
- Typical state includes balance, prices, holdings, and technical indicators.
- Often includes a turbulence/risk indicator.

Why this matters:

- Practical frameworks make feature selection a configuration choice, encouraging experimentation but also increasing the risk of ad hoc selection.

Selection style:

- Toolkit-style configurable state design with practitioner defaults.

### Lei et al. (2020): Feature-aware DRL with adaptive selection

Key variable-selection decisions:

- Explicitly addresses limitations of manual feature selection.
- Uses feature-aware components (e.g., gating/attention-like mechanisms) to adapt feature importance over time.

Why this matters:

- A clearer example of learned feature selection/reweighting in financial DRL.

Selection style:

- Hybrid: initial feature set from domain knowledge, relevance learned by the model.

### Majidi et al. (2022/2024): Algorithmic Trading Using Continuous Action Space DRL (TD3)

Key variable-selection decisions:

- Single-asset trading is modeled with a continuous action in `[-1, 1]` representing the position/allocation signal.
- Define `x_t` as the close-price percentage change at time `t`.
- Build the state as a fixed lookback window of close returns:
  - `s_t = [x_t, x_{t-1}, ..., x_{t-w+1}]`
- No explicit OHLCV tensor, technical-indicator set, fundamentals, sentiment, or order-book variables are included in the TD3 state definition.
- Transaction costs are handled in reward/PnL computation, not passed as a standalone observed input feature.

Why this matters:

- This is a clear, minimalist example of state construction where the model relies on a return window rather than broad handcrafted feature inventories.
- It cleanly separates action-space design (continuous position sizing) from feature design (state inputs), which is often conflated in trading papers.

Selection style:

- Domain-driven and compact: price-return-only state with fixed temporal context.

### Optimal execution literature (RL execution / microstructure)

Key variable-selection decisions:

- Variables are selected from market microstructure theory:
  - order book levels,
  - spread,
  - imbalance,
  - recent order flow,
  - inventory,
  - time remaining,
  - execution progress.

Why this matters:

- Execution papers often use a more explicit decomposition:
  - external market state
  - internal agent state

Selection style:

- Theory-driven state construction with strong task-specific constraints.

## Common Variable Categories Across DRL Trading Papers

### Market-state variables

- OHLCV / returns
- spread / bid-ask features
- realized volatility / rolling variance
- volume, liquidity proxies
- order book imbalance (execution/high-frequency tasks)
- regime indicators (trend, volatility state)

### Agent-state variables

- current position / inventory
- holdings per asset
- cash balance / net asset value
- previous action
- previous portfolio weights
- time remaining / episode progress

### Cost and risk variables (explicit or implicit)

- transaction fees / slippage parameters
- turnover proxy (via previous weights)
- turbulence / stress indicators
- drawdown-aware state variables (less common)

## What the Literature Usually Does Not Do

Many papers do not:

- use nested validation for feature selection,
- perform purged or embargoed CV for feature-combination search,
- report stability of feature importance across market regimes,
- separate feature selection effects from reward-shaping effects.

This is a major methodological weakness in parts of the DRL trading literature.

## Practical Guidance for This Thesis / Project

A defensible feature-selection process for this codebase should be:

### 1. Define the state from the MDP first

Use a clear decomposition:

- external market state,
- internal agent state,
- friction/risk context.

### 2. Start with a minimal baseline feature set

For example:

- returns / price-based core features,
- position or portfolio weight,
- cash / portfolio value (if available),
- one or two risk proxies.

This makes later gains from extra features interpretable.

### 3. Add features in groups, not one-by-one

Evaluate feature groups such as:

- price-only
- price + volume
- price + technical indicators
- price + indicators + regime/risk features
- with vs without calendar features

Grouped ablations are easier to analyze and less noisy than single-feature toggles.

### 4. Keep preprocessing leakage-safe

- fit scalers on train only,
- apply the same transforms to validation/test,
- avoid feature engineering steps that peek into future windows,
- perform all selection decisions using train/validation only.

### 5. Distinguish training reward from evaluation returns

If using shaped rewards (e.g., Differential Sharpe Ratio):

- do not use reward as a proxy for financial returns in evaluation metrics,
- compute evaluation metrics from realized portfolio value / actual returns.

This is critical because reward engineering and feature engineering interact, and mixing them can produce misleading conclusions.

### 6. Report feature sensitivity, not just the best run

At minimum, report:

- feature set definition,
- normalization method,
- lookback window,
- performance dispersion across seeds,
- robustness across splits/regimes.

## Suggested Thesis Framing (Optional)

A strong way to describe feature selection in the thesis:

"Feature selection is treated as state-space design under an MDP formulation. Candidate variables are chosen from financial domain knowledge and agent-state requirements, then validated through grouped ablation studies under leakage-safe temporal splits."

This framing is more accurate than claiming a purely statistical feature-selection pipeline if that is not what was done.

## References (Starting Set)

- Moody, J., and Saffell, M. (2001). Learning to trade via direct reinforcement.
- Deng, Y., Bao, F., Kong, Y., Ren, Z., and Dai, Q. (2017). Deep direct reinforcement learning for financial signal representation and trading.
- Jiang, Z., Xu, D., and Liang, J. (2017). A deep reinforcement learning framework for the financial portfolio management problem.
- Liu, X., et al. (2018). A practical deep reinforcement learning approach for stock trading.
- Liang, Z., Chen, H., Zhu, J., Jiang, K., Li, Y., and Zhu, Y. (2018). Adversarial deep reinforcement learning in portfolio management.
- FinRL (Liu et al., 2020+): framework papers and documentation for DRL-based portfolio trading.
- Lei, K., Zhong, B., Yang, F., Dai, J., and Yang, B. (2020). Time-driven feature-aware jointly deep reinforcement learning for financial signal representation and algorithmic trading.
- Majidi, E., et al. (2022/2024). Algorithmic trading using continuous action space deep reinforcement learning (TD3); arXiv:2210.03469; Expert Systems with Applications, 237, 121292.
- RL optimal execution literature (e.g., market-state + agent-state decomposition in execution tasks).

## Notes for Future Expansion

Useful additions later:

- a comparison table (`paper | task | state variables | selection method | ablation? | leakage controls?`)
- explicit mapping from this literature review to the feature sets in `src/configs/scenarios/*.yaml`
- a thesis-ready subsection with formal citations in the project's chosen bibliography style
