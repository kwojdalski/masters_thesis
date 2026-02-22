## 4.x TD3-Based Training Pipeline for Continuous-Action Trading

This subchapter presents the training workflow used for experiments based on the Twin Delayed Deep Deterministic Policy Gradient (TD3) algorithm in a reinforcement learning framework for algorithmic trading. The focus is on a continuous-action setting in which the agent outputs portfolio allocation weights bounded in the interval `[-1, 1]`, where negative values represent short exposure, positive values represent long exposure, and magnitudes represent position intensity. The aim of this section is to describe the methodological pipeline used to conduct controlled, reproducible experiments suitable for academic evaluation, rather than to document low-level software implementation.

The description assumes a single-asset trading setting with market data represented as OHLCV time series and optional engineered features. However, the same workflow generalizes to multi-asset settings with vector-valued allocations.

### 4.x.1 Rationale for TD3 in Continuous Portfolio Allocation

TD3 is a suitable choice for portfolio-allocation problems in which the action space is continuous rather than categorical. In many trading formulations, a decision is not limited to discrete actions such as buy, hold, or sell; instead, the agent must determine a target exposure level. This formulation is particularly relevant when leverage, short-selling, or partial allocations are allowed. In such cases, an action of `0.25` may represent a moderate long position, while `-0.80` may represent a strong short allocation.

Compared with discrete-action methods, continuous-action algorithms provide finer control over risk exposure and can model more realistic portfolio management behavior. TD3 is especially attractive because it addresses well-known instability issues of deterministic actor-critic methods through three mechanisms: twin critics (to reduce overestimation bias), delayed policy updates (to stabilize actor learning), and target policy smoothing (to regularize target value estimation under noisy actions). These properties are useful in financial environments, where rewards are noisy, transitions are non-stationary, and overestimation can lead to unstable policies.

### 4.x.2 Configuration-Driven Experiment Design

The training pipeline is organized as a configuration-driven workflow. Experimental scenarios are defined externally (for example, in YAML files with override support), allowing the researcher to vary data sources, feature sets, reward functions, environment settings, and optimization hyperparameters without modifying source code. This design supports repeatability and reduces the risk of undocumented manual changes between runs.

A typical experiment configuration specifies the dataset path, data split sizes, feature engineering configuration, environment settings (including transaction costs and reward type), model architecture, and TD3 training hyperparameters. It also includes logging and checkpointing settings to support experiment tracking and recovery. This approach is valuable in thesis research because it creates a transparent mapping between reported results and the conditions under which they were produced.

An additional benefit of configuration-driven experiments is compatibility with systematic validation prior to training. A validation stage can verify that referenced files exist, feature definitions are valid, required columns are available, and data splits are feasible before computationally expensive training begins.

### 4.x.3 Data Preparation and Leakage-Controlled Splitting

A critical aspect of the workflow is the separation of data preparation into a sequence that prevents information leakage. The dataset is first loaded and cleaned (for example, removal of invalid or missing rows). It is then split into three chronological subsets: training, validation, and test. Chronological splitting is essential in financial time series because shuffling would destroy temporal structure and create unrealistic information flow.

The three subsets serve different purposes. The training set is used to fit the policy and value functions. The validation set is used for model selection, intermediate evaluation, and diagnostic analysis during development. The test set is reserved for final out-of-sample assessment and should not influence hyperparameter tuning. This separation is central to maintaining methodological rigor and avoiding optimistic performance estimates.

Feature engineering is performed only after the split. More precisely, any feature transformation that requires fitted parameters (such as normalization or scaling statistics) is fit on the training subset only and then applied to validation and test subsets using the learned parameters. This ensures that future information from validation or test periods does not influence the representation learned from training data. In a quantitative finance context, violating this rule can materially inflate apparent performance, especially when normalization compresses future volatility regimes into the training representation.

Because the feature pipeline is defined externally, the workflow supports rapid experimentation with technical indicators, lagged returns, volatility estimators, and momentum measures while preserving the same leakage-control logic.

### 4.x.4 Environment Construction and Action Semantics

After data preparation, the environment is created from the processed training dataset. The environment exposes observations composed of raw market fields (for example, prices) and engineered features, and it defines a continuous action space corresponding to portfolio weights. The environment also encodes market frictions such as transaction fees and, when applicable, financing or borrowing costs.

In this formulation, the semantics of the action are not "trade direction only," but target exposure. This distinction is important for interpreting model behavior and evaluation plots. A continuous trajectory of actions across time is expected and reflects dynamic risk allocation rather than discrete state switching. Consequently, action visualizations typically display smooth or noisy values in a continuous interval rather than only a small set of fixed labels.

Reward design is another key component of environment construction. Depending on the research question, the reward may be defined as log return, differential Sharpe-like objectives, or other risk-adjusted formulations. The pipeline can support multiple reward types, but the evaluation stage must remain consistent with the chosen reward semantics. In particular, if the reward is not an exact per-step log return, it should not be directly interpreted as such in downstream performance calculations.

### 4.x.5 TD3 Training Loop and Optimization Workflow

Once the environment and models are initialized, training proceeds through iterative interaction and optimization. At each step, the agent observes the current state and proposes a continuous action. During training, exploration noise is added to the action to encourage state-space coverage and prevent premature convergence to a narrow policy. The environment returns the next state, reward, and termination signal, and the transition is stored in a replay buffer.

The replay buffer is central to off-policy learning and improves sample efficiency by enabling repeated reuse of transitions. Mini-batches sampled from this buffer are used to update the critics and actor. TD3 employs two critics and uses the minimum of their target estimates when constructing the Bellman target, reducing overestimation bias. In addition, target policy smoothing injects noise into the target action when computing target Q-values, which discourages the critics from exploiting sharp value peaks caused by function approximation artifacts.

A defining feature of TD3 is delayed policy updates: the critics are updated more frequently than the actor. This allows the value estimates to stabilize before they are used to update the policy. Target networks for both actor and critics are updated gradually, typically through soft updates, further improving training stability.

The following pseudo-code summarizes the workflow at a methodological level:

```text
Algorithm: TD3 Trading Experiment Workflow

Input: experiment configuration (data, features, env, TD3 hyperparameters, logging)
Output: trained policy, checkpoints, evaluation metrics, plots

1. Load experiment configuration and apply optional overrides
2. Validate configuration, paths, feature definitions, and split feasibility
3. Load raw market dataset and sort chronologically
4. Split data into train / validation / test subsets
5. Fit feature transformations on training subset only
6. Transform train, validation, and test subsets using fitted feature pipeline
7. Build training environment from processed training data
8. Initialize TD3 actor, twin critics, target networks, replay buffer, and optimizers
9. For each training step until max_steps:
   a. Select action from actor + exploration noise
   b. Step environment and collect transition
   c. Store transition in replay buffer
   d. Sample mini-batch and update twin critics
   e. Periodically update actor (delayed policy update)
   f. Soft-update target networks
   g. Periodically log metrics and save checkpoints
10. Save final checkpoint (and interrupt checkpoint if training stops early)
11. Build evaluation environment on validation or test data
12. Run deterministic policy evaluation and baseline comparisons
13. Compute quantitative finance performance metrics and export results
```

### 4.x.6 Checkpointing, Resumption, and Reproducibility

Reproducibility is a fundamental requirement in a master's thesis. The pipeline therefore includes explicit controls for random seeding, configuration logging, and checkpoint persistence. Random seeds should be applied consistently across numerical libraries and the RL framework so that repeated runs under identical settings produce comparable trajectories, subject to stochastic effects introduced by the environment or hardware.

Checkpointing serves two roles. First, it protects long-running experiments from interruption by periodically saving the training state. Second, it supports continuation experiments in which training resumes for additional steps. This is especially useful when exploratory runs reveal that longer training horizons are needed. Resumption should preserve not only model parameters but also optimizer states, counters, and (where relevant) experiment-tracking metadata, so that learning dynamics remain continuous rather than effectively restarting.

Experiment tracking with a system such as MLflow contributes to scientific traceability by storing parameters, metrics, artifacts, and run metadata. In a thesis setting, this enables auditable comparisons across hyperparameter settings, feature sets, and reward functions. It also facilitates retrospective analysis when results are unexpected or unstable.

### 4.x.7 Evaluation Protocol and Quantitative Performance Assessment

The evaluation stage should be conceptually separated from training. In a rigorous workflow, the final reported performance should be obtained on a dedicated out-of-sample dataset (validation for model selection, test for final reporting), using an evaluation environment built from that dataset. This distinction is critical because evaluating on the same environment used for training can produce overly optimistic conclusions.

Evaluation typically includes both behavioral and financial outputs. Behavioral outputs include action trajectories (portfolio allocation weights over time), position-change frequency, and turnover-related statistics. These are important for understanding whether a strategy is practically plausible, especially under transaction costs.

Financial outputs include cumulative reward curves and benchmark comparisons, such as buy-and-hold performance. However, benchmark alignment must be performed on the same price series as the evaluated rollout to avoid misleading visual comparisons.

To support quantitative finance interpretation, the pipeline can compute a set of performance metrics including annualized return, annualized volatility, Sharpe ratio, Sortino ratio, maximum drawdown, win rate, profit factor, and downside risk measures such as Value at Risk (VaR) and Conditional Value at Risk (CVaR). These metrics provide complementary views of performance and are particularly valuable because raw reward alone may not capture risk concentration, tail losses, or trading aggressiveness.

### 4.x.8 Methodological Risks and Common Pipeline Pitfalls

Even a well-structured RL trading workflow is vulnerable to methodological errors. A primary risk is train/evaluation leakage, which may occur if evaluation rollouts are inadvertently executed on the training environment while benchmark or plotting data are taken from validation or test data. This can create internally inconsistent results that are difficult to diagnose from summary metrics alone.

A second risk concerns reward interpretation. If the training reward is a shaped or risk-adjusted signal rather than direct financial return, using it as though it were per-period return in evaluation metrics can invalidate reported Sharpe, drawdown, or CAGR estimates. The evaluation stage must distinguish between optimization reward and realized portfolio return.

A third risk is overfitting through repeated validation exposure. Frequent inspection of validation plots and metrics can gradually turn the validation set into a tuning target. In a thesis context, this should be mitigated by preserving a final untouched test set and clearly documenting when and how validation feedback influenced design choices.

Finally, operational issues such as missing checkpoints, inconsistent configuration paths, or unvalidated feature definitions can compromise reproducibility. A pre-training validation stage and standardized logging materially reduce these risks.

### 4.x.9 Summary

The TD3 training pipeline described in this subchapter provides a structured methodology for continuous-action trading experiments in a quantitative finance research setting. Its key strengths are configuration-driven experiment specification, leakage-aware data preparation, continuous-action environment modeling, stable off-policy learning through TD3 mechanisms, and reproducible experiment management via seeding, checkpointing, and tracking. Equally important, the pipeline highlights the need for strict separation between training and evaluation and for consistent interpretation of reward and return. These design choices are essential for producing results that are not only computationally reproducible but also methodologically defensible in an academic thesis.
