# Critique: Can Artificial Intelligence Trade the Stock Market?

**Critique of Working Paper by Maskiewicz & Sakowski**

While the paper demonstrates a promising application of Deep Reinforcement Learning (DRL) with Transformer architectures in financial markets, several methodological choices and assumptions warrant critical examination. These issues potentially inflate the reported performance or limit the generalizability of the findings to real-world trading.

## 1. Metric Inflation: The Risk-Free Rate Assumption
*   **Issue:** The authors assume a risk-free rate ($R_f$) of **0%** for the entire period (2019–2023).
*   **Reality:** While rates were near zero until early 2022, the US Federal Reserve aggressively hiked rates throughout 2022 and 2023, reaching over 5%.
*   **Impact:** By keeping $R_f = 0$, the Sharpe and Sortino ratios are **significantly inflated** for the latter half of the study. A strategy earning 5% in 2023 would have a Sharpe ratio near zero (or negative) in reality, but the paper would report it as positive. This distorts the risk-adjusted performance comparison against the "Buy and Hold" benchmark, especially since the benchmark (holding the asset) bears full market risk.

## 2. Data Scarcity vs. Model Complexity
*   **Issue:** The models (especially Transformers) are trained on a rolling window of **one year of daily data** (approx. 252-365 data points).
*   **Critique:** Deep Learning models, particularly Transformers, are notoriously data-hungry. Training a Transformer on merely ~250 samples is highly prone to overfitting or failing to learn robust generalized patterns. The "patterns" detected might simply be noise or specific regime artifacts.
*   **Implication:** The success of the `PPO_T` model might be due to the "strategic non-engagement" (staying out of the market) rather than learning complex sequential alpha. The Transformer might just be better at "memorizing" that high volatility = bad, rather than predicting price action.

## 3. The "Perfect" Benchmark Fallacy
*   **Issue:** The paper compares agents against a "Perfect Annual Strategy" (perfect foresight of the year's direction).
*   **Critique:** While interesting as a theoretical ceiling, this is **not a valid financial benchmark**. It creates a false sense of "beating the market" if the agent outperforms it (which happened with PPO_T on Bitcoin). A more rigorous comparison would be against established risk-parity strategies, moving average crossovers, or standard CTAs (Commodity Trading Advisors).

## 4. Hyperparameter Selection & Gamma
*   **Issue:** The discount factor ($\gamma$) was fixed at **0.75**, and the authors admit this focuses on short-term rewards.
*   **Critique:** A $\gamma$ of 0.75 implies the agent cares very little about rewards more than a few days out. For a strategy trading on *daily* candles, this is extremely myopic. It essentially forces the agent into a mean-reversion or short-swing profile, ignoring the major trends that drive asset classes like Bitcoin or the S&P 500.
*   **Selection:** The authors admitted to relying on "expert knowledge" rather than grid search due to compute constraints. In RL, hyperparameters are critical; a suboptimal gamma could render the entire experiment inconclusive regarding the *potential* of the architecture.

## 5. Statistical Rigor & Reproducibility
*   **Issue:** RL is notoriously unstable. Results can vary wildly based on the random seed.
*   **Critique:** The paper does not clearly state if the results are the **average of multiple runs** or a **single "lucky" seed**. It mentions selecting an agent generation "halfway through training," which is an arbitrary heuristic. Without an ensemble of runs or error bars, it is impossible to know if `PPO_T` is truly superior or if it just got a lucky initialization that `DDQN` missed.

## 6. Real-World Execution Considerations
*   **Issue:** The paper uses fixed, relatively low transaction costs (0.01% - 0.1%) and **does not explicitly account for slippage or market impact**.
*   **Critique:** While such simplifications are common and often necessary in academic research to isolate and focus on algorithmic performance, it's crucial to acknowledge that in real-world trading, these factors significantly erode profits. The assumption of execution at the closing price, especially for frequent or large trades, can be overly optimistic. For practical deployment, a more detailed model of market microstructure (including variable bid-ask spreads, liquidity constraints, and potential price impact) would be essential to validate profitability.

## 7. Bitcoin "Win" Anomaly
*   **Issue:** `PPO_T` achieved a massive 69.65% CAGR on Bitcoin vs. "Buy and Hold" 60.6%.
*   **Critique:** Bitcoin is a momentum asset. A strategy that is "out of the market" ~12% of the time (as `PPO_T` was) beating "Buy and Hold" suggests it successfully dodged specific crashes. However, given the short training window and Bitcoin's regime shifts (e.g., from bull run to crypto winter), this result is highly suspicious of overfitting to the specific validation set regimes.

## Conclusion
The paper provides a strong *proof of concept* for using Transformers in RL for trading. However, the results should be taken with skepticism due to the **inflated risk-adjusted metrics (R_f=0)**, **extreme data scarcity for the chosen architecture**, and **lack of statistical variance reporting**. The "superiority" of the Transformer model may be more about its ability to quickly overfit to the immediate past regime (1-year window) than true generalized market intelligence.
