# Summary: Can Artificial Intelligence Trade the Stock Market?

**Authors:** Jędrzej Maskiewicz, Paweł Sakowski  
**Source:** Faculty of Economic Sciences, University of Warsaw  
**Date:** Working Paper (Context year ~2024/2025 based on data usage up to 2023)

## 1. Abstract & Objective
The paper explores the application of **Deep Reinforcement Learning (DRL)** in algorithmic trading. It hypothesizes that DRL agents can autonomously identify patterns in high-dimensional financial data to outperform traditional strategies.

**Key Comparison:**
*   **Algorithms:** Double Deep Q-Network (**DDQN**) vs. Proximal Policy Optimization (**PPO**).
*   **Architectures:** Fully Connected Neural Networks (**NN**) vs. Transformer Networks (**T**).
*   **Benchmarks:** "Buy and Hold" and a theoretical "Perfect Annual Strategy" (perfect foresight of annual direction).

## 2. Methodology

### Algorithms
*   **DDQN (Value-based):** Addresses overestimation bias in Q-learning by splitting action selection and evaluation between two networks.
*   **PPO (Policy-based / Actor-Critic):** Uses a clipping mechanism to ensure stable policy updates and Generalized Advantage Estimation (GAE) to balance bias and variance.

### Architectures
*   **Fully Connected (NN):** Standard multilayer perceptrons.
*   **Transformer (T):** Utilizes self-attention mechanisms to capture long-range dependencies in sequential time-series data.

### Experimental Setup
*   **Period:** 2019–2023 (Test period).
*   **Training:** Walk-forward (rolling window) optimization.
*   **Assets:** 
    *   **FOREX:** EUR/USD, EUR/JPY, USD/JPY
    *   **Equity:** S&P 500
    *   **Crypto:** Bitcoin (BTC)
*   **Inputs:** Returns, OHLC data, technical indicators (RSI, MACD, ATR), and time encodings.
*   **Reward Function:** Scaled Profit and Loss (PnL).

## 3. Key Results

The **PPO with Transformer architecture (PPO_T)** emerged as the consistently dominant strategy across most asset classes.

### Performance by Asset
*   **EUR/USD:** 
    *   **Winner:** `DDQN_NN` (Sharpe: 1.842).
    *   *Note:* Agents stayed out of the market significantly (56% for DDQN_NN), acting conservatively.
*   **EUR/JPY:** 
    *   **Winner:** `PPO_T` (CAGR: 12.62%, Sharpe: 1.752).
    *   Outperformed all benchmarks.
*   **USD/JPY:** 
    *   **Winner:** `PPO_T` (CAGR: 17.3%, Sharpe: 2.179).
    *   Exceptional performance in 2022-2023 trends.
*   **S&P 500:** 
    *   **Winner:** `PPO_T` (CAGR: 49.76%, Sharpe: 2.158).
    *   `DDQN_T` was notable for profiting enormously during the COVID-19 crash (early 2020).
*   **Bitcoin:** 
    *   **Winner:** `PPO_T` (CAGR: 69.65%).
    *   *Challenge:* Only `PPO_T` beat the Buy & Hold benchmark. Bitcoin proved difficult due to high volatility and a shorter history for training.

## 4. Discussion & Insights

### Strategic Non-Engagement
A critical advantage of DRL over supervised learning is the ability to **stay out of the market**.
*   Agents often chose to remain inactive (approx. 30% of the time, varying by strategy).
*   This "risk control" allowed agents to avoid unfavorable volatility, effectively beating the market by knowing when *not* to trade.

### Architecture Matters
*   **Transformers** significantly outperformed standard NNs, proving that capturing sequential dependencies is vital for financial time series.
*   PPO on standard NNs often underperformed, highlighting that a strong algorithm requires a capable architecture.

### Challenges
*   **Data Sufficiency:** Bitcoin trading suffered from a shorter training history compared to traditional assets.
*   **Gamma Parameter:** The discount factor ($\gamma=0.75$) might have favored shorter-term gains over long-term holding.
*   **Market Impact:** The study assumes agent actions do not impact market prices (valid for small volumes).

## 5. Conclusion
The study concludes that DRL is a viable tool for financial trading, capable of generating superior risk-adjusted returns. The **PPO algorithm combined with Transformer networks** is identified as the most robust approach, excelling in adapting to dynamic market conditions and managing risk through strategic non-engagement.
