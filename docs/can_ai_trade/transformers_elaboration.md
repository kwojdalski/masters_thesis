# How Transformers are Used in the Paper

The paper employs a Transformer architecture to process financial time-series data, replacing the standard Recurrent Neural Networks (RNNs) or Fully Connected Networks (FCNs) typically used in Deep Reinforcement Learning (DRL).

## 1. Input Representation (The "Sequence")
In Natural Language Processing (NLP), a Transformer processes a sequence of words (tokens). In this financial context, the "sequence" is a history of market data.

*   **Look-back Window:** The agent looks at a fixed window of the past **20 days** (time steps).
*   **The "Token":** Each day in this 20-day window is treated like a "word" in a sentence.
*   **Feature Vector:** For each day $t$, the input is a vector containing:
    *   **Market Data:** Returns based on closing prices.
    *   **Price Info:** OHLC (Open, High, Low, Close) data (normalized).
    *   **Technical Indicators:** RSI, MACD, ATR, Moving Averages.
    *   **Time Encoding:** Sine and Cosine of time (to capture seasonality/trends without exploding values).
    *   **Current Position:** What the agent currently holds (Long/Short/Cash).

So, the input to the Transformer is a matrix of shape `(Batch_Size, Sequence_Length=20, Feature_Dimension)`.

## 2. The Architecture (Encoder-Decoder)
The paper references the original "Attention Is All You Need" architecture (Vaswani et al., 2017), which consists of an Encoder and a Decoder.

*   **Encoder:** Processes the 20-day history.
    *   It uses **Self-Attention** to analyze relationships between different days in the window.
    *   *Example:* The model can "attend" to a specific volatility spike 15 days ago and relate it to the price action yesterday, regardless of the days in between. This is the key advantage over RNNs/LSTMs, which process data sequentially and can "forget" older events.
*   **Positional Encodings:** Since Transformers process the whole sequence in parallel, they have no inherent concept of "order." The authors add positional encodings (vectors added to the input) so the model knows that "Day $t-1$" is closer than "Day $t-20$."
*   **Decoder:** Typically used in NLP to generate text. In this DRL context, the decoder uses the encoder's output to generate a "representation" of the current state.

## 3. Decision Making (The Output)
The output of the Transformer (a high-dimensional vector representing the "context" of the market) is fed into a final set of fully connected layers (the "Head").

*   **For DDQN (Value-Based):**
    *   The head outputs **3 Q-Values**, one for each action: $Q(s, \text{Long})$, $Q(s, \text{Short})$, $Q(s, \text{Neutral})$.
    *   The agent picks the action with the highest Q-Value.
*   **For PPO (Policy-Based / Actor-Critic):**
    *   **Actor Head:** Outputs a probability distribution (Softmax) over the 3 actions (e.g., [10% Long, 80% Short, 10% Neutral]).
    *   **Critic Head:** Outputs a single scalar value $V(s)$, estimating how "good" the current market state is (expected future return).

## 4. Why Transformers? (The "Why")
The authors hypothesize that Transformers are superior because:
1.  **Long-Range Dependencies:** The **Attention Mechanism** allows the model to instantly weigh the importance of any day in the 20-day window against any other day. An LSTM might dilute the signal of a critical event 19 days ago; a Transformer sees it clearly.
2.  **Sequential Patterns:** Financial data is sequential. The Transformer is designed to understand complex sequences (like grammar in language) and can theoretically learn "market grammar" (e.g., "if RSI is low AND volume spiked 5 days ago -> likely reversal").
3.  **Parallelism:** They train faster than RNNs because they don't need to loop through time steps sequentially during training.

## Summary
In short, the authors treat the last 20 days of market data as a "sentence" where each day is a "word." They use a Transformer to "read" this sentence, understand the context via attention, and then output a trading decision (Buy/Sell/Hold) based on that understanding.
