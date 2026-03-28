# HFT LOB Feature Formulas

Feature set used in `src/configs/scenarios/aapl_td3_hft_lob.yaml`.
Definitions in `src/configs/features/hft_lob_features.yaml`.
Implementation in `src/trading_rl/features/lob_features.py`.

Data source: Databento MBP-10 (Market by Price, 10 levels) tick snapshots,
AAPL US equity, nanosecond resolution.

**Column naming convention**

| Pattern | Meaning |
|---|---|
| `bid_px_NN` | Bid price at level NN (00 = best bid) |
| `ask_px_NN` | Ask price at level NN (00 = best ask) |
| `bid_sz_NN` | Bid size (shares) at level NN |
| `ask_sz_NN` | Ask size (shares) at level NN |
| `bid_ct_NN` | Number of resting bid orders at level NN |
| `ask_ct_NN` | Number of resting ask orders at level NN |
| `action`    | Event type: `A` add, `C` cancel, `T` trade |
| `side`      | Aggressor: `B` buyer-initiated, `A` seller-initiated |
| `size`      | Order/trade size |

All features marked `normalize: true` use z-score normalization fit on the
training split only.

---

## Existing features (baseline set)

### 1. Book Pressure at level $i$

**Feature type:** `book_pressure` | **Names:** `hft_book_pressure_l0/l1/l2`

$$\text{BookPressure}_i = \frac{V^{bid}_i - V^{ask}_i}{V^{bid}_i + V^{ask}_i}$$

Range $[-1, +1]$. Zero when total volume at level $i$ is zero.
$+1$: all volume on bid; $-1$: all volume on ask.

---

### 2. Microprice

**Feature type:** `microprice` | **Name:** `hft_microprice`

$$\text{Microprice} = \frac{V^{ask}_0 \cdot P^{bid}_0 + V^{bid}_0 \cdot P^{ask}_0}{V^{bid}_0 + V^{ask}_0}$$

Fallback to $\frac{P^{bid}_0 + P^{ask}_0}{2}$ when total size is zero.
Pulls the fair-value estimate toward the side with less liquidity — the side
more likely to move first.

---

### 3. Microprice Divergence

**Feature type:** `microprice_divergence` | **Name:** `hft_microprice_divergence`

$$\text{MicroDiv} = \text{Microprice} - \frac{P^{bid}_0 + P^{ask}_0}{2}$$

Isolates the imbalance signal from the absolute price level.

---

### 4. Spread in Basis Points

**Feature type:** `spread_bps` | **Name:** `hft_spread_bps`

$$\text{SpreadBps} = \frac{P^{ask}_0 - P^{bid}_0}{\frac{P^{bid}_0 + P^{ask}_0}{2}} \times 10{,}000$$

Units: basis points. Regime indicator for liquidity cost.

---

### 5. Order Book Imbalance (multi-level)

**Feature type:** `order_book_imbalance` | **Name:** `hft_order_book_imbalance_3l`

$$\text{OBI} = \frac{\sum_{i=0}^{N-1}(V^{bid}_i - V^{ask}_i)}{\sum_{i=0}^{N-1}(V^{bid}_i + V^{ask}_i)}$$

With $N=3$ levels. More robust than single-level book pressure.
Range $[-1, +1]$.

---

### 6. Depth Ratio

**Feature type:** `depth_ratio` | **Name:** `hft_depth_ratio`

$$\text{DepthRatio} = \frac{V^{bid}_0 + V^{ask}_0}{\sum_{i=1}^{4}(V^{bid}_i + V^{ask}_i)}$$

High ratio: thin book behind best price (fragile liquidity).
Low ratio: deep liquidity support.

---

### 7. VWMP Skew

**Feature type:** `vwmp_skew` | **Name:** `hft_vwmp_skew`

$$\text{VWMP} = \frac{\sum_{i=0}^{2}(P^{bid}_i \cdot V^{bid}_i + P^{ask}_i \cdot V^{ask}_i)}{\sum_{i=0}^{2}(V^{bid}_i + V^{ask}_i)}$$

$$\text{VWMPSkew} = \frac{\text{VWMP} - \text{MidPrice}}{P^{ask}_0 - P^{bid}_0}$$

Positive: deep liquidity priced above mid (bid-heavy book).
Negative: ask-heavy deeper in the book.

---

### 8–9. Bid / Ask Slope

**Feature type:** `bid_ask_slope` | **Names:** `hft_bid_slope`, `hft_ask_slope`

$$\text{BidSlope} = \frac{P^{bid}_0 - P^{bid}_4}{\sum_{i=0}^{4} V^{bid}_i} \qquad
\text{AskSlope} = \frac{P^{ask}_4 - P^{ask}_0}{\sum_{i=0}^{4} V^{ask}_i}$$

Price elasticity of each side across 5 levels. Higher slope = thinner book,
higher market impact per unit volume.

---

### 10. Raw Prices and Sizes (L0–L2)

**Feature type:** `column_value` | 12 features: `hft_bid/ask_px/sz_00/01/02`

Pass-through of raw LOB columns after normalization. Allows the agent to
discover its own combinations without being constrained to the engineered
features above.

---

## New alpha-oriented features

All validated against the AAPL MBP-10 dataset (2.8M rows, zero nulls).

---

### 11. Order Flow Imbalance (OFI)

**Feature type:** `ofi` | Reference: Cont, Kukanov & Stoikov (2014)

The most predictive short-term price signal from LOB data. Captures the net
directional flow from order book events between consecutive snapshots —
distinguishing aggressive submission from cancellation.

Let $\Delta V^{bid}_t = V^{bid}_{0,t} - V^{bid}_{0,t-1}$ and
$\Delta V^{ask}_t = V^{ask}_{0,t} - V^{ask}_{0,t-1}$.

$$\text{OFI}_t = \Delta V^{bid}_t \cdot \mathbf{1}\!\left[P^{bid}_{0,t} \geq P^{bid}_{0,t-1}\right]
- \Delta V^{ask}_t \cdot \mathbf{1}\!\left[P^{ask}_{0,t} \leq P^{ask}_{0,t-1}\right]$$

The indicator conditions ensure only queue-building events (not price
improvements alone) are counted on each side.
Positive: net buying pressure. Negative: net selling.
First row is 0.

**Feasibility:** fully computable from consecutive book snapshots.

---

### 12. Rolling OFI

**Feature type:** `ofi_rolling` | Param: `window` (default 50)

$$\text{RollingOFI}_{W,t} = \sum_{\tau=t-W+1}^{t} \text{OFI}_\tau$$

Captures persistent directional pressure over multiple tick horizons.
Use multiple instances with different windows (e.g. 10, 50, 200) to give the
agent short-, medium-, and longer-term flow signals simultaneously.

**Feasibility:** derived from OFI; same inputs.

---

### 13. Queue Depletion Rate

**Feature type:** `queue_depletion` | Param: `side` (`"bid"` or `"ask"`)

$$\text{Depletion}^{side}_t = \frac{\max\!\left(V^{side}_{0,t-1} - V^{side}_{0,t},\ 0\right)}{V^{side}_{0,t-1}}$$

Measures how fast the top-of-book queue is being consumed. A rapidly
depleting bid queue signals imminent downward price pressure; ask depletion
signals upward pressure. First row is 0.
Range $[0, 1]$.

**Feasibility:** fully computable from consecutive `bid_sz_00` / `ask_sz_00`.

---

### 14. Mid-Price Acceleration

**Feature type:** `mid_price_acceleration`

$$M_t = \frac{P^{bid}_{0,t} + P^{ask}_{0,t}}{2}$$

$$\text{Accel}_t = M_t - 2M_{t-1} + M_{t-2}$$

Second finite difference of mid-price. Positive: price accelerating upward.
Negative: decelerating or reversing. Helps distinguish momentum from
mean-reversion regimes at tick frequency. First two rows are 0.

**Feasibility:** fully computable from `bid_px_00`, `ask_px_00`.

---

### 15. Log Inter-Event Time

**Feature type:** `inter_event_time`

$$\text{Gap}_t = t_{\text{ns}} - t_{\text{ns},t-1} \quad \text{(nanoseconds)}$$

$$\text{LogGap}_t = \log(1 + \text{Gap}_t)$$

Time between consecutive book update events. Sparse updates (large gap)
signal low activity and lower adverse-selection risk. Dense bursts signal
informed order flow or a large order working through the book.
Uses the nanosecond DatetimeIndex directly.

**Feasibility:** fully computable. Timestamps verified as nanosecond-precision
DatetimeIndex in the dataset.

---

### 16. Spread Ratio

**Feature type:** `spread_ratio` | Param: `window` (default 100)

$$\text{SpreadRatio}_t = \frac{\text{SpreadBps}_t}{\overline{\text{SpreadBps}}_{W,t}}$$

where $\overline{\text{SpreadBps}}_{W,t}$ is the rolling mean over $W$ ticks.

Values $> 1$: spread wider than recent average — elevated toxicity / adverse
selection risk. Values $< 1$: tighter cost environment.
Defaults to 1.0 during warm-up when rolling mean is zero.

**Feasibility:** fully computable from `bid_px_00`, `ask_px_00`.

---

### 17. Book Convexity

**Feature type:** `book_convexity` | Param: `side` (`"bid"` or `"ask"`)

$$\text{Convexity}^{bid}_t = (P^{bid}_0 - P^{bid}_1) - (P^{bid}_1 - P^{bid}_2)$$

$$\text{Convexity}^{ask}_t = (P^{ask}_2 - P^{ask}_1) - (P^{ask}_1 - P^{ask}_0)$$

Curvature of the book price levels. Negative value: gaps widen deeper in the
book (standard; signals discontinuous liquidity — large orders face sudden
price jumps). Positive: unusually tightly packed levels.
Relevant for position-sizing decisions under transaction costs.

**Feasibility:** fully computable from `bid_px_00/01/02` or `ask_px_00/01/02`.

---

### 18. Order Count Imbalance

**Feature type:** `order_count_imbalance` | Param: `level` (default 0)

$$\text{OCI}_i = \frac{C^{bid}_i - C^{ask}_i}{C^{bid}_i + C^{ask}_i}$$

where $C^{bid}_i$ = `bid_ct_0i` and $C^{ask}_i$ = `ask_ct_0i` (number of
resting orders, not volume).

Complements book pressure (volume-based): a level can show high volume
imbalance from a single large institutional order while OCI reveals the
opposite from many small orders. Range $[-1, +1]$.

**Feasibility:** `bid_ct_NN` / `ask_ct_NN` columns confirmed present in data.

---

### 19. Signed Trade Flow (Cumulative Delta)

**Feature type:** `signed_trade_flow` | Param: `window` (default 100)

Requires the Databento `action` and `side` columns.

$$\text{SignedVol}_t = \begin{cases}
+\text{size}_t & \text{if } \texttt{action}_t = \texttt{T},\ \texttt{side}_t = \texttt{B} \\
-\text{size}_t & \text{if } \texttt{action}_t = \texttt{T},\ \texttt{side}_t = \texttt{A} \\
0              & \text{otherwise}
\end{cases}$$

$$\text{CumDelta}_{W,t} = \sum_{\tau=t-W+1}^{t} \text{SignedVol}_\tau$$

Positive: net buyer-initiated volume over window.
Negative: net seller-initiated volume.
Distinct from OFI (which measures order book changes, not executed trades).

**Feasibility:** trade tape present in dataset — 226,197 trade rows with
`action='T'`, `side='B'/'A'` confirmed non-null. `size` column confirmed
non-null for all trade rows.

---

## Summary table

| # | Feature type | Key inputs | Range | Normalized |
|---|---|---|---|---|
| 1–3 | `book_pressure` L0–L2 | `bid/ask_sz_00–02` | $[-1,1]$ | yes |
| 4 | `microprice` | `bid/ask_px/sz_00` | price scale | yes |
| 5 | `microprice_divergence` | same | price scale | yes |
| 6 | `spread_bps` | `bid/ask_px_00` | $[0,\infty)$ bps | yes |
| 7 | `order_book_imbalance` (3L) | `bid/ask_sz_00–02` | $[-1,1]$ | yes |
| 8 | `depth_ratio` | `bid/ask_sz_00–04` | $[0,\infty)$ | yes |
| 9 | `vwmp_skew` | `bid/ask_px/sz_00–02` | unbounded | yes |
| 10–11 | `bid_ask_slope` bid/ask | `bid/ask_px/sz_00–04` | $[0,\infty)$ | yes |
| 12–23 | `column_value` raw L0–L2 | passthrough | price/size | yes |
| 24 | `ofi` | `bid/ask_px/sz_00` | unbounded | yes |
| 25 | `ofi_rolling` (W=50) | same | unbounded | yes |
| 26 | `queue_depletion` bid | `bid_sz_00` | $[0,1]$ | yes |
| 27 | `queue_depletion` ask | `ask_sz_00` | $[0,1]$ | yes |
| 28 | `mid_price_acceleration` | `bid/ask_px_00` | unbounded | yes |
| 29 | `inter_event_time` | index timestamps | $[0,\infty)$ | yes |
| 30 | `spread_ratio` (W=100) | `bid/ask_px_00` | $[0,\infty)$ | yes |
| 31 | `book_convexity` bid | `bid_px_00–02` | unbounded | yes |
| 32 | `book_convexity` ask | `ask_px_00–02` | unbounded | yes |
| 33 | `order_count_imbalance` L0 | `bid/ask_ct_00` | $[-1,1]$ | yes |
| 34 | `signed_trade_flow` (W=100) | `action`, `side`, `size` | unbounded | yes |
