# HFT LOB Feature Formulas

Definitions in `src/configs/features/hft_lob_features.yaml`.
Implementation in `src/trading_rl/features/lob_features.py`.

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
**Literature:** Cao, Chen & Fan (2004) *The information content of an open limit-order book*; Biais, Hillion & Spatt (1995) *An empirical analysis of the limit order book*

$$\text{BookPressure}_i = \frac{V^{bid}_i - V^{ask}_i}{V^{bid}_i + V^{ask}_i}$$

Range $[-1, +1]$. Zero when total volume at level $i$ is zero.
$+1$: all volume on bid; $-1$: all volume on ask.

---

### 2. Microprice

**Feature type:** `microprice` | **Name:** `hft_microprice`
**Literature:** Stoikov (2018) *The micro-price: a high-frequency estimator of future prices*

$$\text{Microprice} = \frac{V^{ask}_0 \cdot P^{bid}_0 + V^{bid}_0 \cdot P^{ask}_0}{V^{bid}_0 + V^{ask}_0}$$

Fallback to $\frac{P^{bid}_0 + P^{ask}_0}{2}$ when total size is zero.
Pulls the fair-value estimate toward the side with less liquidity — the side
more likely to move first.

---

### 3. Microprice Divergence

**Feature type:** `microprice_divergence` | **Name:** `hft_microprice_divergence`
**Literature:** Stoikov (2018) *The micro-price: a high-frequency estimator of future prices*

$$\text{MicroDiv} = \text{Microprice} - \frac{P^{bid}_0 + P^{ask}_0}{2}$$

Isolates the imbalance signal from the absolute price level.

---

### 4. Spread in Basis Points

**Feature type:** `spread_bps` | **Name:** `hft_spread_bps`
**Literature:** Kyle (1985) *Continuous auctions and insider trading*; Glosten & Milgrom (1985) *Bid, ask and transaction prices in a specialist market*

$$\text{SpreadBps} = \frac{P^{ask}_0 - P^{bid}_0}{\frac{P^{bid}_0 + P^{ask}_0}{2}} \times 10{,}000$$

Units: basis points. Regime indicator for liquidity cost.

---

### 5. Order Book Imbalance (multi-level)

**Feature type:** `order_book_imbalance` | **Name:** `hft_order_book_imbalance_3l`
**Literature:** Cont, Kukanov & Stoikov (2014) *The price impact of order book events*; Biais, Hillion & Spatt (1995)

$$\text{OBI} = \frac{\sum_{i=0}^{N-1}(V^{bid}_i - V^{ask}_i)}{\sum_{i=0}^{N-1}(V^{bid}_i + V^{ask}_i)}$$

With $N=3$ levels. More robust than single-level book pressure.
Range $[-1, +1]$.

---

### 6. Depth Ratio

**Feature type:** `depth_ratio` | **Name:** `hft_depth_ratio`
**Literature:** Biais, Hillion & Spatt (1995); no single canonical citation — derived from market depth concept in Glosten (1994) *Is the electronic open limit order book inevitable?*

$$\text{DepthRatio} = \frac{V^{bid}_0 + V^{ask}_0}{\sum_{i=1}^{4}(V^{bid}_i + V^{ask}_i)}$$

High ratio: thin book behind best price (fragile liquidity).
Low ratio: deep liquidity support.

---

### 7. VWMP Skew

**Feature type:** `vwmp_skew` | **Name:** `hft_vwmp_skew`
**Literature:** Extension of Stoikov (2018) microprice framework to multi-level volume-weighted price; no single canonical citation.

$$\text{VWMP} = \frac{\sum_{i=0}^{2}(P^{bid}_i \cdot V^{bid}_i + P^{ask}_i \cdot V^{ask}_i)}{\sum_{i=0}^{2}(V^{bid}_i + V^{ask}_i)}$$

$$\text{VWMPSkew} = \frac{\text{VWMP} - \text{MidPrice}}{P^{ask}_0 - P^{bid}_0}$$

Positive: deep liquidity priced above mid (bid-heavy book).
Negative: ask-heavy deeper in the book.

---

### 8–9. Bid / Ask Slope

**Feature type:** `bid_ask_slope` | **Names:** `hft_bid_slope`, `hft_ask_slope`
**Literature:** Bouchaud, Mézard & Potters (2002) *Statistical properties of stock order books*; related to Kyle (1985) lambda (price impact per unit volume)

$$\text{BidSlope} = \frac{P^{bid}_0 - P^{bid}_4}{\sum_{i=0}^{4} V^{bid}_i} \qquad
\text{AskSlope} = \frac{P^{ask}_4 - P^{ask}_0}{\sum_{i=0}^{4} V^{ask}_i}$$

Price elasticity of each side across 5 levels. Higher slope = thinner book,
higher market impact per unit volume.

---

### 10. Raw Prices and Sizes (L0–L2)

**Feature type:** `column_value` | 12 features: `hft_bid/ask_px/sz_00/01/02`
**Literature:** No specific citation. Included to allow the agent to discover non-linear combinations not captured by the engineered features; standard practice in LOB-based deep RL (Nevmyvaka, Feng & Kearns, 2006 *Reinforcement learning for optimized trade execution*).

Pass-through of raw LOB columns after normalization. Allows the agent to
discover its own combinations without being constrained to the engineered
features above.

---

## New alpha-oriented features

All validated against the AAPL MBP-10 dataset (2.8M rows, zero nulls).

---

### 11. Order Flow Imbalance (OFI)

**Feature type:** `ofi`
**Literature:** Cont, Kukanov & Stoikov (2014) *The price impact of order book events* — the primary empirical paper establishing OFI as the dominant short-term price predictor in LOB data.

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
**Literature:** Extension of Cont, Kukanov & Stoikov (2014); multi-horizon OFI used in Kolm, Turiel & Westray (2023) *Deep order flow imbalance*.

$$\text{RollingOFI}_{W,t} = \sum_{\tau=t-W+1}^{t} \text{OFI}_\tau$$

Captures persistent directional pressure over multiple tick horizons.
Use multiple instances with different windows (e.g. 10, 50, 200) to give the
agent short-, medium-, and longer-term flow signals simultaneously.

**Feasibility:** derived from OFI; same inputs.

---

### 13. Queue Depletion Rate

**Feature type:** `queue_depletion` | Param: `side` (`"bid"` or `"ask"`)
**Literature:** Foucault, Kadan & Kandel (2005) *Limit order book as a market for liquidity*; practitioner usage described in Harris (2013) *Maker-taker pricing effects on market quotations*. No single canonical citation for the depletion ratio formulation.

$$\text{Depletion}^{side}_t = \frac{\max\!\left(V^{side}_{0,t-1} - V^{side}_{0,t},\ 0\right)}{V^{side}_{0,t-1}}$$

Measures how fast the top-of-book queue is being consumed. A rapidly
depleting bid queue signals imminent downward price pressure; ask depletion
signals upward pressure. First row is 0.
Range $[0, 1]$.

**Feasibility:** fully computable from consecutive `bid_sz_00` / `ask_sz_00`.

---

### 14. Mid-Price Acceleration

**Feature type:** `mid_price_acceleration`
**Literature:** No single canonical citation. Second finite difference used as a regime indicator in Abergel et al. (2016) *Limit order books* (Cambridge); the acceleration concept is related to momentum factor research (Jegadeesh & Titman, 1993) applied at tick frequency.

$$M_t = \frac{P^{bid}_{0,t} + P^{ask}_{0,t}}{2}$$

$$\text{Accel}_t = M_t - 2M_{t-1} + M_{t-2}$$

Second finite difference of mid-price. Positive: price accelerating upward.
Negative: decelerating or reversing. Helps distinguish momentum from
mean-reversion regimes at tick frequency. First two rows are 0.

**Feasibility:** fully computable from `bid_px_00`, `ask_px_00`.

---

### 15. Log Inter-Event Time

**Feature type:** `inter_event_time`
**Literature:** Engle (2000) *The econometrics of ultra-high-frequency data* (ACD model); Hasbrouck (1991) *Measuring the information content of stock trades*. Inter-event time as an adverse-selection proxy is standard in market microstructure.

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
**Literature:** Derived from Amihud & Mendelson (1986) *Asset pricing and the bid-ask spread*; relative spread used as a toxicity proxy in Easley, López de Prado & O'Hara (2012) *Flow toxicity and liquidity in a high-frequency world*.

$$\text{SpreadRatio}_t = \frac{\text{SpreadBps}_t}{\overline{\text{SpreadBps}}_{W,t}}$$

where $\overline{\text{SpreadBps}}_{W,t}$ is the rolling mean over $W$ ticks.

Values $> 1$: spread wider than recent average — elevated toxicity / adverse
selection risk. Values $< 1$: tighter cost environment.
Defaults to 1.0 during warm-up when rolling mean is zero.

**Feasibility:** fully computable from `bid_px_00`, `ask_px_00`.

---

### 17. Book Convexity

**Feature type:** `book_convexity` | Param: `side` (`"bid"` or `"ask"`)
**Literature:** Bouchaud, Mézard & Potters (2002) *Statistical properties of stock order books*; Potters & Bouchaud (2003) *More statistical properties of order books*. Price-level curvature used as a market impact measure.

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
**Literature:** Cao, Chen & Fan (2004) *The information content of an open limit-order book*; complements volume-based imbalance as in Glosten (1994). Count-vs-volume distinction discussed in Biais, Hillion & Spatt (1995).

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
**Literature:** Lee & Ready (1991) *Inferring trade direction from intraday data* (tick rule for trade classification); cumulative delta widely used in practitioner literature (e.g. Jansen, 2020 *Machine learning for algorithmic trading*). Distinct from OFI — measures executed trade flow, not order book changes.

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

### 22. Volume Adjusted Mid Price (VAMP)

**Feature type:** `price_vamp` | Params: `levels` (default 5), `notional` (default 1000), `bid_notional`, `ask_notional`
**Literature:** SSRN: 4351947 — short-term crypto price prediction; implementation adapted from `quantpylib.hft.cfeatures.vamp`.

Computes a volume-weighted mid price that accounts for order book depth and
imbalance, as opposed to the simple mid price which only considers the best
bid and best ask.

The algorithm walks into the book from both sides up to a notional amount:

1. **impact_price(asks, bid_notional):** Walk up the ask book, consuming volume
   until `bid_notional` dollars are spent. Returns the volume-weighted average
   ask price $P_1$.
2. **impact_price(bids, ask_notional):** Walk down the bid book, consuming
   volume until `ask_notional` dollars are spent. Returns the volume-weighted
   average bid price $P_2$.
3. $\text{VAMP} = \frac{P_1 + P_2}{2}$

For each level $i$ with price $p_i$ and size $v_i$:

$$\text{impact\_price} = \frac{\sum_{i} p_i \cdot \min\!\left(v_i,\ \frac{N_{\text{remaining}}}{p_i}\right)}{\sum_{i} \min\!\left(v_i,\ \frac{N_{\text{remaining}}}{p_i}\right)}$$

where $N_{\text{remaining}}$ is the notional still to be filled. The walk
stops early when the notional is fully consumed.

- **Small notional** (e.g., 1000): VAMP close to simple mid-price (only BBA
  involved); captures best-level imbalance similar to microprice.
- **Large notional** (e.g., 5000+): VAMP deviates from mid-price toward the
  side with more depth; captures deeper structural imbalance.
- The `notional` should be calibrated to the instrument — for high-priced
  assets like BTC, a larger notional is needed to walk past the best level.

**Params:**
- `levels`: Number of book levels to walk (default: 5, i.e., L0–L4)
- `notional`: Dollar amount to walk into the book on each side (default: 1000)
- `bid_notional`: Separate notional for the bid side (falls back to `notional` if 0)
- `ask_notional`: Separate notional for the ask side (falls back to `notional` if 0)

**Feasibility:** fully computable from `bid_px_00..04`, `ask_px_00..04`,
`bid_sz_00..04`, `ask_sz_00..04` (with default `levels=5`).

---

### 23. Cancel-to-Trade Ratio

**Feature type:** `cancel_to_trade_ratio` | Param: `window` (default 200), `action_col` (default "action")
**Literature:** Hasbrouck & Saar (2009) *Technology and liquidity provision: The blurring of the definition of liquidity*; Menkveld (2013) *High frequency trading and the new market makers*. Cancel-to-trade ratio used as a spoofing detection signal in SEC/CFTC enforcement guidelines.

Measures the fraction of order book events that are cancellations relative to
actual trades. A high ratio indicates that liquidity providers are repeatedly
posting and cancelling orders without commitment — either as spoofing/layering
or as rapid adaptation to perceived information asymmetry.

$$\text{CancelToTrade}_{W,t} = \frac{\sum_{\tau=t-W+1}^{t} \mathbf{1}[\text{action}_\tau = C]}{\max\!\left(\sum_{\tau=t-W+1}^{t} \mathbf{1}[\text{action}_\tau = T],\ 1\right)}$$

- **Range:** $[0, \infty)$. Typical values for liquid US equities: 1–5.
- Values much greater than 5 indicate highly ephemeral liquidity.
- Complements `inter_event_time` (which measures timing, not cancellation intensity).
- Only `action == 'C'` and `action == 'T'` events contribute; add events are excluded.

**Feasibility:** `action` column confirmed non-null in dataset.

---

### 24. Multi-level Order Flow Imbalance

**Feature type:** `ofi_multilevel` | Param: `levels` (default 3)
**Literature:** Cont, Kukanov & Stoikov (2014) *The price impact of order book events* — the original paper defines OFI at each level with cross-price effects. Multi-level aggregation improves predictive power over best-level OFI alone.

Extends the best-level OFI to multiple book levels, handling the three cases
for each level as defined in the original paper:

For level $i$, bid side:
$$e^{B}_{i,t} = \begin{cases}
V^{B}_{i,t}                      & \text{if } P^{B}_{i,t} > P^{B}_{i,t-1} \quad \text{(price improved)} \\
V^{B}_{i,t} - V^{B}_{i,t-1}      & \text{if } P^{B}_{i,t} = P^{B}_{i,t-1} \quad \text{(queue changed)} \\
-V^{B}_{i,t-1}                   & \text{if } P^{B}_{i,t} < P^{B}_{i,t-1} \quad \text{(price deteriorated)}
\end{cases}$$

For level $i$, ask side (mirrored):
$$e^{A}_{i,t} = \begin{cases}
V^{A}_{i,t}                      & \text{if } P^{A}_{i,t} < P^{A}_{i,t-1} \quad \text{(price improved)} \\
V^{A}_{i,t} - V^{A}_{i,t-1}      & \text{if } P^{A}_{i,t} = P^{A}_{i,t-1} \quad \text{(queue changed)} \\
-V^{A}_{i,t-1}                   & \text{if } P^{A}_{i,t} > P^{A}_{i,t-1} \quad \text{(price deteriorated)}
\end{cases}$$

$$\text{MultiLevelOFI} = \sum_{i=0}^{N-1} (e^{B}_{i,t} - e^{A}_{i,t})$$

- **Range:** unbounded. Positive: net buying pressure across levels.
- Handles the key edge case: when a price level appears/disappears, the full
  volume at that level counts as flow, not just the queue size change.
- First row is 0 (no prior snapshot for comparison).
- Default `levels=3` covers the best 3 bid/ask levels. Use `levels=5` for deeper
  book coverage.

**Feasibility:** fully computable from `bid_px_00..02`, `ask_px_00..02`,
`bid_sz_00..02`, `ask_sz_00..02` (with default `levels=3`).

---

### 25. VPIN (Volume-synchronized Probability of Informed Trading)

**Feature type:** `vpin` | Param: `window` (default 100), `action_col`, `side_col`, `size_col`
**Literature:** Easley, Lopez de Prado & O'Hara (2012) *Flow toxicity and liquidity in a high-frequency world*; the original VPIN uses volume-clock bucketing with bulk volume classification. This implementation uses a rolling tick window with actual trade-side classification, which is more accurate when side labels are available.

Tick-based approximation of VPIN that measures the imbalance between buyer-
and seller-initiated trade volume normalized by total trade volume. High VPIN
indicates toxic order flow (informed trading) and signals market makers to
widen quotes.

$$\text{VPIN}_{W,t} = \frac{\left|\sum_{\tau=t-W+1}^{t} V^{B}_\tau - \sum_{\tau=t-W+1}^{t} V^{S}_\tau\right|}{\max\!\left(\sum_{\tau=t-W+1}^{t} (V^{B}_\tau + V^{S}_\tau),\ 1\right)}$$

where $V^{B}_\tau$ = trade size if `action=T, side=B` else 0, and
$V^{S}_\tau$ = trade size if `action=T, side=A` else 0.

- **Range:** $[0, 1]$. Values near 1: one-directional informed flow (toxic).
- Values near 0: balanced, uninformed flow.
- Distinct from `odd_lot_imbalance` which filters to small trades only.
- Distinct from `signed_trade_flow` which is unnormalized and cumulative.
- The original VPIN uses volume-clock bucketing; this tick-window approximation
  is compatible with the DataFrame-based feature pipeline.

**Feasibility:** `action`, `side`, `size` columns confirmed non-null in dataset.

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
| 35 | `odd_lot_trade_ratio` (W=200) | `action`, `size` | $[0,1]$ | yes |
| 36 | `odd_lot_imbalance` (W=200) | `action`, `side`, `size` | $[-1,1]$ | yes |
| 37 | `price_vamp` | `bid/ask_px/sz_00–04` | price scale | yes |
| 38 | `cancel_to_trade_ratio` (W=200) | `action` | $[0,\infty)$ | yes |
| 39 | `ofi_multilevel` (3L/5L) | `bid/ask_px/sz_00–04` | unbounded | yes |
| 40 | `vpin` (W=100) | `action`, `side`, `size` | $[0,1]$ | yes |

---

### 20. Odd Lot Trade Ratio

**Feature type:** `odd_lot_trade_ratio` | Param: `window` (default 200), `round_lot` (default 100)
**Literature:** Boehmer, Jones, Zhang & Zhang (2021) *Tracking retail investor activity*; Hu (2014) *Odd-lot trading and the markets*; Chordia, Goyal, Lehmann & Saar (2013) *High-frequency trading*.

An odd lot is a trade with `size < round_lot` (100 shares for US equities).
In AAPL MBP-10 data, 81% of all trades are odd-lot sized (median 25 shares),
reflecting the dominance of retail and algorithmic small-lot activity.

$$\text{OddLotRatio}_{W,t} = \frac{\sum_{\tau=t-W+1}^{t} \mathbf{1}[\text{action}_\tau = T,\ \text{size}_\tau < L]}{\max\!\left(\sum_{\tau=t-W+1}^{t} \mathbf{1}[\text{action}_\tau = T],\ 1\right)}$$

where $L$ = `round_lot`. Non-trade rows contribute 0 to both numerator and
denominator.

- **Range:** $[0, 1]$
- High ratio: retail-dominated tape, lower adverse selection risk
- Sudden drop: institutional participation increasing — higher adverse selection
- Warms up over the first `window` ticks from the start of each split

**Feasibility:** `action` and `size` confirmed non-null in dataset.

---

### 21. Odd Lot Imbalance

**Feature type:** `odd_lot_imbalance` | Param: `window` (default 200), `round_lot` (default 100)
**Literature:** Boehmer, Jones, Zhang & Zhang (2021); Kelley & Tetlock (2013) *How wise are crowds? Insights from retail orders and stock returns*. Directional decomposition of odd-lot flow into buy/sell imbalance.

Signed directional bias in odd-lot (retail) order flow, normalized by total
odd-lot volume. Isolates retail pressure from institutional flow (which
dominates round-lot and block trades).

$$B_W = \sum_{\tau} \text{size}_\tau \cdot \mathbf{1}[\text{action}_\tau=T,\ \text{side}_\tau=B,\ \text{size}_\tau < L]$$

$$S_W = \sum_{\tau} \text{size}_\tau \cdot \mathbf{1}[\text{action}_\tau=T,\ \text{side}_\tau=A,\ \text{size}_\tau < L]$$

$$\text{OddLotImbalance}_{W,t} = \frac{B_W - S_W}{\max(B_W + S_W,\ 1)}$$

- **Range:** $[-1, +1]$
- $+1$: all odd-lot flow is buyer-initiated (retail buying pressure)
- $-1$: all odd-lot flow is seller-initiated (retail selling pressure)
- Complements `signed_trade_flow` (which includes all trade sizes) by isolating
  the retail component

**Feasibility:** `action`, `side`, and `size` confirmed non-null in dataset.
Observed mean imbalance ~0.05 (slight retail buy bias over the sample period).
