# End-to-End Workflow Overview

This diagram covers the full pipeline from raw tick data through feature research, RL training, and final evaluation.

```mermaid
flowchart TD
    subgraph ACQUIRE["1. Data Acquisition"]
        PARQUET["Raw MBP-10 parquet files\nper symbol · ~7M tick events\n(DataBento, us_hours filter)"]
    end

    subgraph FEATURE_RESEARCH["2. Offline Feature Research (optional)"]
        FR_SPLIT["Chronological split\ntrain_size rows for IC scoring"]
        FR_PIPE["Feature pipeline\nfit on train, transform both splits"]
        FR_IC["IC / ICIR scoring per feature\nrolling Spearman rank correlation\nvs Sharpe-proxy target"]
        FR_AGG["Aggregate across symbols\nmean ICIR ranking"]
        FR_SEL["Greedy conditional IC selection\nlinear residualisation for redundancy"]
        FR_YAML["selected_features.yaml\nreduced feature config"]

        FR_SPLIT --> FR_PIPE --> FR_IC --> FR_AGG --> FR_SEL --> FR_YAML
    end

    subgraph DATA_PREP["3. Data Preparation"]
        DP_SPLIT["Chronological split per symbol\ntrain / val / test"]
        DP_FIT["Feature pipeline\nfit on train only\nWelford online normalisation"]
        DP_CACHE["Parquet cache\ndata/prepared/"]
        DP_MEM["Memmap files per symbol\ndata/memmap/\n{i}_train_data.npy"]

        DP_SPLIT --> DP_FIT --> DP_CACHE
        DP_FIT --> DP_MEM
    end

    subgraph TRAIN["4. RL Training (TD3)"]
        ENV["StreamingTradingEnvXY\nrandom episode_length window\nfrom memmap at each reset()"]
        COLLECT["SyncDataCollector\nframes_per_batch transitions"]
        BUFFER["Replay Buffer\nbuffer_size transitions"]
        TD3["TD3Loss\ntwin critics · delayed actor update\ntarget policy smoothing"]
        SOFT["Soft target network update\nτ = 0.001"]
        CKPT["Checkpoint\nactor + critic weights\noptimizer state"]

        ENV --> COLLECT --> BUFFER --> TD3 --> SOFT
        SOFT --> COLLECT
        TD3 --> CKPT
    end

    subgraph EVAL["5. Evaluation"]
        BACKTEST["Deterministic rollout\nval split + test split"]
        METRICS["Per-split metrics\nSharpe · max drawdown\ntotal return · win rate"]
        STATS["Statistical significance tests\nt-test · Sharpe bootstrap\nvs buy-and-hold / TWAP / VWAP"]
        EXPL["Explainability\npermutation feature importance"]
        MLFLOW["MLflow\nparameters · metrics · artifacts"]

        BACKTEST --> METRICS --> STATS --> MLFLOW
        BACKTEST --> EXPL --> MLFLOW
    end

    PARQUET --> FR_SPLIT
    PARQUET --> DP_SPLIT
    FR_YAML -.->|"feeds feature config\nfor next training run"| DP_FIT
    DP_MEM --> ENV
    DP_CACHE --> BACKTEST
    CKPT --> BACKTEST
```

## Stage summary

| Stage | CLI command | Key config |
|---|---|---|
| Feature research | `cli.py feature-research --scenario <name>` | `data.train_size`, `research.horizons`, `research.top_k` |
| Data preparation | runs automatically inside `train` | `data.prepared_data_dir`, `data.memmap_dir` |
| Training | `cli.py train --scenario <name>` | `training.max_steps`, `training.algorithm` |
| Evaluation | runs automatically at end of `train` | `statistical_testing.enabled` |

## Key numbers (pooled HFT scenario)

| Parameter | Value | Meaning |
|---|---|---|
| Raw rows per symbol | ~7M | ~4 trading days of MBP-10 tick data |
| `train_size` | 200 000 | ~2.7 hours of tick events used for training |
| `episode_length` | 50 000 | ~41 minutes per episode window |
| Max start offsets | 150 000 | sliding window within the 200k train block |
| Symbols | 6 | AAPL, MSFT, TSLA, META, AMZN, AVGO |

See [training_pipeline.md](./training_pipeline.md) for detailed per-step diagrams.
