# Training Pipeline Architecture

End-to-end reference for the training pipeline, from CLI invocation to final evaluation.

---

## 1. Top-Level Flow

```mermaid
flowchart TD
    CLI["cli.py\ntrain command"] --> TC["TrainingCommand.execute()"]
    TC --> LC["_load_training_config()\nresolve YAML + CLI overrides"]
    LC --> RSE["run_single_experiment()"]
    RSE --> ESE["execute_single_experiment()"]

    ESE --> RR["_resolve_runtime()\nbuild_experiment_runtime()"]
    RR --> CPH["_configure_periodic_hooks()\neval + explainability at intervals"]
    CPH --> RTP["_run_training_phase()\ntrainer.train()"]
    RTP --> SFC["save_final_checkpoint()"]
    SFC --> EAS["evaluate_all_splits()\ntrain / val / test backtest"]
    EAS --> RPE["run_primary_split_explainability()"]
    RPE --> BFM["build_final_metrics()\nlog to MLflow"]
```

---

## 2. Data Loading: `build_prepared_dataset`

Two paths exist depending on whether a prepared parquet cache is already on disk.

```mermaid
flowchart TD
    BPD["build_prepared_dataset()"] --> CHK{"lazy_load=True\n+ parquet cache exists\n+ memmaps ready?"}

    CHK -- "Yes (fast path)" --> LPS["load_prepared_splits(prepared_dir)\nread parquet cache"]
    LPS --> LMP["load_memmap_paths(memmap_dir)"]
    LMP --> MD["return PreparedDataset"]

    CHK -- "No (full build)" --> DP{"data_paths list\nin config?"}

    DP -- "Yes (pooled)" --> BPS["_build_pooled_splits()\nsee Section 3"]
    DP -- "No (single symbol)" --> BSS["_build_single_symbol_splits()\nprepare_data() once"]
    BSS --> SSM["save_symbol_memmap()\nif memmap_dir set"]

    BPS --> SAVE{"lazy_load=True\n+ prepared_dir set?"}
    BSS --> SAVE
    SSM --> SAVE
    SAVE -- "Yes" --> SPQ["save_prepared_splits()\nwrite parquet cache"]
    SAVE -- "No" --> MD2["return PreparedDataset"]
    SPQ --> MD2
```

**`PreparedDataset` fields:**

| Field | Content |
|---|---|
| `train_df` | First symbol's train split (streaming mode) or full concat |
| `val_df` | All symbols' val splits concatenated |
| `test_df` | All symbols' test splits concatenated |
| `memmap_train_paths` | List of `MemmapPaths` per symbol, or `None` |
| `feature_columns` | `feature_*` column names |
| `price_column` | Column used for portfolio valuation |

---

## 3. Pooled Multi-Symbol Data Build: `_build_pooled_splits`

Peak memory stays at ~1 symbol at a time during feature engineering.

```mermaid
flowchart TD
    START["_build_pooled_splits(data_paths, memmap_dir)"] --> PIPE["_resolve_feature_pipeline()\nfrom YAML groups or feature_config"]
    PIPE --> LOOP["for each symbol path"]

    LOOP --> PD["prepare_data(path, cfg, pipeline)\nsee Section 4"]
    PD --> ECC["ensure_close_column_for_hft()\nderive mid-price if needed"]
    ECC --> EUI["ensure_unique_index_for_hft_tradingenv()\ndeduplicate timestamps (1µs gap)"]
    EUI --> MEM{"memmap_dir\nconfigured?"}

    MEM -- "Yes, file exists" --> LMM["load existing MemmapPaths\nfrom disk"]
    MEM -- "Yes, file missing" --> SMM["save_symbol_memmap(train_i)\nwrite .npy + index + columns"]
    MEM -- "No" --> WP

    LMM --> WP["write train/val/test\nto temp parquets"]
    SMM --> WP
    WP --> FREE["del train_i, val_i, test_i\ngc.collect()"]
    FREE --> LOOP

    LOOP --> CONCAT{"memmap_dir\nconfigured?"}
    CONCAT -- "Yes (streaming)" --> FIRST["train_df = first symbol parquet only\nskip full train concat"]
    CONCAT -- "No" --> FULL["train_df = concat all symbol parquets"]
    FIRST --> VCNC["val_df = concat all val parquets\ntest_df = concat all test parquets"]
    FULL --> VCNC
    VCNC --> EUI2["ensure_unique_index_for_hft_tradingenv()\ncross-symbol timestamp dedup on val/test"]
    EUI2 --> VPD["validate_prepared_data()"]
    VPD --> RET["return train_df, val_df, test_df,\ncollected_memmap_paths"]
```

---

## 4. Per-Symbol Feature Engineering: `prepare_data`

**Critical invariant: split before fit — no future data leaks into normalization.**

```mermaid
flowchart TD
    PDI["prepare_data(data_path, cfg, pipeline)"] --> CHK{"Feature cache\nhit?"}
    CHK -- "Yes" --> CACHED["load cached\ntrain/val/test parquets"]
    CHK -- "No" --> LOAD["load_trading_data()\nread parquet, drop NaNs"]
    LOAD --> SPLIT["chronological split\ntrain / val / test\nNO shuffling"]
    SPLIT --> FIT["pipeline.fit(train_df_raw)\ncompute normalization stats\non training data ONLY"]
    FIT --> TR["pipeline.transform(train_df_raw)"]
    FIT --> TV["pipeline.transform(val_df_raw)"]
    FIT --> TT["pipeline.transform(test_df_raw)"]
    TR --> CT["concat raw + features → train_df\ndel train_df_raw, train_features"]
    TV --> CV["concat raw + features → val_df\ndel val_df_raw, val_features"]
    TT --> CTT["concat raw + features → test_df\ndel test_df_raw, test_features"]
    CT --> CACHE["optionally cache\nto .cache/feature_transformation/"]
    CV --> CACHE
    CTT --> CACHE
    CACHE --> RET2["return train_df, val_df, test_df"]
    CACHED --> RET2
```

The feature pipeline uses **Welford's online algorithm** for cumulative running normalization — statistics are updated per-event so the scaler at step $t$ has only seen events $1..t$. Statistics reset at session boundaries (gaps > 1 hour).

---

## 5. Memmap Storage Format

A memory-mapped file lets the OS map bytes on disk directly into the process address space. Only the pages you actually read are loaded into RAM, so the full 200k-row train array stays on disk and each episode reads only its 50k-row window.

```mermaid
flowchart LR
    RAW["Raw parquet\n~7M tick events\n(4 trading days)"]
    SPLIT["Chronological split\ntrain: first train_size rows\nval + test: remainder"]
    FIT["Feature pipeline\nfit on train only"]
    NP["float32 array\nsaved to disk:\n{i}_train_data.npy"]
    MM["Memmap on disk\n(not loaded into RAM)"]
    RESET["reset()\npick random start\nin [0, train_size − episode_length]"]
    WINDOW["Episode window\nepisode_length rows\n(only this slice in RAM)"]
    ENV["TradingEnv\nwraps window as DataFrame"]
    AGENT["Agent steps\nthrough episode"]

    RAW --> SPLIT
    SPLIT --> FIT
    FIT --> NP
    NP --> MM
    MM --> RESET
    RESET --> WINDOW
    WINDOW --> ENV
    ENV --> AGENT
```

With `train_size=200_000` and `episode_length=50_000` there are at most 150k possible start offsets, giving a sliding window of episode positions. With 2M training steps at 200 frames/batch, the agent visits many overlapping windows across the full 200k training rows.

Each symbol produces three files in `memmap_dir/`:

```
{i}_train_data.npy     float32 array, shape (n_rows, n_cols)
{i}_train_index.npy    int64 array,   shape (n_rows,)  — nanosecond timestamps
{i}_columns.json       list[str]      — column names
```

`MemmapPaths` records `data_path`, `index_path`, `n_rows`, and `columns`.

---

## 6. Environment Construction: `AlgorithmicEnvironmentBuilder.create()`

```mermaid
flowchart TD
    CREATE["AlgorithmicEnvironmentBuilder.create(train_df, config)"] --> RMP["_resolve_memmap_paths(config)\ncheck memmap_dir"]
    RMP --> MPATH{"memmap_paths\nfound?"}

    MPATH -- "Yes" --> BACK{"backend?"}
    BACK -- "tradingenv" --> STXY["StreamingTradingEnvXY\nsee Section 7"]
    BACK -- "gym_trading_env" --> STE["StreamingTradingEnv\nnp.memmap episode windows"]
    STXY --> GW["GymWrapper\n(TorchRL)"]
    STE --> GW
    GW --> SC["StepCounter transform"]
    SC --> ENV["TransformedEnv\n(returned)"]

    MPATH -- "No (fallback)" --> ALGO{"algorithm?"}
    ALGO -- "TD3 / DDPG" --> CONT["continuous tradingenv backend"]
    ALGO -- "PPO" --> PPO["discrete or continuous\nper config"]
    CONT --> BBE["build_backend_env(train_df, config)"]
    PPO --> BBE
    BBE --> ENV
```

---

## 7. Streaming Episode Sampling: `StreamingTradingEnvXY`

On every `reset()` a fresh episode window is sampled from a random symbol.
Peak memory ≈ `episode_length × n_features × 4 bytes`.

```mermaid
sequenceDiagram
    participant Collector as SyncDataCollector
    participant Env as StreamingTradingEnvXY
    participant MM as numpy memmap files
    participant TEnv as tradingenv.TradingEnv

    Collector->>Env: reset()
    Env->>Env: pick random symbol idx
    Env->>Env: pick random start row
    Env->>MM: np.load(data_path, mmap_mode='r')[start:end]
    MM-->>Env: float32 window array (episode_length × n_cols)
    Env->>Env: reconstruct DataFrame + DatetimeIndex
    Env->>TEnv: _build_inner_env(window_df)
    TEnv-->>Env: fresh TradingEnv (broker reset, new prices)
    Env->>TEnv: reset()
    TEnv-->>Env: {"CustomFeature": obs_array}
    Env->>Env: _extract_obs() → flat np.ndarray
    Env-->>Collector: (obs, {})

    loop each step
        Collector->>Env: step(action)
        Env->>TEnv: step(action)
        TEnv-->>Env: {"CustomFeature": obs}, reward, done, info
        Env->>Env: _extract_obs()
        Env-->>Collector: obs, reward, done, truncated, info
    end
```

---

## 8. Trainer Setup and TD3 Training Loop

```mermaid
flowchart TD
    BT["_build_trainer(env, config, algorithm)"] --> CLS["resolve trainer class\nTD3Trainer / PPOTrainer / DDPGTrainer"]
    CLS --> BM["build_models(n_obs, n_act, config, env)\ncreate actor + critic networks"]
    BM --> INIT["Trainer.__init__(actor, critics, env, config)"]
    INIT --> RB["ReplayBuffer\nLazyTensorStorage(buffer_size)"]
    INIT --> SDC["SyncDataCollector(env, actor,\nframes_per_batch, total_frames)"]

    SDC --> TLOOP["Training loop"]
    TLOOP --> COLLECT["collect frames_per_batch transitions\nactor + env interaction"]
    COLLECT --> STORE["store batch in ReplayBuffer"]
    STORE --> OPTIM["optim_steps_per_batch × update steps:\nsample(sample_size) → TD3Loss\nupdate actor + critics"]
    OPTIM --> SU["SoftUpdate target networks\n(tau)"]
    SU --> EVAL{"eval_interval\nreached?"}
    EVAL -- "Yes" --> PE["periodic evaluation\non train split window"]
    EVAL -- "No" --> TLOOP
    PE --> TLOOP
```

**TD3-specific details:**
- Two independent Q-networks to reduce overestimation bias
- Policy delay: actor updates every `policy_delay` critic updates (default 2)
- Target policy smoothing: Gaussian noise clipped to `noise_clip` added to target actions
- Exploration: Gaussian noise with std `exploration_noise_std` during collection

---

## 9. Post-Training Evaluation

```mermaid
flowchart TD
    EAS2["evaluate_all_splits(trainer, train_df, val_df, test_df)"] --> ETRA["evaluate split=train\ndeterministic rollout"]
    EAS2 --> EVAL2["evaluate split=val"]
    EAS2 --> ETES["evaluate split=test"]

    ETRA --> METRICS["compute per-split metrics:\nSharpe ratio, total return,\nmax drawdown, win rate,\nstatistical significance tests"]
    EVAL2 --> METRICS
    ETES --> METRICS

    METRICS --> PSR["resolve_primary_split_result()\nprimary = test"]
    PSR --> EXPL["run_primary_split_explainability()\npermutation importance\nintegrated gradients"]
    EXPL --> LOG["log_final_metrics()\nMLflow + artifacts"]
```

---

## 10. Cache and Artifact Locations

| Path | Content |
|---|---|
| `data/raw/stocks/` | Raw MBP-10 parquet files from DataBento |
| `data/prepared/{name}/` | Cached train/val/test parquets (lazy_load fast path) |
| `data/memmap/{name}/` | Per-symbol numpy memmap arrays for streaming |
| `.cache/feature_transformation/` | Per-symbol feature engineering cache (keyed by file hash + config hash) |
| `mlruns/` | MLflow experiment tracking |
| `checkpoints/` | Model checkpoints (weights + optimizer state) |
