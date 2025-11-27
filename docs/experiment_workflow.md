# Trading RL Experiment Workflow

This document describes the complete workflow for running trading reinforcement learning experiments in this project.

## Overview

The experiment system can train PPO, DDPG, or TD3 agents on trading environments with comprehensive MLflow tracking, evaluation, and visualization. Choose the algorithm via `training.algorithm` in the config (PPO, DDPG, or TD3).

## Workflow Diagram

``` mermaid
flowchart TD
    A[CLI: python src/cli.py experiment] --> B[ExperimentCommand.execute]
    B --> C{--scenario provided?}
    C -->|Yes| D[Load scenario config<br/>from src/configs/]
    C -->|No| E[Load config from<br/>--config path]
    D --> F[ExperimentParams validation]
    E --> F
    F --> G{--generate-data?}
    G -->|Yes| H[DataGeneratorCommand.execute<br/>Generate synthetic data]
    G -->|No| I[run_multiple_experiments]
    H --> I
    
    I --> J{For each trial}
    J --> K[run_single_experiment]
    K --> L[Setup Logging & Seed]
    L --> M[Prepare Data]
    M --> N[Create Environment]
    N --> O{Algorithm type?}
    O -->|PPO| P[Create PPO Actor-Critic]
    O -->|DDPG| Q[Create Actor & Value Networks]
    O -->|TD3| Q2[Create TD3 Actor & Stacked Twin Q-Net]
    P --> R[Setup PPO Loss & Optimizer]
    Q --> S[Setup DDPG Loss & Optimizer]
    Q2 --> S2[Setup TD3 Loss & Optimizers]
    R --> T[Create Data Collector]
    S --> T
    S2 --> T
    T --> U[Initialize Replay Buffer]
    U --> V[Start MLflow Run]
    
    V --> W[Training Loop]
    W --> X{Training Steps < Max?}
    X -->|Yes| Y[Collect Data]
    Y --> Z[Update Replay Buffer]
    Z --> AA[Optimize Networks]
    AA --> BB[Log Metrics]
    BB --> CC{Evaluation Interval?}
    CC -->|Yes| DD[Evaluate Agent]
    DD --> EE[Log Episode Metrics]
    EE --> X
    CC -->|No| X
    
    X -->|No| FF[Final Evaluation]
    FF --> GG[Generate Plots]
    GG --> HH[Save Checkpoint]
    HH --> II[Log Artifacts]
    II --> JJ[End MLflow Run]
    
    JJ --> KK{More Trials?}
    KK -->|Yes| J
    KK -->|No| LL{--dashboard?}
    LL -->|Yes| MM[Launch MLflow UI]
    LL -->|No| NN[End]
    MM --> NN
    
    style A fill:#e1f5fe
    style B fill:#f3e5f5
    style W fill:#fff3e0
    style FF fill:#e8f5e8
    style NN fill:#ffebee
    style H fill:#ffe0b2
```

## Component Details

### 1. Configuration Loading

-   Loads YAML configuration file or uses defaults
-   Validates configuration parameters
-   Sets up experiment-specific parameters

### 2. Data Preparation

-   **Function**: `prepare_data()`
-   **Location**: `src/trading_rl/data_utils.py`
-   **Steps**:
    -   Load raw data from parquet files
    -   Create technical features (if not using `--no-features`)
    -   Apply normalization and preprocessing
    -   Split into train/validation sets

### 3. Environment Creation

-   **Function**: `create_environment()`
-   **Location**: `src/trading_rl/train_trading_agent.py`
-   **Steps**:
    -   Initialize TorchRL trading environment
    -   Apply transforms (StepCounter, etc.)
    -   Set reward function and trading parameters

### 4. Network Architecture

-   **Actor Network**: Policy network for action selection
-   **Value Network**: Critic network for value estimation (twin critics for TD3)
-   **Configurable**: Hidden dimensions, activation functions
-   **Location**: `src/trading_rl/models.py`

### 5. Training Loop

The main training loop performs these steps cyclically:

#### Data Collection

-   Use current policy to interact with environment
-   Collect experience tuples (state, action, reward, next_state)
-   Handle exploration vs exploitation

#### Network Updates

-   Sample batches from replay buffer
-   Compute losses (PPO, DDPG, or TD3)
-   Update networks using gradient descent
-   Apply soft target updates (DDPG/TD3)

#### Evaluation & Logging

-   **Metrics Logged**:
    -   Actor and critic losses
    -   Episode rewards and lengths
    -   Position change ratios
    -   Portfolio performance
    -   Network gradients and weights

### 6. MLflow Integration

``` mermaid
graph LR
    A[Experiment Run] --> B[Parameters]
    A --> C[Metrics]
    A --> D[Artifacts]
    
    B --> B1[Network Config]
    B --> B2[Training Config]
    B --> B3[Data Config]
    
    C --> C1[Losses]
    C --> C2[Rewards]
    C --> C3[Position Stats]
    C --> C4[Performance]
    
    D --> D1[Model Checkpoint]
    D --> D2[Evaluation Plots]
    D --> D3[Configuration Files]
    D --> D4[Training Logs]
```

## Key Components

### ExperimentConfig

Configuration dataclass containing all experiment parameters: - **DataConfig**: Data paths, symbols, preprocessing options - **NetworkConfig**: Architecture specifications - **TrainingConfig**: Learning rates, batch sizes, etc. - **LoggingConfig**: Output directories and verbosity

### Data Flow

1.  **Raw Data** → `load_trading_data()`
2.  **Feature Engineering** → `create_features()`
3.  **Environment** → TorchRL trading environment
4.  **Agent Training** → DDPG algorithm
5.  **Evaluation** → Performance metrics and plots

### Evaluation Process

-   **Function**: `evaluate_agent()`
-   **Generates**:
    -   Reward comparison plots (agent vs benchmarks)
    -   Action sequence visualizations
    -   Performance metrics
-   **Benchmarks**:
    -   Buy-and-hold strategy
    -   Maximum theoretical profit

## Usage Examples

### Basic Experiment

``` bash
python src/cli.py experiment --config ./src/configs/default.yaml --trials 3
```

### PPO/DDPG/TD3 Selection

- Set `training.algorithm` in your config to `PPO`, `DDPG`, or `TD3`.
- TD3 requires continuous-action setups (or a discretized wrapper) and twin Q-value networks; defaults are built when you pick TD3.

### Custom Configuration

``` bash
python src/cli.py experiment \
  --config ./src/configs/upward_drift_optimized.yaml \
  --trials 5 \
  --name "upward_drift_test" \
  --dashboard
```

### Configuration Options

-   `--no-features`: Skip feature engineering, use raw OHLCV data
-   `--dashboard`: Launch MLflow UI after completion
-   `--clear-cache`: Clear data processing cache
-   `--seed`: Set random seed for reproducibility

## Output Structure

```         
logs/
├── experiment_name/
│   ├── training.log
│   ├── model_checkpoint.pt
│   └── evaluation_plots/
└── mlruns/
    └── experiment_id/
        ├── run_id_1/
        ├── run_id_2/
        └── ...
```

## Error Handling

The system includes comprehensive error handling for: - **Data Loading**: Missing files, corrupt data - **Network Training**: Gradient explosions, convergence issues - **Environment**: Invalid actions, state inconsistencies - **MLflow**: Logging failures, artifact corruption

## Performance Optimization

-   **Joblib Caching**: Expensive data operations are cached
-   **Parallel Data Collection**: Vectorized environment interactions
-   **Memory Management**: Efficient replay buffer implementation
-   **GPU Support**: Automatic CUDA detection and usage

## Monitoring & Debugging

### Key Metrics to Watch

-   **Actor Loss**: Should generally decrease over time
-   **Critic Loss**: Should stabilize after initial training
-   **Episode Reward**: Should show improvement trend
-   **Position Change Ratio**: Indicates trading frequency vs exploration

### Common Issues

-   **High Position Changes**: Often indicates scale/normalization issues
-   **Flat Learning**: May need different learning rates or architecture
-   **Evaluation Errors**: Usually related to data preprocessing mismatches
