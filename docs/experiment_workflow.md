# Trading RL Experiment Workflow

This document describes the complete workflow for running trading reinforcement learning experiments in this project.

## Overview

The experiment system is designed to train DDPG (Deep Deterministic Policy Gradient) agents on trading environments with comprehensive MLflow tracking, evaluation, and visualization.

## Workflow Diagram

```mermaid
flowchart TD
    A[CLI Command: experiment] --> B[Load Configuration]
    B --> C[run_multiple_experiments]
    C --> D{For each trial}
    
    D --> E[run_single_experiment]
    E --> F[Setup Logging & Seed]
    F --> G[Prepare Data]
    G --> H[Create Environment]
    H --> I[Create Actor & Value Networks]
    I --> J[Setup DDPG Loss & Optimizer]
    J --> K[Create Data Collector]
    K --> L[Initialize Replay Buffer]
    L --> M[Start MLflow Run]
    
    
    M --> N[Training Loop]
    N --> O{Training Steps < Max?}
    O -->|Yes| P[Collect Data]
    P --> Q[Update Replay Buffer]
    Q --> R[Optimize Networks]
    R --> S[Log Metrics]
    S --> T{Evaluation Interval?}
    T -->|Yes| U[Evaluate Agent]
    U --> V[Log Episode Metrics]
    V --> O
    T -->|No| O
    
    O -->|No| W[Final Evaluation]
    W --> X[Generate Plots]
    X --> Y[Save Checkpoint]
    Y --> Z[Log Artifacts]
    Z --> AA[End MLflow Run]
    
    AA --> BB{More Trials?}
    BB -->|Yes| D
    BB -->|No| CC[Launch Dashboard?]
    CC -->|Yes| DD[MLflow UI]
    CC -->|No| EE[End]
    DD --> EE
    
    style A fill:#e1f5fe
    style E fill:#f3e5f5
    style N fill:#fff3e0
    style W fill:#e8f5e8
    style EE fill:#ffebee
```

## Component Details

### 1. Configuration Loading
- Loads YAML configuration file or uses defaults
- Validates configuration parameters
- Sets up experiment-specific parameters

### 2. Data Preparation
- **Function**: `prepare_data()`
- **Location**: `src/trading_rl/data_utils.py`
- **Steps**:
  - Load raw data from parquet files
  - Create technical features (if not using `--no-features`)
  - Apply normalization and preprocessing
  - Split into train/validation sets

### 3. Environment Creation
- **Function**: `create_environment()`
- **Location**: `src/trading_rl/train_trading_agent.py`
- **Steps**:
  - Initialize TorchRL trading environment
  - Apply transforms (StepCounter, etc.)
  - Set reward function and trading parameters

### 4. Network Architecture
- **Actor Network**: Policy network for action selection
- **Value Network**: Critic network for value estimation
- **Configurable**: Hidden dimensions, activation functions
- **Location**: `src/trading_rl/models.py`

### 5. Training Loop
The main training loop performs these steps cyclically:

#### Data Collection
- Use current policy to interact with environment
- Collect experience tuples (state, action, reward, next_state)
- Handle exploration vs exploitation

#### Network Updates
- Sample batches from replay buffer
- Compute DDPG losses (actor and critic)
- Update networks using gradient descent
- Apply soft target updates

#### Evaluation & Logging
- **Metrics Logged**:
  - Actor and critic losses
  - Episode rewards and lengths
  - Position change ratios
  - Portfolio performance
  - Network gradients and weights

### 6. MLflow Integration

```mermaid
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
Configuration dataclass containing all experiment parameters:
- **DataConfig**: Data paths, symbols, preprocessing options
- **NetworkConfig**: Architecture specifications
- **TrainingConfig**: Learning rates, batch sizes, etc.
- **LoggingConfig**: Output directories and verbosity

### Data Flow
1. **Raw Data** → `load_trading_data()`
2. **Feature Engineering** → `create_features()`
3. **Environment** → TorchRL trading environment
4. **Agent Training** → DDPG algorithm
5. **Evaluation** → Performance metrics and plots

### Evaluation Process
- **Function**: `evaluate_agent()`
- **Generates**:
  - Reward comparison plots (agent vs benchmarks)
  - Action sequence visualizations
  - Performance metrics
- **Benchmarks**:
  - Buy-and-hold strategy
  - Maximum theoretical profit

## Usage Examples

### Basic Experiment
```bash
python src/cli.py experiment --config ./src/configs/default.yaml --trials 3
```

### Custom Configuration
```bash
python src/cli.py experiment \
  --config ./src/configs/upward_drift_optimized.yaml \
  --trials 5 \
  --name "upward_drift_test" \
  --dashboard
```

### Configuration Options
- `--no-features`: Skip feature engineering, use raw OHLCV data
- `--dashboard`: Launch MLflow UI after completion
- `--clear-cache`: Clear data processing cache
- `--seed`: Set random seed for reproducibility

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

The system includes comprehensive error handling for:
- **Data Loading**: Missing files, corrupt data
- **Network Training**: Gradient explosions, convergence issues
- **Environment**: Invalid actions, state inconsistencies
- **MLflow**: Logging failures, artifact corruption

## Performance Optimization

- **Joblib Caching**: Expensive data operations are cached
- **Parallel Data Collection**: Vectorized environment interactions
- **Memory Management**: Efficient replay buffer implementation
- **GPU Support**: Automatic CUDA detection and usage

## Monitoring & Debugging

### Key Metrics to Watch
- **Actor Loss**: Should generally decrease over time
- **Critic Loss**: Should stabilize after initial training
- **Episode Reward**: Should show improvement trend
- **Position Change Ratio**: Indicates trading frequency vs exploration

### Common Issues
- **High Position Changes**: Often indicates scale/normalization issues
- **Flat Learning**: May need different learning rates or architecture
- **Evaluation Errors**: Usually related to data preprocessing mismatches