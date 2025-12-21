# Parameter FAQ - Trading RL Experiments

This document explains all parameters logged in this experiment.

## Dataset Parameters

### `data_start_date` / `data_end_date`
- **Format**: YYYY-MM-DD (e.g., "2023-01-01")
- **Purpose**: Define the time range of the training dataset
- **Impact**: Larger date ranges provide more diverse market conditions but increase training time

### `data_size`
- **Definition**: Total number of data points (rows) in the complete dataset
- **Typical Range**: 1000 - 100,000+ depending on timeframe and duration
- **Note**: Includes both training and potential test data

### `train_size`
- **Definition**: Number of data points actually used for training
- **Relationship**: Always =< `data_size`
- **Recommendation**: Use 70-80% of total data for training, reserve rest for testing

## Environment Parameters

### `trading_fees`
- **Format**: Decimal (e.g., 0.001 = 0.1%)
- **Purpose**: Transaction cost per trade
- **Impact**: Higher fees penalize frequent trading, encouraging longer-term strategies
- **Typical Values**:
  - Crypto: 0.0001 - 0.005 (0.01% - 0.5%)
  - Stocks: 0.0001 - 0.001 (0.01% - 0.1%)

### `borrow_interest_rate`
- **Format**: Decimal annual rate (e.g., 0.05 = 5% per year)
- **Purpose**: Cost of borrowing capital for short positions
- **Impact**: Affects profitability of short strategies
- **Note**: Set to 0 if short selling is not allowed

### `positions`
- **Options**:
  - `[-1, 0, 1]`: Short, neutral, long positions
  - `[0, 1]`: Neutral, long only (no short selling)
- **Impact**: Constrains available trading strategies
- **Recommendation**: Start with `[0, 1]` for simpler problems

## Network Architecture Parameters

### `actor_hidden_dims`
- **Format**: List of integers (e.g., `[64, 32]`)
- **Purpose**: Hidden layer sizes for the actor (policy) network
- **Impact**: 
  - Larger networks: More capacity but slower training
  - Smaller networks: Faster but may underfit
- **Typical Values**: `[64, 32]`, `[128, 64]`, `[256, 128]`

### `value_hidden_dims`
- **Format**: List of integers (e.g., `[64, 32]`)
- **Purpose**: Hidden layer sizes for the value (critic) network
- **Note**: Often same size as actor network but can be different
- **Rule of Thumb**: Value network can be slightly larger than actor

### `n_observations`
- **Definition**: Input dimension (number of features fed to the network)
- **Examples**:
  - Basic OHLCV: 5 features
  - With technical indicators: 10-50 features
  - With additional features: 50+ features
- **Impact**: More features provide more information but increase complexity

### `n_actions`
- **Definition**: Output dimension (size of action space)
- **Values**:
  - Discrete actions: Number of possible positions (2 or 3)
  - Continuous actions: Usually 1 (position size)
- **Note**: Determined by environment configuration

## Training Parameters

### `seed`
- **Purpose**: Random seed for reproducible experiments
- **Importance**: Critical for comparing different configurations
- **Best Practice**: Use different seeds for multiple runs of same config

### `actor_lr` / `value_lr`
- **Format**: Float (e.g., 0.001, 3e-4)
- **Purpose**: Learning rates for actor and value networks respectively
- **Tuning Guidelines**:
  - Start with: 1e-3 or 3e-4
  - Too high: Training instability, poor convergence
  - Too low: Very slow learning
- **Common Practice**: Value network often has same or slightly higher LR

### `buffer_size`
- **Definition**: Capacity of experience replay buffer
- **Purpose**: Stores past experiences for training
- **Typical Values**: 10,000 - 1,000,000
- **Trade-offs**:
  - Larger buffer: More diverse experiences, better stability
  - Smaller buffer: Less memory usage, faster sampling

## Performance Metrics

### `final_reward`
- **Definition**: Cumulative reward achieved during evaluation
- **Interpretation**: Higher values indicate better performance
- **Note**: Compare across runs with same environment settings

### `training_steps`
- **Definition**: Total number of training iterations completed
- **Purpose**: Indicates training duration and computational cost
- **Relationship**: More steps generally lead to better performance (up to a point)

### `evaluation_steps`
- **Definition**: Number of environment steps used for final evaluation
- **Purpose**: Standardizes performance comparison across runs
- **Note**: Should be consistent across experiments for fair comparison

## Quick Start Guidelines

### For New Users:
1. Start with simple `[0, 1]` positions (long-only)
2. Use moderate network sizes: `[64, 32]`
3. Set learning rates to `1e-3`
4. Use realistic trading fees for your market

### For Hyperparameter Tuning:
1. Fix seed for initial comparisons
2. Try different network architectures
3. Experiment with learning rates (1e-4 to 1e-2)
4. Adjust buffer size based on available memory

### For Production:
1. Test with multiple seeds (5-10 runs)
2. Use market-realistic parameters
3. Include transaction costs and slippage
4. Validate on out-of-sample data
