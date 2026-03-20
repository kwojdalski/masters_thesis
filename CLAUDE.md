# Claude Code Guidelines for Python Data Science Project

This document provides guidelines for working with Claude Code on this Python data science and machine learning project.

## Project Context

This is a master's thesis project focusing on trading strategies and reinforcement learning algorithms. The codebase includes:

- Synthetic price data generation
- Trading environment implementations (TorchRL)
- Deep reinforcement learning algorithms (DQN, PPO)
- Custom trading features and reward functions
- Data processing pipelines

## Python Environment

- Use Python 3.x best practices
- Follow PEP 8 style guidelines
- Use type hints where appropriate
- Prefer numpy/pandas operations over loops for performance

## Data Science Workflow

### Exploratory Data Analysis
- Use pandas for data manipulation
- Create visualizations with plotnine
- Validate data quality before modeling

### Machine Learning Development
- Set random seeds for reproducibility
- Split data appropriately (train/validation/test)
- Log experiments and hyperparameters
- Use version control for models and data

### Code Organization
- Keep notebooks clean and well-documented
- Refactor reusable code into modules
- Separate data loading, processing, and modeling
- Use configuration files for hyperparameters

## Libraries and Tools

### Core Data Science Stack
- **NumPy (>=2)**: Numerical computing and array operations
- **Pandas (2.2.1)**: Data manipulation, time series analysis, and dataframes
- **Matplotlib (3.8.3)**: Plotting and visualization
- **Seaborn (>=0.11.0)**: Statistical data visualization
- **SciPy (>=1.7.0)**: Scientific computing and optimization
- **Scikit-learn (1.6.1)**: Traditional ML algorithms, preprocessing, metrics

### Deep Learning and Reinforcement Learning
- **PyTorch (≥2.2.1)**: Deep learning framework for neural networks
- **TorchRL (≥0.7.0)**: Modular RL library built on PyTorch
  - `torchrl.envs`: Environment wrappers (GymWrapper, TransformedEnv, GymEnv)
  - `torchrl.envs.transforms`: Transforms like StepCounter
  - `torchrl.modules`: Neural network modules (MLP, QValueModule, ProbabilisticActor, ValueOperator, EGreedyModule)
  - `torchrl.objectives`: Loss functions (DQNLoss, DDPGLoss, SoftUpdate)
  - `torchrl.collectors`: Data collection (SyncDataCollector)
  - `torchrl.data`: Replay buffers (ReplayBuffer, LazyTensorStorage, Bounded specs)
  - `torchrl.envs.utils`: Utilities like set_exploration_type
  - `torchrl.record`: Logging and recording (CSVLogger, VideoRecorder)
- **Gymnasium (0.29.1)**: Standard API for RL environments

### Development Tools
- **Ruff**
- **Quarto (≥0.1.0)**: Scientific and technical publishing
- **OmegaConf**: CLI parameters should be passed using `--config-override key=value` dotlist syntax, not ad-hoc flags.

### Custom Components
- Custom trading environments (Gymnasium-compatible)
- TorchRL-integrated trading strategies
- Custom reward functions for trading
- Synthetic price data generators

## Documentation Preferences

- Docstrings: Use NumPy or Google style
- Comments: Explain "why" not "what"
- Type hints: Include for function signatures
- README: Keep updated with setup instructions
- Don't use emojis, please

## Version Control

- Commit frequently with clear messages, every single time you make any change, there should be a commit.
- Don't commit large data files or model weights
- Use .gitignore for notebooks checkpoints
- Track experiment results separately

# Package Manager

This project uses uv. Run commands with `uv run <command>` (e.g., `uv run python src/cli.py ...`, `uv run pytest`). Do not activate the venv manually.
