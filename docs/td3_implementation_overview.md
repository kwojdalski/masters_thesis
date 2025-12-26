# TD3 Implementation Overview

## Summary
- Off-policy actor-critic with twin critics and target policy smoothing.
- Deterministic actor, critic(s) trained via TD3 loss with delayed policy updates.
- Replay buffer and exploration noise drive sample efficiency.

## Flow
```mermaid
flowchart TD
    subgraph Entry
        A["CLI: training.algorithm TD3"]
        A1["Config -> ExperimentConfig"]
    end

    subgraph Data
        B["prepare_data -> DataFrame"]
        B1["create_features if enabled"]
    end

    subgraph Env
        C["create_environment<br/>Gym TradingEnv -> GymWrapper -> TransformedEnv StepCounter"]
    end

    subgraph Models
        D["create_td3_actor<br/>deterministic MLP -> action"]
        E["create_td3_qvalue_network<br/>single-head critic"]
    end

    subgraph Trainer
        F["TD3Trainer.__init__<br/>TD3Loss (actor + critic, num_qvalue_nets=2)<br/>SoftUpdate<br/>ReplayBuffer LazyTensorStorage<br/>SyncDataCollector"]
        G["TD3Trainer.train<br/>for each collector batch"]
        H["ReplayBuffer.extend"]
        I["ReplayBuffer.sample"]
        J["TD3Loss(sample)<br/>policy_noise + noise_clip"]
        K["optimizer_value.step (critics)"]
        L["optimizer_actor.step (delayed)"]
        M["SoftUpdate.step (targets)"]
        N["callback.log_training_step<br/>log_episode_stats -> MLflow"]
    end

    subgraph Eval
        O["evaluate_agent<br/>deterministic rollout"]
        P["Reward/Action plots"]
    end

    subgraph Tracking
        Q["MLflow metrics/params/artifacts"]
        R["Checkpoint Save logs/"]
    end

    A --> A1 --> B --> B1 --> C
    C --> D
    C --> E
    D --> F
    E --> F
    F --> G
    G --> H --> I --> J
    J --> K
    J --> L
    J --> M
    G --> N --> Q
    G --> R
    G --> O --> P --> Q
    P --> R
```

## Core Ideas
- **Twin Critics**: TD3Loss maintains two critic parameter sets and uses the minimum target prediction to curb overestimation.
- **Delayed Policy Updates**: Actor updates happen less frequently than critic updates.
- **Target Policy Smoothing**: Noise is added to target actions during critic updates for regularization.

## Components
- **CLI + configs**: `training.algorithm: TD3` selects TD3 trainer and models.
- **Models**: deterministic actor + critic; TD3Loss expands critic params to two critics.
- **Loss/optimizers**: separate Adam optimizers for actor and critic.
- **Collector/buffer**: `SyncDataCollector` + replay buffer with initial random exploration.

## Training Loop
- Collect batch → replay buffer → sample minibatches.
- Critic update every step; actor update delayed by `policy_delay`.
- Target policy smoothing via `policy_noise` and `noise_clip`.
- Soft-update target params with `tau`.

## Suggested Hyperparameters
- `policy_noise`: 0.2 (relative to action scale)
- `noise_clip`: 0.5
- `delay_actor`: True (actor every 2 critic steps)
- `gamma`: 0.99
- `tau`: 0.005–0.01
- Replay buffer: 1e5–1e6 transitions; batch 64–256

## Integration Notes
- The existing `BaseTrainer` in `src/trading_rl/training.py` handles collection, replay, and logging; a `TD3Trainer` can reuse it similarly to `DDPGTrainer`/`PPOTrainer`.
- Ensure the environment exposes continuous actions (or a discretized wrapper is provided).
- Log both critics’ losses to monitor divergence; watch for action noise magnitude relative to spec bounds.
- Run via CLI by setting `training.algorithm: TD3` in a config (YAML or overrides) and invoking the same entrypoint used for PPO/DDPG (e.g., `python -m trading_rl.train_trading_agent --config path/to/config.yaml`).
