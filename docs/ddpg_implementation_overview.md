# DDPG Implementation Overview

## Summary
- Off-policy deterministic actor-critic for continuous actions.
- Uses replay buffer, target networks, and soft updates.
- Training logs and evaluations flow through MLflow.

## Core Ideas
- **Deterministic Policy**: Actor outputs continuous actions directly.
- **Target Networks**: Stabilize critic targets via lagged copies.
- **Replay Buffer**: Off-policy updates from stored transitions.

## Flow

```mermaid
flowchart TD
    subgraph CLI_Commands
        A[python src/cli.py experiment]
        A1[ExperimentCommand.execute]
        A2[--scenario or --config]
        A3[--generate-data flag]
    end

    subgraph Scenario_Loading
        B["ScenarioConfig<br/>sine_wave / mean_reversion / etc."]
        B1[Load from src/configs/*.yaml]
        B2[Standardized naming:<br/>yaml_name.parquet]
    end

    subgraph Data_Generation
        C["DataGeneratorCommand<br/>if --generate-data"]
        C1[Generate synthetic patterns:<br/>sine_wave, upward_drift, mean_reversion]
        C2[Save to data/raw/synthetic/]
    end

    subgraph DataPrep
        D[prepare_data<br/>load_trading_data<br/>create_features]
        D1[Load from standardized paths]
        D2["Feature engineering if enabled"]
    end

    subgraph EnvSetup
        E["create_environment<br/>Gym TradingEnv -> GymWrapper -> TransformedEnv StepCounter"]
        E1[Environment name from config.env.name]
    end

    subgraph Models
        F["create_ddpg_actor<br/>deterministic MLP -> action"]
        G["create_value_network<br/>ValueOperator over MLP"]
        F1[Architecture from config.network]
    end

    subgraph Trainer
        H["DDPGTrainer.__init__<br/>DDPGLoss + SoftUpdate<br/>ReplayBuffer LazyTensorStorage<br/>SyncDataCollector"]
        I[DDPGTrainer.train<br/>for each collector batch]
        J[ReplayBuffer.extend]
        K[ReplayBuffer.sample]
        L["DDPGLoss sample"]
        M["optimizer_actor.step"]
        N["optimizer_value.step"]
        O["SoftUpdate.step"]
        P[callback.log_training_step<br/>log_episode_stats -> MLflow]
    end

    subgraph Evaluation
        Q[evaluate_agent<br/>Deterministic rollout<br/>Random rollout<br/>compare_rollouts]
        R[Reward/Action plots]
    end

    subgraph Tracking
        S[MLflow<br/>Experiment: config.experiment_name<br/>Metrics + Params + Artifacts]
        T[Checkpoint Save<br/>logs/experiment_name/]
    end

    A --> A1 --> A2
    A2 --> B --> B1 --> B2
    A3 --> C --> C1 --> C2
    B2 --> D --> D1 --> D2
    C2 --> D
    D2 --> E --> E1
    E1 --> F --> F1
    E1 --> G
    F1 --> H
    G --> H
    H --> I
    I --> J --> K --> L
    L --> M
    L --> N
    L --> O
    I --> P --> S
    I --> T
    I --> Q --> R --> S
    R --> T
```

## Optimization Detail
```mermaid
flowchart LR
    subgraph Critic_Update["Critic optimization (every step)"]
        C1["Sample (s,a,r,s',d) from replay buffer"]
        C2["Compute target action: a_tgt = target_actor(s')"]
        C3["Compute target Q: y = r + gamma*(1-d)*Q_t(s', a_tgt)"]
        C4[Compute critic loss]
        C5[Backprop critic loss]
        C6[optimizer_value.step]
        C1 --> C2 --> C3 --> C4 --> C5 --> C6
    end

    subgraph Actor_Update["Actor optimization (every step)"]
        A1["Sample s from replay buffer"]
        A2["Compute action: a = actor(s)"]
        A3["Compute actor loss J = -Q(s,a)"]
        A4[Backprop actor loss]
        A5[optimizer_actor.step]
        A6[SoftUpdate targets]
        A1 --> A2 --> A3 --> A4 --> A5 --> A6
    end

    Critic_Update --- Actor_Update
```

## Math Summary

Let the actor be $\mu_\theta(s)$ and the critic be $Q_\phi(s,a)$. Target networks are $\mu_{\bar\theta}, Q_{\bar\phi}$.

**Notation**
- $s, a, r, s', d$: state, action, reward, next state, and done flag.
- $\mathcal{B}$: replay buffer distribution.
- $\gamma$: discount factor.
- $\tau$: soft-update rate.
- $\theta$: actor parameters; $\bar\theta$: target actor parameters.
- $\phi$: critic parameters; $\bar\phi$: target critic parameters.
- $a_{tgt}$: target action from target actor.

**Critic target**
$$
y = r + \gamma (1-d)\, Q_{\bar\phi}(s', \mu_{\bar\theta}(s'))
$$

**Critic loss**
$$
L(\phi) = \mathbb{E}_{(s,a,r,s',d)\sim\mathcal{B}} \left( Q_\phi(s,a) - y \right)^2
$$

**Actor loss**
$$
J(\theta) = - \mathbb{E}_{s\sim\mathcal{B}} \left[ Q_\phi(s, \mu_\theta(s)) \right]
$$

**Soft target updates**
$$
\bar\phi \leftarrow \tau \phi + (1-\tau)\bar\phi,\quad
\bar\theta \leftarrow \tau \theta + (1-\tau)\bar\theta
$$

## Components
- **CLI + configs**: scenario/config selection and optional data generation.
- **Environment**: Gym trading env wrapped for TorchRL and `StepCounter`.
- **Models**: deterministic actor (`create_ddpg_actor`) + critic (`create_value_network`).
- **Trainer**: `DDPGTrainer` with `DDPGLoss`, replay buffer, and `SoftUpdate`.

## Training Loop
- Collect batch → extend replay buffer → sample minibatches.
- Critic and actor updates each step.
- Soft-update target networks with `tau`.
