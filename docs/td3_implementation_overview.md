# TD3 Implementation Overview

## Summary
- Off-policy actor-critic with twin critics and target policy smoothing.
- Deterministic actor, critic(s) trained via TD3 loss with delayed policy updates.
- Replay buffer and exploration noise drive sample efficiency.

## Core Ideas
- **Twin Critics**: TD3Loss maintains two critic parameter sets and uses the minimum target prediction to curb overestimation.
- **Delayed Policy Updates**: Actor updates happen less frequently than critic updates.
- **Target Policy Smoothing**: Noise is added to target actions during critic updates for regularization.

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

    subgraph Evaluation
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

## Optimization Detail
```mermaid
flowchart LR
    subgraph Critic_Update["Critic optimization (every step)"]
        C1["Sample (s,a,r,s',d) from replay buffer"]
        C2["Compute target action: a_tilde = target_actor(s') + noise"]
        C3["Compute target Q: y = r + gamma*(1-d)*min(Q1_t, Q2_t)"]
        C4[Compute critic losses L1, L2]
        C5[Backprop critic loss]
        C6[optimizer_value.step]
        C1 --> C2 --> C3 --> C4 --> C5 --> C6
    end

    subgraph Actor_Update["Actor optimization (delayed)"]
        A1["Sample s from replay buffer"]
        A2["Compute action: a = actor(s)"]
        A3["Compute actor loss J = -Q1(s,a)"]
        A4[Backprop actor loss]
        A5[optimizer_actor.step]
        A6[SoftUpdate targets]
        A1 --> A2 --> A3 --> A4 --> A5 --> A6
    end

    Critic_Update --- Actor_Update
```

## Math Summary

Let the actor be $\mu_\theta(s)$ and critics be $Q_{\phi_1}(s,a), Q_{\phi_2}(s,a)$. Target networks are $\mu_{\bar\theta}, Q_{\bar\phi_1}, Q_{\bar\phi_2}$.

**Notation**
- $s, a, r, s', d$: state, action, reward, next state, and done flag.
- $\mathcal{B}$: replay buffer distribution.
- $\gamma$: discount factor.
- $\tau$: soft-update rate.
- $\theta$: actor parameters; $\bar\theta$: target actor parameters.
- $\phi_i$: critic parameters; $\bar\phi_i$: target critic parameters.
- $\tilde{a}$: target action after policy smoothing.
- $\sigma, c$: policy noise std and clip range.
- $d_{delay}$: policy delay (actor/target update period).

**Target policy smoothing**
$$
\tilde{a} = \mu_{\bar\theta}(s') + \epsilon,\quad \epsilon \sim \mathrm{clip}(\mathcal{N}(0,\sigma^2), -c, c)
$$

**Clipped double-Q target**
$$
y = r + \gamma (1-d)\, \min_{i=1,2} Q_{\bar\phi_i}(s', \tilde{a})
$$

**Critic loss (each critic)**
$$
L(\phi_i) = \mathbb{E}_{(s,a,r,s',d)\sim\mathcal{B}} \left( Q_{\phi_i}(s,a) - y \right)^2
$$

**Actor loss (delayed updates)**
$$
J(\theta) = - \mathbb{E}_{s\sim\mathcal{B}} \left[ Q_{\phi_1}(s, \mu_\theta(s)) \right]
$$

**Delayed update schedule**
$$
\text{Update actor/targets at step } t \text{ if } t \bmod d_{delay} = 0,\ \ d_{delay}=\text{policy\_delay}
$$

**Soft target updates**
$$
\bar\phi \leftarrow \tau \phi + (1-\tau)\bar\phi,\quad
\bar\theta \leftarrow \tau \theta + (1-\tau)\bar\theta
$$

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

## Integration Notes
- The existing `BaseTrainer` in `src/trading_rl/training.py` handles collection, replay, and logging; a `TD3Trainer` can reuse it similarly to `DDPGTrainer`/`PPOTrainer`.
- Ensure the environment exposes continuous actions (or a discretized wrapper is provided).
- Log both critics’ losses to monitor divergence; watch for action noise magnitude relative to spec bounds.
- Run via CLI by setting `training.algorithm: TD3` in a config (YAML or overrides) and invoking the same entrypoint used for PPO/DDPG (e.g., `python -m trading_rl.train_trading_agent --config path/to/config.yaml`).
