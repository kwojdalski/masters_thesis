# The Big Picture of Reinforcement Learning (Enhanced Proposal)

Based on the [Reddit discussion](https://www.reddit.com/r/reinforcementlearning/comments/1prttxz/let_me_know_whether_my_bigpicture_understanding/), this document maps the landscape of Reinforcement Learning (RL) by starting with the theoretical ideal and introducing algorithms as solutions to specific practical constraints ("relaxations").

## 1. The Theoretical Foundation (The "Ideal")
At its core, RL models problems as **Markov Decision Processes (MDPs)**.
*   **Goal:** Find an optimal policy $\pi^*$ that maximizes expected return.
*   **The Math:** This is solved using the **Bellman Optimality Equation**.
*   **The Constraint:** To solve this *exactly*, you need:
    1.  A known model of the environment (transition probabilities $P(s'|s,a)$ and reward function $R(s,a)$).
    2.  A manageable (small/discrete) state and action space.
    3.  Full observability of the state.

## 2. Relaxing Assumptions (The "Real World")

Modern RL algorithms exist because real-world problems violate the "Ideal" constraints. We move from exact solutions to approximations by relaxing specific assumptions.

### Relaxation A: "We don't know the Model" (Estimation)
**Problem:** We don't know $P(s'|s,a)$. We can't calculate exact expectations.
**Solution:** We must **sample** the environment.
*   **Monte Carlo:** Wait for an episode to finish, then average the actual returns. (Unbiased, high variance).
    *   *Examples:* First-Visit MC, Every-Visit MC, MC Control
*   **Temporal Difference (TD):** Update estimates based on other estimates ($V(s_{t+1})$) without waiting for the end. This is **Bootstrapping**.
    *   *Examples:* TD(0), TD(λ), SARSA, Q-Learning, Expected SARSA, Double Q-Learning

### Relaxation B: "The State Space is too Big" (Approximation)
**Problem:** The world is continuous or has too many states to store in a table (e.g., pixels in an image).
**Solution:** Use **Function Approximation**. Instead of a table $Q(s,a)$, we learn a function $f(s,a; \theta)$ (e.g., a Neural Network) that estimates the value.
*   **Deep Q-Learning (DQN):** Combines Q-Learning (TD) with Neural Networks (Approximation).
    *   *Variants:* Double DQN, Dueling DQN, Prioritized Experience Replay, Rainbow DQN, Noisy DQN

### Relaxation C: "We want to plan" (Model-Based)
**Problem:** Model-free methods require massive amounts of experience.
**Solution:** **Indirect RL / Model-Based RL**.
1.  Learn the model of the world (approximate $P$ and $R$).
2.  Plan inside that mental model (e.g., using Dynamic Programming or Tree Search).
    *   *Examples:* Dyna-Q, World Models, MuZero, Dreamer

### Relaxation D: "Continuous Action Spaces"
**Problem:** Value-based methods struggle with continuous actions (can't maximize over infinite actions).
**Solution:** Optimize the **Policy** directly ($\pi(a|s)$) or use deterministic policies.
*   **Policy Gradients:** Adjust policy parameters $\theta$ to increase the probability of high-reward actions.
    *   *Examples:* REINFORCE, TRPO, PPO
*   **Deterministic Policy Gradients:** For continuous control with deterministic policies.
    *   *Examples:* DDPG, TD3
*   **Actor-Critic:** Learn a Policy ("Actor") *and* a Value function ("Critic") to reduce variance.
    *   *Examples:* A2C, A3C, PPO, SAC, TD3

### Relaxation E: "On-Policy vs Off-Policy"
**Problem:** Some algorithms can only learn from their current policy (on-policy), limiting sample efficiency.
**Solution:** **Off-Policy Learning** - learn from any experience, including old policies or other agents.
*   **On-Policy:** SARSA, A2C, A3C, PPO, TRPO
*   **Off-Policy:** Q-Learning, DQN, DDPG, TD3, SAC

### Relaxation F: "Partial Observability"
**Problem:** Agent cannot fully observe the state (e.g., vision with occlusions, hidden variables).
**Solution:** **Partially Observable MDPs (POMDPs)** - use history or memory.
*   *Examples:* Recurrent policies (LSTM/GRU), Attention mechanisms, Belief-state planning

---

## 3. Visual Map (Enhanced)

```mermaid
graph TD
    A[<b>The Core Problem</b><br/>Markov Decision Process] -->|Mathematical Solution| B(<b>Bellman Equation</b>)

    B --> C{<b>Constraint:</b><br/>Do we know the<br/>Transition Model?}

    C -->|Yes| D[<b>Dynamic Programming</b><br/>Value Iteration<br/>Policy Iteration<br/>Modified Policy Iteration]
    C -->|No| E[<b>Reinforcement Learning</b><br/>Learn from Experience]

    E --> F{<b>Constraint:</b><br/>How do we estimate<br/>future value?}

    F -->|Wait for end| G[<b>Monte Carlo</b><br/>First-Visit MC<br/>Every-Visit MC<br/>MC Control]
    F -->|Bootstrap| H[<b>Temporal Difference</b><br/>TD-0 / TD-λ<br/>SARSA / Expected SARSA<br/>Q-Learning / Double Q-Learning]

    E --> I{<b>Constraint:</b><br/>Is State Space<br/>Small/Discrete?}

    I -->|Yes| J[<b>Tabular Methods</b><br/>Q-Table<br/>State-Value Table]
    I -->|No - Too Big/Continuous| K[<b>Function Approximation</b><br/>Deep RL / Neural Networks]

    K --> L{<b>Constraint:</b><br/>Action Space Type?}

    L -->|Discrete| M[<b>Value-Based Deep RL</b><br/>DQN Family]
    L -->|Continuous| N[<b>Policy-Based Deep RL</b><br/>Continuous Control]
    L -->|Either| O[<b>Actor-Critic Methods</b><br/>Hybrid Approach]

    %% Discrete Action Space - DQN Improvements
    M --> M1{<b>DQN Enhancements?</b>}
    M1 -->|Basic| M2[<b>Standard DQN</b><br/>Experience Replay<br/>Target Network]
    M1 -->|Overestimation Fix| M3[<b>Double DQN</b><br/>Decouple action selection<br/>from evaluation]
    M1 -->|Architecture| M4[<b>Dueling DQN</b><br/>Separate V-s and A-s-a<br/>streams]
    M1 -->|Distributional| M5[<b>Distributional RL</b><br/>C51 / QR-DQN / IQN<br/>Learn full Q distribution]
    M1 -->|All Combined| M6[<b>Rainbow DQN</b><br/>Double + Dueling + PER<br/>+ Noisy + Multi-step + Distributional]

    %% Continuous Action Space - Policy Types
    N --> N0[<b>Continuous Control</b><br/>Approach Selection]
    N0 --> N1{<b>Policy Type?</b>}
    N1 -->|Deterministic| N2[<b>Deterministic Policy</b><br/>DDPG / TD3]
    N1 -->|Stochastic| N3[<b>Stochastic Policy</b><br/>PPO / TRPO / SAC]

    %% Deterministic Policy Refinements
    N2 --> N2A{<b>Stability?</b>}
    N2A -->|Basic| N2B[<b>DDPG</b><br/>Actor-Critic<br/>Deterministic Policy Gradient]
    N2A -->|Twin Critics| N2C[<b>TD3</b><br/>Twin Delayed DDPG<br/>Clipped Double Q-Learning<br/>Delayed Policy Updates]

    %% Stochastic Policy Refinements
    N3 --> N3A{<b>Optimization?</b>}
    N3A -->|Basic Gradient| N3B[<b>REINFORCE</b><br/>Vanilla Policy Gradient<br/>High Variance]
    N3A -->|Trust Region| N3C[<b>TRPO</b><br/>Constrained Optimization<br/>KL Divergence Constraint]
    N3A -->|Clipped| N3D[<b>PPO</b><br/>Clipped Surrogate Objective<br/>Simpler than TRPO]
    N3A -->|Max Entropy| N3E[<b>SAC</b><br/>Soft Actor-Critic<br/>Maximize Reward + Entropy<br/>Off-Policy]

    %% Actor-Critic Branch
    O --> O1{<b>Data Usage?</b>}
    O1 -->|On-Policy| O2[<b>On-Policy Actor-Critic</b><br/>A2C / A3C<br/>PPO]
    O1 -->|Off-Policy| O3[<b>Off-Policy Actor-Critic</b><br/>DDPG / TD3<br/>SAC]

    %% Model-Based Branch
    E --> S{<b>Sample Efficiency?</b>}
    S -->|Need Planning| T[<b>Model-Based RL</b><br/>Learn World Model]
    S -->|Model-Free OK| U[<b>Model-Free RL</b><br/>Most algorithms above]

    T --> T1{<b>Model Type?</b>}
    T1 -->|Explicit Model| T2[<b>Dyna-Q</b><br/>Learn model<br/>Plan with sweeps]
    T1 -->|Latent Model| T3[<b>World Models / Dreamer</b><br/>Learn in latent space<br/>Imagine trajectories]
    T1 -->|Implicit Model| T4[<b>MuZero</b><br/>Model predicts<br/>value/policy/reward only]

    %% Styling
    style A fill:#f9f,stroke:#333,stroke-width:2px,color:#000
    style B fill:#f9f,stroke:#333,stroke-width:2px,color:#000
    style K fill:#bbf,stroke:#333,stroke-width:2px,color:#000
    style M fill:#ffa,stroke:#333,stroke-width:2px,color:#000
    style N fill:#faa,stroke:#333,stroke-width:2px,color:#000
    style O fill:#bfb,stroke:#333,stroke-width:2px,color:#000
    style T fill:#fab,stroke:#333,stroke-width:2px,color:#000
    style M6 fill:#ff6,stroke:#333,stroke-width:2px,color:#000
    style N2C fill:#f88,stroke:#333,stroke-width:2px,color:#000
    style N3E fill:#8f8,stroke:#333,stroke-width:2px,color:#000
    style T4 fill:#f8f,stroke:#333,stroke-width:2px,color:#000
```

## 4. Key Distinctions

### By Learning Type
- **Value-Based:** Learn Q(s,a) or V(s), derive policy implicitly (e.g., DQN, Q-Learning)
- **Policy-Based:** Learn π(a|s) directly (e.g., REINFORCE, TRPO, PPO)
- **Actor-Critic:** Learn both Q/V and π (e.g., A2C, PPO, SAC, TD3)

### By Data Usage
- **On-Policy:** Must use data from current policy (e.g., A2C, PPO)
- **Off-Policy:** Can use old data, more sample-efficient (e.g., DQN, SAC, TD3)

### By Action Space
- **Discrete Actions:** Q-Learning, DQN, SARSA
- **Continuous Actions:** DDPG, TD3, SAC, PPO

### By Model Usage
- **Model-Free:** Don't learn environment dynamics (most algorithms)
- **Model-Based:** Learn world model, plan ahead (Dyna-Q, MuZero, Dreamer)

---

## 5. Algorithm Selection Guide

| Scenario | Recommended Algorithms |
|----------|------------------------|
| Discrete actions, small state space | Q-Learning, SARSA |
| Discrete actions, large state space | DQN, Rainbow, PPO |
| Continuous actions, continuous states | TD3, SAC, PPO |
| Need sample efficiency | Model-Based (MuZero, Dreamer), Off-Policy (SAC, TD3) |
| Stable training priority | PPO, SAC |
| Maximum performance | SAC (continuous), Rainbow (discrete) |
| Simple baseline | DQN (discrete), DDPG (continuous) |
| Partial observability | Recurrent policies (LSTM + any algorithm) |

