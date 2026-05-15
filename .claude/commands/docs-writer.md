Write a new documentation file for the component or topic given in `$ARGUMENTS`.

Determine the **doc type** from `$ARGUMENTS` using the table below, then read the relevant source files before writing. Output the file to `docs/<slug>.md` (or to the path implied by the argument if it includes a subdirectory). Always cross-link to related docs that already exist.

---

## Step 0 — Classify the request

| If `$ARGUMENTS` mentions… | Doc type |
|---|---|
| an algorithm (PPO, DDPG, TD3) | **Algorithm Overview** |
| a CLI command, pipeline, workflow, or end-to-end flow | **Workflow / Pipeline** |
| a subsystem: environment, reward, feature, trainer, config | **Component / Architecture** |
| setup, howto, guide, data, configuration reference | **Quick Reference / Guide** |

Read source files relevant to the topic (e.g., `src/trading_rl/trainers/`, `src/cli/`, `src/configs/`) before writing. Do not invent behaviour — derive every claim from what the code actually does.

---

## Template A — Algorithm Implementation Overview

Use for: PPO, DDPG, TD3, or any new RL algorithm added to the project.

```markdown
# <Algorithm> Implementation Overview

## Summary
- <algorithm family: on-policy / off-policy, actor-critic variant>
- <key distinguishing property vs siblings (e.g., twin critics, clipped surrogate)>
- <exploration strategy and replay / no-replay>

## Core Ideas
- **<Concept 1>**: one-sentence explanation
- **<Concept 2>**: one-sentence explanation
- **<Concept 3>**: one-sentence explanation

## Flow

```mermaid
flowchart TD
    subgraph Entry
        CLI["CLI: training.algorithm <ALG>"]
        CFG["Config -> ExperimentConfig"]
    end

    subgraph Data
        DATA["prepare_data -> DataFrame"]
        FEAT["create_features if enabled"]
    end

    subgraph Env
        ENV["create_environment ..."]
    end

    subgraph Models
        ACTOR["<actor construction>"]
        CRITIC["<critic construction>"]
    end

    subgraph Trainer
        INIT["<Trainer>.__init__ ..."]
        LOOP["<Trainer>.train — for each batch"]
        UPDATE["network updates"]
        LOG["callback.log_training_step -> MLflow"]
    end

    subgraph Evaluation
        EVAL["evaluate_agent — deterministic rollout"]
        PLOTS["reward / action plots"]
    end

    subgraph Tracking
        MLFLOW["MLflow metrics / params / artifacts"]
        CKPT["Checkpoint Save logs/"]
    end

    CLI --> CFG --> DATA --> FEAT --> ENV
    ENV --> ACTOR
    ENV --> CRITIC
    ACTOR --> INIT
    CRITIC --> INIT
    INIT --> LOOP --> UPDATE --> LOG --> MLFLOW
    LOOP --> CKPT
    LOOP --> EVAL --> PLOTS --> MLFLOW
```

## Optimization Detail

```mermaid
flowchart LR
    subgraph Critic_Update["Critic update (every step)"]
        ...
    end

    subgraph Actor_Update["Actor update (<delayed / every step>)"]
        ...
    end

    Critic_Update --- Actor_Update
```

## Math Summary

**Notation**

| Symbol | Meaning |
|---|---|
| $s, a, r, s', d$ | state, action, reward, next state, done |
| $\mathcal{B}$ | replay buffer distribution |
| $\gamma$ | discount factor |
| $\tau$ | soft-update coefficient |
| ... | ... |

**Key equations** (one block per update rule)

$$
\text{Bellman target: } y = r + \gamma(1-d)\,[\ldots]
$$

$$
\text{Critic loss: } L(\phi) = \mathbb{E}[(Q_\phi(s,a) - y)^2]
$$

$$
\text{Actor loss: } J(\theta) = -\mathbb{E}[Q_{\phi}(s,\mu_\theta(s))]
$$

**Soft target update**

$$
\bar\phi \leftarrow \tau\phi + (1-\tau)\bar\phi
$$

## Reference Configuration

Derived from `src/configs/scenarios/<group>/<name>/train.yaml`.

| Parameter | Value |
|---|---|
| Actor hidden dims | |
| Critic hidden dims | |
| Actor lr / Critic lr | |
| ... | |

## Components

- **CLI + configs**: `training.algorithm: <ALG>` selects this trainer and model builders.
- **Models**: `<actor_fn>` + `<critic_fn>` in `src/trading_rl/trainers/<alg>.py`.
- **Loss / optimizers**: `<LossClass>` with `<optimizer>` for actor and critic separately.
- **Collector / buffer**: `SyncDataCollector` + `<ReplayBuffer or on-policy batch>`.

## Training Loop

- <bullet: step 1 of per-step procedure>
- <bullet: step 2>
- <bullet: …>

## See Also

- [Experiment Workflow](./experiment_workflow.md)
- [PPO Implementation](./ppo_implementation_overview.md)
- [DDPG Implementation](./ddpg_implementation_overview.md)
- [TD3 Implementation](./td3_implementation_overview.md)
- [Data Guide](./data_guide.md)
- [Trading RL Package](../src/trading_rl/README.md)
```

---

## Template B — Workflow / Pipeline

Use for: CLI commands, end-to-end flows (training, evaluation, feature research), multi-step pipelines.

```markdown
# <Topic> Workflow

## Overview

One paragraph: what this workflow does, when to use it, and what it produces.

## Workflow Diagram

```mermaid
flowchart TD
    A["Entry point"] --> B["..."]
    B --> C["..."]
```

## Component Details

### 1. <First Stage>

- **Entry point**: `<function or CLI command>`
- **Location**: `src/<path>`
- **Steps**:
  1. <what happens>
  2. <what happens>

### 2. <Second Stage>

- **Entry point**: `<function>`
- **Location**: `src/<path>`
- **Steps**:
  1. ...

(Continue for each stage in the workflow.)

## Key Data Structures

| Type | Fields | Purpose |
|---|---|---|
| `<DataclassName>` | `field1`, `field2` | <what it represents> |
| ... | | |

## Usage Examples

### Basic

```bash
uv run python src/cli.py <command> --scenario <group/name>
```

### With overrides

```bash
uv run python src/cli.py <command> \
  --scenario <group/name> \
  --config-override training.max_steps=50000
```

### Common flags

| Flag | Purpose |
|---|---|
| `--scenario` | scenario shorthand or directory path |
| `--config-override` | OmegaConf dotlist override (repeatable) |
| `--verbose / -v` | debug-level logging |

## Configuration

Key fields in `train.yaml` / `evaluate.yaml` that control this workflow:

| Key | Default | Effect |
|---|---|---|
| `training.max_steps` | `100000` | total environment steps |
| ... | | |

## Output Structure

```
logs/
├── <experiment>/
│   ├── <name>_checkpoint.pt
│   └── ...
mlruns/
└── <experiment_id>/
    └── <run_id>/
        ├── params/
        ├── metrics/
        └── artifacts/
```

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `KeyError` on feature column | feature not in prepared data | check `features.yaml` against raw data columns |
| `NaN` loss | unscaled inputs or high lr | normalize data; lower actor/critic lr |
| ... | | |

## See Also

- [Data Guide](./data_guide.md)
- [PPO Implementation](./ppo_implementation_overview.md)
- [DDPG Implementation](./ddpg_implementation_overview.md)
- [TD3 Implementation](./td3_implementation_overview.md)
- [Training Pipeline](./training_pipeline.md)
```

---

## Template C — Component / Architecture

Use for: environments, reward functions, feature pipeline, trainer base, config system, any self-contained subsystem.

```markdown
# <Component> Architecture

## Overview

One paragraph: what this component does, where it sits in the pipeline, and why it is designed this way.

## Architecture Diagram

```mermaid
flowchart TD
    INPUT["Input"] --> COMP["<Component>"]
    COMP --> OUTPUT["Output"]

    COMP -.->|config| CFG["<config key>"]
    COMP -.->|extends| BASE["<base class>"]
```

## Subsystems

### <Subsystem 1>

- **Purpose**: <what it does>
- **Key classes / functions**: `<ClassName>` in `src/<path>`
- **Config keys**: `<config.section.key>`
- **Notes**: any non-obvious constraints or invariants

### <Subsystem 2>

(Continue for each logical subsystem.)

## Configuration Reference

All config keys that affect this component, sourced from the relevant dataclass in `src/trading_rl/config.py`:

| Key | Type | Default | Description |
|---|---|---|---|
| `env.backend` | `str` | `"gym_trading_env.discrete"` | environment backend |
| ... | | | |

## Extension Points

How to add a new implementation (e.g., new reward function, new environment backend, new feature):

1. <step: implement the interface>
2. <step: register in the registry / add to the factory>
3. <step: add a YAML example under `src/configs/`>
4. <step: add a smoke test under `tests/`>

## Known Constraints

- <constraint 1 — e.g., "causal normalizer must be reset at session boundaries">
- <constraint 2>

## See Also

- [Experiment Workflow](./experiment_workflow.md)
- [Feature Pipeline](./feature_pipeline.md)
- [Training Pipeline](./training_pipeline.md)
```

---

## Template D — Quick Reference / Guide

Use for: setup guides, data guides, how-to references, FAQ expansions, configuration references.

```markdown
# <Topic> Guide

## When to use this

One paragraph: the scenario this guide targets (e.g., "use this when you need to download proprietary OHLCV data from Google Drive before training").

## Prerequisites

- <prerequisite 1 (installed tool, env var, API key)>
- <prerequisite 2>

## Step-by-Step

### 1. <Step name>

```bash
# concrete command
uv run python scripts/<script>.py ...
```

Explanation of what happens and what to expect.

### 2. <Step name>

...

## Configuration Options

| Option / Env var | Purpose | Default |
|---|---|---|
| `SOME_ENV_VAR` | ... | none (required) |
| `--flag` | ... | ... |

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `FileNotFoundError` | data path not downloaded | run step 1 first |
| `AuthenticationError` | missing API key | set `SOME_API_KEY` env var |
| ... | | |

## See Also

- [Experiment Workflow](./experiment_workflow.md)
- [Training Pipeline](./training_pipeline.md)
```

---

## Formatting rules (apply to all templates)

- Title: `# <Name>` — sentence case, no trailing punctuation.
- Section order: match the template exactly; omit a section only if it has no content.
- Mermaid diagrams: `flowchart TD` for vertical flows, `flowchart LR` for side-by-side update loops.
- Tables: always include a header row; align with `|---|---|`.
- Code blocks: use `bash` for shell commands, `python` for Python snippets, `yaml` for config.
- Cross-links: always use relative Markdown links (`./other-doc.md`), not absolute URLs.
- No emojis. No trailing periods on section headings.
- After writing the file, add a bullet for it under `## Docs and References` in `README.md` if it is not already listed there.
