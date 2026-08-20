# MCP Server Workflow

## Overview

`src/mcp_server/` exposes the same command classes used by `src/cli.py` (see `src/cli/commands/`) as [MCP](https://modelcontextprotocol.io) tools, so an agent (Claude Code, Claude Desktop, or any other MCP client) can drive training, evaluation, and data-pipeline commands directly instead of shelling out to the CLI. It is a thin adapter layer: no command logic is duplicated, only translated into MCP's tool-call/result shape. Long-running commands (training, evaluation, multi-trial experiments, feature research, results collection) run on background threads so a single tool call doesn't block past an MCP client's call timeout; the caller polls a job id for status and logs.

## Workflow Diagram

``` mermaid
flowchart TD
    CLIENT["MCP client<br/>(agent)"] -->|tool call| TOOL["server.py tool function"]

    TOOL --> SYNC{"long-running?"}
    SYNC -->|no: checkpoints, scenarios,<br/>artifacts, experiments,<br/>validate*, peek, generate_data| RUNSYNC["_run_sync()"]
    SYNC -->|yes: train, evaluate,<br/>experiment, feature_research,<br/>collect_results| JOBSTART["JOBS.start(kind, run)"]

    RUNSYNC --> CAPTURE1["new_capture_console()"]
    JOBSTART --> THREAD["daemon thread"]
    THREAD --> CAPTURE2["per-job Console + StringIO"]

    CAPTURE1 --> RUNCMD["run_command()"]
    CAPTURE2 --> RUNCMD

    RUNCMD --> CMD["cli/commands/*.py<br/>Command.execute(params)"]
    CMD -->|typer.Exit / exception| NORMALIZE["normalize to<br/>{ok, output, result/error}"]
    CMD -->|return| NORMALIZE

    NORMALIZE -->|sync path| RESULT["tool result (dict)"]
    NORMALIZE -->|job path| JOBSTATE["Job.outcome + Job.status"]

    JOBSTATE -.->|job_status / job_logs| CLIENT
    RESULT --> CLIENT
```

## Component Details

### 1. Tool Layer

- **Entry point**: `mcp_server` (an `mcp.server.mcpserver.MCPServer` instance) with one `@mcp_server.tool()` function per CLI command class
- **Location**: `src/mcp_server/server.py`
- **Steps**:
  1. Accept flat keyword arguments (MCP tools need a JSON-schema-able signature, so each tool re-declares the fields of the corresponding `*Params` dataclass rather than taking the dataclass itself)
  2. Build the `*Params` dataclass and, for delete-capable tools, call `require_force_for_delete` first
  3. Dispatch to either `_run_sync` (synchronous tools) or `JOBS.start` (background tools)

### 2. Command Execution Adapter

- **Entry point**: `run_command(console, buffer, build, *args, **kwargs)`
- **Location**: `src/mcp_server/support.py`
- **Steps**:
  1. Construct the command with `build(console)`, where `console` renders to an in-memory `io.StringIO` instead of a terminal (`new_capture_console()`)
  2. Call `command.execute(*args, **kwargs)`
  3. Normalize the outcome: CLI commands in this repo don't return structured results, they print to a `rich.Console` and use `typer.Exit` / `typer.BadParameter` for control flow (see `BaseCommand.handle_error`) — `run_command` catches `typer.Exit` and treats a zero exit code as success, and catches any other exception as a tool-level failure, always returning `{"ok": bool, "output": str, ...}`
- **Also in this module**: `require_force_for_delete`, which rejects a delete request that isn't `force=True` (or `dry_run=True`). CLI delete flows fall back to an interactive `typer.confirm()` prompt when `force` is unset; an MCP tool call has no stdin to answer it, so without this guard the call would hang.

### 3. Background Job Manager

- **Entry point**: `JobManager.start(kind, run)` on the module-level `JOBS` singleton
- **Location**: `src/mcp_server/jobs.py`
- **Steps**:
  1. Allocate a short job id, a dedicated `Console`/`StringIO` pair, and a `Job` record (`status=PENDING`)
  2. Launch a daemon thread that sets `status=RUNNING`, calls `run(job.console, job.buffer)` (a closure built by the `*_start` tool that calls `run_command`), and sets `status` to `COMPLETED` or `FAILED` based on the outcome's `ok` field
  3. `job_status(job_id)` returns `Job.summary()`; `job_logs(job_id)` returns the buffer's current contents, which are readable while the job is still running since the same console keeps writing into it; `job_list()` returns all jobs, newest first

## Key Data Structures

| Type | Fields | Purpose |
|---|---|---|
| `run_command` result (`dict`) | `ok`, `output`, and one of `result` / `error` / `exit_code` | Normalized outcome of any wrapped CLI command |
| `Job` (`jobs.py`) | `id`, `kind`, `status`, `created_at`, `started_at`, `finished_at`, `outcome`, `error` | Background job state, polled via `job_status`/`job_logs` |
| `JobStatus` (`StrEnum`) | `PENDING`, `RUNNING`, `COMPLETED`, `FAILED` | Job lifecycle state |

## Usage Examples

### Run the server directly (stdio transport)

```bash
uv run trading-rl-mcp
```

### Register it with an MCP client

```json
{
  "mcpServers": {
    "trading-rl": {
      "command": "uv",
      "args": ["run", "trading-rl-mcp"],
      "cwd": "/path/to/masters_thesis"
    }
  }
}
```

### Start a training run and poll it

A client calls, in sequence:

```text
train_start(scenario="sine_wave/ppo_no_trend", config_overrides=["training.max_steps=10000"])
  -> {"job_id": "a1b2c3d4e5f6", "status": "started", "kind": "train"}

job_status(job_id="a1b2c3d4e5f6")
  -> {"job_id": "...", "status": "running", ...}

job_logs(job_id="a1b2c3d4e5f6")
  -> {"job_id": "...", "status": "running", "logs": "Starting Trading Agent Training\n..."}
```

### Available tools

| Tool | Maps to | Execution |
|---|---|---|
| `train_start` | `TrainingCommand` | background job |
| `evaluate_start` | `EvaluateCommand` | background job |
| `experiment_start` | `ExperimentCommand` (multi-trial) | background job |
| `feature_research_start` | `FeatureResearchCommand` | background job |
| `collect_results_start` | `CollectResultsCommand` | background job |
| `job_status`, `job_logs`, `job_list` | `JobManager` | synchronous |
| `checkpoints` | `CheckpointsCommand` | synchronous |
| `scenarios` | `ScenariosCommand` | synchronous |
| `artifacts` | `ArtifactsCommand` | synchronous |
| `experiments` | `ExperimentsCommand` (MLflow experiment housekeeping) | synchronous |
| `validate_config` | `ValidationCommand` | synchronous |
| `validate_data` | `ValidateDataCommand` | synchronous |
| `peek` | `PeekCommand` | synchronous |
| `generate_data` | `DataGeneratorCommand` | synchronous |

`dashboard` (`DashboardCommand`) is not exposed — see Known Constraints below.

## Configuration

| Key | Location | Effect |
|---|---|---|
| `[project.scripts] trading-rl-mcp` | `pyproject.toml` | console-script entry point, `mcp_server.server:main` |
| `mcp` dependency | `pyproject.toml` `[project.dependencies]` | official MCP Python SDK (`modelcontextprotocol/python-sdk`) |
| `logs/mcp_server.log` | written by `main()` via `logger.setup_logging(console_output=False, log_file=...)` | server-side loguru output; kept off stdout (see Known Constraints) |

## Output Structure

Background job output isn't written to disk by the job manager itself — it's held in memory (`Job.buffer`) for the life of the server process. The underlying CLI commands still write their usual artifacts (checkpoints under `logs/`, MLflow runs, evaluation reports) exactly as they would from `cli.py`; see [Training Pipeline](./training_pipeline.md) and [Experiment Workflow](./experiment_workflow.md) for those locations.

## Known Constraints

- **stdio transport reserves stdout for protocol frames.** Any stray write to stdout (an errant `print()`, a subprocess with inherited stdio) corrupts the MCP session. Every command is invoked with its own `rich.Console` pointed at an in-memory buffer, and server-side logging is redirected to `logs/mcp_server.log` instead of the console, but this cannot guarantee a third-party dependency never prints directly to stdout.
- **`dashboard` is intentionally not exposed.** `DashboardCommand._launch_mlflow_ui` runs `subprocess.run([...])` with inherited stdio, and `mlflow ui` is a long-lived foreground server — both properties are a bad fit for a stdio-transport MCP tool. Run `uv run python src/cli.py dashboard` directly instead.
- **`src/mcp_server/`, not `src/mcp/`.** This project's editable install (`where = ["src"]`) makes every top-level directory under `src/` directly importable; naming the package `mcp` would shadow the installed `mcp` SDK package of the same name.
- **Interactive prompts are refused, not silenced.** `TrainingParams.interactive` is hardcoded to `False` in `train_start`, and delete tools raise before reaching `typer.confirm()` unless `force=True` is passed explicitly — an MCP tool call has no stdin to answer a prompt with.
- **Job state is process-local and in-memory.** Restarting the MCP server loses all job history; there's no persistence layer.

## See Also

- [CLI Overview](./cli/overview.md)
- [CLI Workflow Commands](./cli/workflow_commands.md)
- [CLI MLflow Management](./cli/mlflow_management.md)
- [Experiment Workflow](./experiment_workflow.md)
- [Training Pipeline](./training_pipeline.md)
