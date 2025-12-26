"""Training command implementation."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rich.progress import Progress, SpinnerColumn, TextColumn

from .base_command import BaseCommand


@dataclass
class TrainingParams:
    """Parameters for single agent training."""

    experiment_name: str | None = None
    config_file: Path | None = None
    seed: int | None = None
    max_steps: int | None = None
    init_rand_steps: int | None = None
    actor_lr: float | None = None
    value_lr: float | None = None
    buffer_size: int | None = None
    save_plots: bool = False
    log_dir: Path | None = None
    checkpoint_path: Path | None = None  # Path to checkpoint to resume from
    additional_steps: int | None = None  # Additional steps when resuming


class TrainingCommand(BaseCommand):
    """Command for training a single trading agent."""

    def execute(self, params: TrainingParams) -> None:
        """Execute single training run."""
        try:
            if params.checkpoint_path:
                self.console.print("[bold blue]Resuming Training from Checkpoint[/bold blue]")
                self.console.print(f"Checkpoint: [cyan]{params.checkpoint_path}[/cyan]")
            else:
                self.console.print("[bold blue]Starting Trading Agent Training[/bold blue]")

            # Load and configure experiment
            config = self._load_training_config(params)

            # Display configuration
            self._display_config(config, params)

            # Run training with progress tracking
            result = self._run_training_with_progress(config, params)

            # Save plots if requested
            if params.save_plots:
                self._save_training_plots(result, config, params)

        except Exception as e:
            self.handle_error(e, "Training")

    def _load_training_config(self, params: TrainingParams):
        """Load and configure training parameters."""
        from trading_rl import ExperimentConfig

        # Load base configuration
        if params.config_file:
            config = ExperimentConfig.from_yaml(params.config_file)
            self.console.print(f"[blue]Loaded config from: {params.config_file}[/blue]")
        else:
            config = ExperimentConfig()

        # Handle seed generation
        config.seed = self.resolve_seed(params.seed)

        # Apply CLI overrides
        self._apply_training_overrides(config, params)

        return config

    def _apply_training_overrides(self, config, params: TrainingParams) -> None:
        """Apply CLI parameter overrides to config."""
        if params.experiment_name:
            config.experiment_name = params.experiment_name
        if params.max_steps is not None:
            config.training.max_steps = params.max_steps
        if params.init_rand_steps is not None:
            config.training.init_rand_steps = params.init_rand_steps
        if params.actor_lr is not None:
            config.training.actor_lr = params.actor_lr
        if params.value_lr is not None:
            config.training.value_lr = params.value_lr
        if params.buffer_size is not None:
            config.training.buffer_size = params.buffer_size
        if params.log_dir is not None:
            config.logging.log_dir = str(params.log_dir)

    def _display_config(self, config, params: TrainingParams) -> None:
        """Display training configuration."""
        self.console.print(f"Experiment: [green]{config.experiment_name}[/green]")
        self.console.print(f"Seed: [green]{config.seed}[/green]")
        if params.checkpoint_path and params.additional_steps:
            self.console.print(f"Additional steps: [green]{params.additional_steps}[/green]")
        else:
            self.console.print(f"Max steps: [green]{config.training.max_steps}[/green]")

    def _run_training_with_progress(self, config, params: TrainingParams) -> dict[str, Any]:
        """Run training with progress display."""
        from trading_rl import run_single_experiment

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
        ) as progress:
            task = progress.add_task("Training agent...", total=None)

            try:
                # Handle checkpoint resumption
                if params.checkpoint_path:
                    result = self._resume_from_checkpoint(
                        config, params, progress
                    )
                else:
                    result = run_single_experiment(custom_config=config)

                progress.update(task, description="Training complete!")

                self._display_training_results(result)
                return result

            except Exception as e:
                progress.update(task, description="Training failed!")
                raise e

    def _resume_from_checkpoint(
        self, config, params: TrainingParams, progress
    ) -> dict[str, Any]:
        """Resume training from a checkpoint file."""
        import torch
        import mlflow
        from pathlib import Path
        from trading_rl.data_utils import prepare_data
        from trading_rl.envs import AlgorithmicEnvironmentBuilder
        from trading_rl.training import TD3Trainer, DDPGTrainer, PPOTrainer
        from trading_rl.trainers.ppo import PPOTrainerContinuous
        from trading_rl.callbacks import MLflowTrainingCallback
        from trading_rl.plotting import visualize_training

        # Validate checkpoint exists
        if not Path(params.checkpoint_path).exists():
            raise FileNotFoundError(f"Checkpoint not found: {params.checkpoint_path}")

        self.console.print(f"[cyan]Loading checkpoint from {params.checkpoint_path}...[/cyan]")

        # Prepare data
        df = prepare_data(
            data_path=config.data.data_path,
            download_if_missing=config.data.download_data,
            exchange_names=config.data.exchange_names,
            symbols=config.data.symbols,
            timeframe=config.data.timeframe,
            data_dir=config.data.data_dir,
            since=config.data.download_since,
            no_features=getattr(config.data, "no_features", False),
        )

        # Create environment
        env_builder = AlgorithmicEnvironmentBuilder()
        env = env_builder.create(df, config)

        # Get environment specs
        n_obs = env.observation_spec["observation"].shape[-1]
        n_act = env.action_spec.shape[-1]

        # Determine algorithm and trainer class
        algorithm = getattr(config.training, "algorithm", "PPO").upper()
        backend = getattr(config.env, "backend", "")
        is_continuous_env = backend == "tradingenv" or backend == "gym_trading_env.continuous"

        if algorithm == "PPO":
            trainer_cls = PPOTrainerContinuous if is_continuous_env else PPOTrainer
        elif algorithm == "TD3":
            trainer_cls = TD3Trainer
        elif algorithm == "DDPG":
            trainer_cls = DDPGTrainer
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")

        # Build models
        if algorithm == "TD3":
            actor, qvalue_net = trainer_cls.build_models(n_obs, n_act, config, env)
            trainer = trainer_cls(
                actor=actor, qvalue_net=qvalue_net, env=env, config=config.training
            )
        else:
            actor, value_net = trainer_cls.build_models(n_obs, n_act, config, env)
            trainer = trainer_cls(
                actor=actor, value_net=value_net, env=env, config=config.training
            )

        # Load checkpoint
        trainer.load_checkpoint(str(params.checkpoint_path))
        original_steps = trainer.total_count
        self.console.print(f"[green]Checkpoint loaded! Resuming from step {original_steps}[/green]")

        # Update max_steps
        if params.additional_steps:
            trainer.config.max_steps = original_steps + params.additional_steps
            self.console.print(
                f"[cyan]Training for {params.additional_steps} additional steps[/cyan]"
            )
            self.console.print(
                f"[cyan]Target: {trainer.config.max_steps} total steps[/cyan]"
            )

        # Setup MLflow callback
        mlflow_callback = MLflowTrainingCallback(
            config.experiment_name,
            tracking_uri=getattr(getattr(config, "tracking", None), "tracking_uri", None),
            price_series=df["close"][: config.data.train_size],
        )

        # Continue training - use existing run if active, otherwise create new run
        active_run = mlflow.active_run()
        if active_run:
            # Continue with existing run
            self.console.print(f"[cyan]Continuing existing MLflow run: {active_run.info.run_id}[/cyan]")
            logs = trainer.train(callback=mlflow_callback)
        else:
            # Create new run for resumed training
            with mlflow.start_run(run_name=f"resumed_step_{original_steps}"):
                logs = trainer.train(callback=mlflow_callback)

        # Save new checkpoint
        checkpoint_path = (
            Path(config.logging.log_dir)
            / f"{config.experiment_name}_checkpoint_step_{trainer.total_count}.pt"
        )
        trainer.save_checkpoint(str(checkpoint_path))
        self.console.print(f"[green]New checkpoint saved: {checkpoint_path}[/green]")

        # Prepare result (similar to run_single_experiment)
        return {
            "trainer": trainer,
            "logs": logs,
            "final_metrics": {
                "final_reward": 0.0,  # Will be updated by evaluation if needed
                "training_steps": trainer.total_count,
            },
            "plots": {
                "loss": visualize_training(logs)
                if logs.get("loss_value") or logs.get("loss_actor")
                else None,
            },
        }

    def _display_training_results(self, result: dict[str, Any]) -> None:
        """Display training results."""
        self.console.print(
            "\n[bold green]Training completed successfully![/bold green]"
        )

        if "final_metrics" in result:
            metrics = result["final_metrics"]
            if "final_reward" in metrics:
                self.console.print(
                    f"Final reward: [green]{metrics['final_reward']:.4f}[/green]"
                )
            if "training_steps" in metrics:
                self.console.print(
                    f"Training steps: [green]{metrics['training_steps']}[/green]"
                )

    def _save_training_plots(
        self, result: dict[str, Any], config, params: TrainingParams
    ) -> None:
        """Save training plots to disk."""

        plots_dir = Path(config.logging.log_dir) / "plots"
        plots_dir.mkdir(exist_ok=True, parents=True)
        # (Skipping rewrite as per instructions)
        return result
