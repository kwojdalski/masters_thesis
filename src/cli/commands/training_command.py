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
    scenario: str | None = None
    config_overrides: list[str] | None = None
    seed: int | None = None
    max_steps: int | None = None
    no_features: bool = False
    checkpoint_path: Path | None = None  # Path to checkpoint to resume from
    additional_steps: int | None = None  # Additional steps when resuming
    from_checkpoint: Path | None = None  # Path to checkpoint alias
    from_last_checkpoint: bool = False  # Resume from most recent checkpoint
    mlflow_run_id: str | None = None  # Resume MLflow run by ID


class TrainingCommand(BaseCommand):
    """Command for training a single trading agent."""

    def execute(self, params: TrainingParams) -> None:
        """Execute single training run."""
        try:
            if params.from_checkpoint and params.from_last_checkpoint:
                raise ValueError(
                    "Use only one of --from-checkpoint or --from-last-checkpoint."
                )

            # Load and configure experiment
            config = self._load_training_config(params)
            self._resolve_checkpoint_path(config, params)

            if params.checkpoint_path:
                self.console.print(
                    "[bold blue]Resuming Training from Checkpoint[/bold blue]"
                )
                self.console.print(f"Checkpoint: [cyan]{params.checkpoint_path}[/cyan]")
            else:
                self.console.print(
                    "[bold blue]Starting Trading Agent Training[/bold blue]"
                )

            # Display configuration
            self._display_config(config, params)

            # Run training with progress tracking
            result = self._run_training_with_progress(config, params)

            # Save plots if requested
            if config.logging.save_plots:
                self._save_training_plots(result, config, params)

        except Exception as e:
            self.handle_error(e, "Training")

    def _load_training_config(self, params: TrainingParams):
        """Load and configure training parameters."""
        from trading_rl import ExperimentConfig

        # Load base configuration
        if params.config_file and params.scenario:
            raise ValueError("Cannot specify both --config and --scenario.")

        if params.config_file:
            config = ExperimentConfig.from_yaml(
                params.config_file, overrides=params.config_overrides
            )
            self.console.print(f"[blue]Loaded config from: {params.config_file}[/blue]")
        elif params.scenario:
            config_file = self._resolve_scenario_config_path(params.scenario)
            config = ExperimentConfig.from_yaml(
                config_file, overrides=params.config_overrides
            )
            self.console.print(
                f"[blue]Loaded config from scenario: {params.scenario} -> {config_file}[/blue]"
            )
        else:
            if params.config_overrides:
                raise ValueError("--config-override requires --config or --scenario.")
            config = ExperimentConfig()

        # Apply CLI overrides
        self._apply_training_overrides(config, params)

        # Handle seed generation
        config.seed = self.resolve_seed(config.seed)

        return config

    def _apply_training_overrides(self, config, params: TrainingParams) -> None:
        """Apply CLI parameter overrides to config."""
        if params.experiment_name:
            config.experiment_name = params.experiment_name
        if params.seed is not None:
            config.seed = params.seed
        if params.max_steps is not None:
            config.training.max_steps = params.max_steps
        if params.no_features:
            config.data.no_features = True

    def _resolve_scenario_config_path(self, scenario: str) -> Path:
        """Resolve scenario name to config file path."""
        candidate_path = Path(scenario)

        if candidate_path.is_dir():
            candidate_path = candidate_path / "config.yaml"

        search_paths = [
            candidate_path,
            Path("src/configs/scenarios") / scenario,
            Path("src/configs/scenarios") / f"{scenario}.yaml",
        ]

        for path in search_paths:
            if path.exists():
                return path.resolve()

        raise ValueError(
            f"Scenario '{scenario}' not found. Provide a valid path or name in src/configs/scenarios."
        )

    def _display_config(self, config, params: TrainingParams) -> None:
        """Display training configuration."""
        self.console.print(f"Experiment: [green]{config.experiment_name}[/green]")
        self.console.print(f"Seed: [green]{config.seed}[/green]")
        if params.checkpoint_path and params.additional_steps:
            self.console.print(
                f"Additional steps: [green]{params.additional_steps}[/green]"
            )
        else:
            self.console.print(f"Max steps: [green]{config.training.max_steps}[/green]")

    def _resolve_checkpoint_path(self, config, params: TrainingParams) -> None:
        """Resolve checkpoint path from aliases or latest checkpoint option."""
        if params.checkpoint_path:
            return
        if params.from_checkpoint:
            params.checkpoint_path = params.from_checkpoint
            return
        if not params.from_last_checkpoint:
            return

        log_dir = Path(config.logging.log_dir)
        pattern = f"{config.experiment_name}_checkpoint*.pt"
        matches = list(log_dir.rglob(pattern))
        if not matches:
            raise FileNotFoundError(
                f"No checkpoints found for {config.experiment_name} in {log_dir}"
            )
        latest = max(matches, key=lambda p: p.stat().st_mtime)
        params.checkpoint_path = latest

    def _run_training_with_progress(
        self, config, params: TrainingParams
    ) -> dict[str, Any]:
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
                    result = self._resume_from_checkpoint(config, params, progress)
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
        from pathlib import Path

        import mlflow

        from trading_rl.callbacks import MLflowTrainingCallback
        from trading_rl.plotting import visualize_training
        from trading_rl.train_trading_agent import (
            build_training_context,
            setup_mlflow_experiment,
        )

        # Validate checkpoint exists
        if not Path(params.checkpoint_path).exists():
            raise FileNotFoundError(f"Checkpoint not found: {params.checkpoint_path}")

        self.console.print(
            f"[cyan]Loading checkpoint from {params.checkpoint_path}...[/cyan]"
        )

        context = build_training_context(config, create_mlflow_callback=False)
        logger = context["logger"]
        df = context["df"]
        trainer = context["trainer"]
        algorithm = context["algorithm"]

        logger.info("Resuming training from checkpoint")
        logger.info(f"Checkpoint: {params.checkpoint_path}")

        # Load checkpoint
        logger.info(f"Loading checkpoint from {params.checkpoint_path}...")
        trainer.load_checkpoint(str(params.checkpoint_path))
        original_steps = trainer.total_count
        # if trainer.total_episodes == 0:
        # raise ValueError(
        #     f"Checkpoint has total_episodes={trainer.total_episodes}; episode metrics will reset on resume."
        # )
        logger.info(f"Checkpoint loaded! Resuming from step {original_steps}")
        self.console.print(
            f"[green]Checkpoint loaded! Resuming from step {original_steps}[/green]"
        )
        tracking_uri = getattr(trainer, "mlflow_tracking_uri", None)
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

        resume_experiment_name = getattr(trainer, "mlflow_experiment_name", None)
        if not resume_experiment_name:
            experiment_id = getattr(trainer, "mlflow_experiment_id", None)
            if experiment_id:
                experiment = mlflow.get_experiment(experiment_id)
                resume_experiment_name = experiment.name if experiment else None

        if resume_experiment_name:
            if resume_experiment_name != config.experiment_name:
                logger.info(
                    "Resuming MLflow experiment name from checkpoint: %s",
                    resume_experiment_name,
                )
            config.experiment_name = resume_experiment_name
            if hasattr(trainer, "checkpoint_prefix"):
                trainer.checkpoint_prefix = resume_experiment_name
        setup_mlflow_experiment(config, config.experiment_name)
        mlflow_callback = None

        # Update max_steps
        if params.additional_steps:
            trainer.config.max_steps = original_steps + params.additional_steps
            logger.info(f"Training for {params.additional_steps} additional steps")
            logger.info(f"Target: {trainer.config.max_steps} total steps")
            self.console.print(
                f"[cyan]Training for {params.additional_steps} additional steps[/cyan]"
            )
            self.console.print(
                f"[cyan]Target: {trainer.config.max_steps} total steps[/cyan]"
            )

        # Continue training - use existing run if active, otherwise create new run
        logger.info("Starting training...")
        resume_run_id = None

        def _episode_count_value() -> int:
            logged = (
                trainer.logs.get("episode_log_count")
                if hasattr(trainer, "logs")
                else None
            )
            if logged:
                return int(logged[-1])
            total = trainer.total_episodes
            if hasattr(total, "item"):
                return int(total.item())
            return int(total)

        def _log_resume_artifacts() -> None:
            """Log config/data artifacts during resume when a run is active."""
            if not mlflow.active_run():
                logger.warning("No active MLflow run - skipping resume artifacts")
                return
            MLflowTrainingCallback.log_parameter_faq_artifact()
            MLflowTrainingCallback.log_training_parameters(config)
            MLflowTrainingCallback.log_config_artifact(config)
            MLflowTrainingCallback.log_data_overview(df, config)

        active_run = mlflow.active_run()
        if active_run:
            # Continue with existing run
            logger.info(f"Continuing existing MLflow run: {active_run.info.run_id}")
            self.console.print(
                f"[cyan]Continuing existing MLflow run: {active_run.info.run_id}[/cyan]"
            )
            resume_run_id = active_run.info.run_id
            mlflow_callback = MLflowTrainingCallback(
                config.experiment_name,
                tracking_uri=getattr(
                    getattr(config, "tracking", None), "tracking_uri", None
                ),
                price_series=df["close"][: config.data.train_size],
                start_run=False,
            )
            mlflow_callback._episode_count = _episode_count_value()
            _log_resume_artifacts()
            logs = trainer.train(callback=mlflow_callback)
            checkpoint_path = (
                Path(config.logging.log_dir)
                / f"{config.experiment_name}_checkpoint_step_{trainer.total_count}.pt"
            )
            trainer.save_checkpoint(str(checkpoint_path))
            logger.info(f"Checkpoint saved to: {checkpoint_path}")
            self.console.print(
                f"[green]New checkpoint saved: {checkpoint_path}[/green]"
            )
        elif getattr(trainer, "mlflow_run_id", None):
            run_id = trainer.mlflow_run_id
            logger.info(f"Resuming MLflow run: {run_id}")
            self.console.print(f"[cyan]Resuming MLflow run: {run_id}[/cyan]")
            resume_run_id = run_id
            with mlflow.start_run(run_id=run_id):
                mlflow_callback = MLflowTrainingCallback(
                    config.experiment_name,
                    tracking_uri=tracking_uri
                    or getattr(getattr(config, "tracking", None), "tracking_uri", None),
                    price_series=df["close"][: config.data.train_size],
                    start_run=False,
                )
                mlflow_callback._episode_count = _episode_count_value()
                _log_resume_artifacts()
                logs = trainer.train(callback=mlflow_callback)
                checkpoint_path = (
                    Path(config.logging.log_dir)
                    / f"{config.experiment_name}_checkpoint_step_{trainer.total_count}.pt"
                )
                trainer.save_checkpoint(str(checkpoint_path))
                logger.info(f"Checkpoint saved to: {checkpoint_path}")
                self.console.print(
                    f"[green]New checkpoint saved: {checkpoint_path}[/green]"
                )
        elif params.mlflow_run_id:
            run_id = params.mlflow_run_id
            logger.info(f"Resuming MLflow run (CLI): {run_id}")
            self.console.print(f"[cyan]Resuming MLflow run (CLI): {run_id}[/cyan]")
            resume_run_id = run_id
            with mlflow.start_run(run_id=run_id):
                mlflow_callback = MLflowTrainingCallback(
                    config.experiment_name,
                    tracking_uri=getattr(
                        getattr(config, "tracking", None), "tracking_uri", None
                    ),
                    price_series=df["close"][: config.data.train_size],
                    start_run=False,
                )
                mlflow_callback._episode_count = _episode_count_value()
                _log_resume_artifacts()
                logs = trainer.train(callback=mlflow_callback)
                checkpoint_path = (
                    Path(config.logging.log_dir)
                    / f"{config.experiment_name}_checkpoint_step_{trainer.total_count}.pt"
                )
                trainer.save_checkpoint(str(checkpoint_path))
                logger.info(f"Checkpoint saved to: {checkpoint_path}")
                self.console.print(
                    f"[green]New checkpoint saved: {checkpoint_path}[/green]"
                )
        else:
            # Create new run for resumed training
            logger.info(
                f"Creating new MLflow run for resumed training from step {original_steps}"
            )
            with mlflow.start_run(run_name=f"resumed_step_{original_steps}") as run:
                resume_run_id = run.info.run_id
                mlflow_callback = MLflowTrainingCallback(
                    config.experiment_name,
                    tracking_uri=getattr(
                        getattr(config, "tracking", None), "tracking_uri", None
                    ),
                    price_series=df["close"][: config.data.train_size],
                    start_run=False,
                )
                mlflow_callback._episode_count = _episode_count_value()
                _log_resume_artifacts()
                logs = trainer.train(callback=mlflow_callback)
                checkpoint_path = (
                    Path(config.logging.log_dir)
                    / f"{config.experiment_name}_checkpoint_step_{trainer.total_count}.pt"
                )
                trainer.save_checkpoint(str(checkpoint_path))
                logger.info(f"Checkpoint saved to: {checkpoint_path}")
                self.console.print(
                    f"[green]New checkpoint saved: {checkpoint_path}[/green]"
                )

        def _evaluate_and_log() -> dict[str, Any]:
            """Evaluate and log results under the active MLflow run."""
            logger.info("Evaluating agent...")
            self.console.print("[cyan]Evaluating agent...[/cyan]")
            eval_max_steps = min(
                config.training.eval_steps, len(df) - 1, config.data.train_size - 1
            )
            logger.debug(f"Evaluation max_steps: {eval_max_steps}")

            (
                reward_plot,
                action_plot,
                action_probs_plot,
                final_reward,
                last_positions,
            ) = trainer.evaluate(
                df[: config.data.train_size],
                max_steps=eval_max_steps,
                config=config,
                algorithm=algorithm,
            )

            # Log evaluation plots to MLflow
            MLflowTrainingCallback.log_evaluation_plots(
                reward_plot=reward_plot,
                action_plot=action_plot,
                action_probs_plot=action_probs_plot,
                logs=logs,
            )

            # Detect backend type for proper metric naming
            is_portfolio_backend = config.env.backend == "tradingenv"

            # Prepare comprehensive metrics (matching run_single_experiment)
            final_metrics = {
                "final_reward": final_reward,
                "training_steps": trainer.total_count,
            (
                "portfolio_weights"
                if is_portfolio_backend
                else "last_position_per_episode"
            ): last_positions,
                "data_start_date": str(df.index[0]) if not df.empty else "unknown",
                "data_end_date": str(df.index[-1]) if not df.empty else "unknown",
                "data_size": len(df),
                "train_size": config.data.train_size,
                "trading_fees": config.env.trading_fees,
                "experiment_name": config.experiment_name,
                "resumed_from_step": original_steps,
            }

            # Log final metrics to MLflow
            MLflowTrainingCallback.log_final_metrics(
                logs, final_metrics, mlflow_callback
            )

            # Final logging (like run_single_experiment)
            logger.info("Training complete!")
            logger.info(f"Final reward: {final_reward:.4f}")
            logger.info(f"Total training steps: {trainer.total_count}")

            # Prepare result (matching run_single_experiment)
            return {
                "trainer": trainer,
                "logs": logs,
                "final_metrics": final_metrics,
                "plots": {
                    "loss": visualize_training(logs)
                    if logs.get("loss_value") or logs.get("loss_actor")
                    else None,
                    "reward": reward_plot,
                    "action": action_plot,
                },
            }

        if not mlflow.active_run() and resume_run_id:
            with mlflow.start_run(run_id=resume_run_id):
                return _evaluate_and_log()

        return _evaluate_and_log()

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
