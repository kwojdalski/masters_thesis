from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd

import trading_rl.pipeline.experiment_runner as runner
from trading_rl.data_utils import PreparedDataset
from trading_rl.pipeline.training import ExperimentRuntime, TrainingBundle


class _Logger:
    def __init__(self) -> None:
        self.infos: list[tuple[Any, ...]] = []
        self.warnings: list[tuple[Any, ...]] = []

    def info(self, *args: Any, **_kwargs: Any) -> None:
        self.infos.append(args)

    def warning(self, *args: Any, **_kwargs: Any) -> None:
        self.warnings.append(args)


class _Trainer:
    def __init__(self, *, interrupt_training: bool = False) -> None:
        self.interrupt_training = interrupt_training
        self.logs = {"training_duration_s": [0.25]}
        self.total_count = 12
        self.total_episodes = 3
        self.callbacks: list[Any] = []

    def train(self, callback: Any = None) -> dict[str, Any]:
        self.callbacks.append(callback)
        if self.interrupt_training:
            raise KeyboardInterrupt
        return dict(self.logs)


def _config(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        profiling=SimpleNamespace(level=0),
        logging=SimpleNamespace(log_dir=str(tmp_path)),
    )


def _dataset() -> PreparedDataset:
    df = pd.DataFrame({"close": [100.0, 101.0, 102.0]})
    return PreparedDataset(
        train_df=df,
        val_df=df,
        test_df=df,
        feature_columns=["close"],
        price_column="close",
        raw_columns=["close"],
        feature_pipeline_state={"feature_close": {"mean": 1.0}},
    )


def _runtime(
    trainer: _Trainer,
    *,
    callback: Any = None,
    effective_name: str = "experiment",
) -> ExperimentRuntime:
    return ExperimentRuntime(
        logger=_Logger(),
        effective_experiment_name=effective_name,
        prepared_dataset=_dataset(),
        training_bundle=TrainingBundle(
            train_env=object(),
            trainer=trainer,
            mlflow_callback=callback,
            algorithm="PPO",
            n_obs=1,
            n_act=2,
        ),
    )


def _patch_finalization(monkeypatch, *, evaluate_raises: bool = False) -> dict[str, Any]:
    calls: dict[str, Any] = {"metrics": [], "checkpoints": [], "saved_json": 0}
    monkeypatch.setattr(runner, "_configure_periodic_hooks", lambda **_kwargs: None)
    monkeypatch.setattr(
        runner,
        "save_final_checkpoint",
        lambda **kwargs: calls["checkpoints"].append(kwargs) or Path("final.pt"),
    )

    def fake_evaluate_all_splits(**_kwargs: Any) -> dict[str, dict[str, Any]]:
        if evaluate_raises:
            raise KeyboardInterrupt
        return {"test": {"split": "test", "final_reward": 1.25}}

    monkeypatch.setattr(runner, "evaluate_all_splits", fake_evaluate_all_splits)
    monkeypatch.setattr(
        runner,
        "resolve_primary_split_result",
        lambda split_results: ("test", 1.25, [0.0, 1.0], {"total_return": 0.1}),
    )
    monkeypatch.setattr(runner, "run_primary_split_explainability", lambda **_kwargs: None)

    def fake_build_final_metrics(**kwargs: Any) -> dict[str, Any]:
        calls["metrics"].append(kwargs)
        return {
            "interrupted": kwargs["interrupted"],
            "experiment_name": kwargs["effective_experiment_name"],
            "split_results": kwargs["split_results"],
            "final_reward": kwargs["final_reward"],
        }

    monkeypatch.setattr(runner, "build_final_metrics", fake_build_final_metrics)
    monkeypatch.setattr(runner, "log_final_metrics", lambda **_kwargs: None)
    monkeypatch.setattr(
        runner,
        "save_training_results_json",
        lambda *_args, **_kwargs: calls.__setitem__("saved_json", calls["saved_json"] + 1),
    )
    return calls


def test_execute_single_experiment_finalizes_after_training_interrupt(
    monkeypatch,
    tmp_path: Path,
) -> None:
    trainer = _Trainer(interrupt_training=True)
    calls = _patch_finalization(monkeypatch)

    result = runner.execute_single_experiment(
        config=_config(tmp_path),
        build_experiment_runtime_fn=lambda **_kwargs: _runtime(trainer, callback="mlflow"),
    )

    assert result["interrupted"] is True
    assert trainer.callbacks == ["mlflow"]
    assert calls["metrics"][0]["interrupted"] is True
    assert calls["metrics"][0]["logs"] == trainer.logs
    assert calls["saved_json"] == 1


def test_execute_single_experiment_saves_partial_metrics_after_evaluation_interrupt(
    monkeypatch,
    tmp_path: Path,
) -> None:
    trainer = _Trainer()
    calls = _patch_finalization(monkeypatch, evaluate_raises=True)

    result = runner.execute_single_experiment(
        config=_config(tmp_path),
        build_experiment_runtime_fn=lambda **_kwargs: _runtime(trainer),
    )

    assert result["interrupted"] is False
    assert result["final_metrics"]["interrupted"] is True
    assert calls["metrics"][0]["split_results"] == {}
    assert calls["metrics"][0]["final_reward"] != calls["metrics"][0]["final_reward"]
    assert calls["saved_json"] == 1


def test_execute_single_experiment_resume_uses_checkpoint_callback_and_name(
    monkeypatch,
    tmp_path: Path,
) -> None:
    trainer = _Trainer()
    resume_callback = object()
    runtime_kwargs: dict[str, Any] = {}
    resume_kwargs: dict[str, Any] = {}
    calls = _patch_finalization(monkeypatch)

    def build_runtime(**kwargs: Any) -> ExperimentRuntime:
        runtime_kwargs.update(kwargs)
        return _runtime(trainer, callback=None, effective_name="fresh")

    def fake_setup_checkpoint_resumption(**kwargs: Any) -> SimpleNamespace:
        resume_kwargs.update(kwargs)
        return SimpleNamespace(
            effective_experiment_name="resumed",
            mlflow_callback=resume_callback,
        )

    monkeypatch.setattr(
        runner,
        "setup_checkpoint_resumption",
        fake_setup_checkpoint_resumption,
    )

    result = runner.execute_single_experiment(
        config=_config(tmp_path),
        checkpoint_path="checkpoint.pt",
        additional_steps=50,
        build_experiment_runtime_fn=build_runtime,
    )

    assert result["interrupted"] is False
    assert runtime_kwargs["create_mlflow_callback"] is False
    assert trainer.callbacks == [resume_callback]
    assert resume_kwargs["checkpoint_path"] == "checkpoint.pt"
    assert resume_kwargs["additional_steps"] == 50
    assert calls["metrics"][0]["effective_experiment_name"] == "resumed"
    assert calls["checkpoints"][0]["checkpoint_path"] == "checkpoint.pt"


def test_configure_periodic_hooks_filters_splits_and_clamps_eval_steps(
    monkeypatch,
) -> None:
    class _HookTrainer:
        def __init__(self) -> None:
            self.eval_kwargs: dict[str, Any] | None = None
            self.explain_kwargs: dict[str, Any] | None = None

        def setup_periodic_evaluation(self, **kwargs: Any) -> None:
            self.eval_kwargs = kwargs

        def setup_periodic_explainability(self, **kwargs: Any) -> None:
            self.explain_kwargs = kwargs

    train_df = pd.DataFrame({"close": [100.0, 101.0, 102.0, 103.0]})
    val_df = pd.DataFrame({"close": [200.0]})
    test_df = pd.DataFrame({"close": [300.0, 301.0, 302.0]})
    dataset = PreparedDataset(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        feature_columns=["close"],
        price_column="close",
        raw_columns=["close"],
        feature_pipeline_state={},
    )
    trainer = _HookTrainer()
    runtime = ExperimentRuntime(
        logger=_Logger(),
        effective_experiment_name="hooks",
        prepared_dataset=dataset,
        training_bundle=TrainingBundle(
            train_env=object(),
            trainer=trainer,
            mlflow_callback=None,
            algorithm="TD3",
            n_obs=1,
            n_act=1,
        ),
    )
    config = SimpleNamespace(
        training=SimpleNamespace(
            temp_eval_splits=["train", "val", "missing", "test"],
            temp_eval_max_steps=2,
        ),
        explainability=SimpleNamespace(n_steps=7),
    )
    built: list[str] = []

    def fake_build_context_for_split(*, split, df, config):
        built.append(split)
        return SimpleNamespace(split=split, df=df, max_steps=len(df) + 10, env=f"env-{split}")

    monkeypatch.setattr(
        runner,
        "build_evaluation_context_for_split",
        fake_build_context_for_split,
    )

    runner._configure_periodic_hooks(runtime=runtime, config=config)

    assert built == ["train", "test", "train"]
    assert trainer.eval_kwargs is not None
    assert trainer.eval_kwargs["algorithm"] == "TD3"
    eval_splits = trainer.eval_kwargs["splits"]
    assert [ctx.split for ctx in eval_splits] == ["train", "test"]
    assert [ctx.max_steps for ctx in eval_splits] == [2, 2]
    assert [ctx.eval_env for ctx in eval_splits] == ["env-train", "env-test"]
    assert trainer.explain_kwargs == {
        "df": train_df,
        "max_steps": 7,
        "config": config,
        "eval_env": "env-train",
    }
