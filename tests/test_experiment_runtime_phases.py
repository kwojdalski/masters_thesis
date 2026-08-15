"""Regression tests for experiment configuration/construction phase separation."""

from types import SimpleNamespace

import trading_rl.pipeline.experiment_runner as runner
import trading_rl.pipeline.training as training


def test_build_runtime_uses_preconfigured_environment_without_global_setup(
    monkeypatch,
) -> None:
    logger = SimpleNamespace(info=lambda *args: None, debug=lambda *args: None)
    environment = training.ExperimentEnvironment(
        logger=logger,
        effective_experiment_name="configured",
    )
    dataset = SimpleNamespace(train_df=[], val_df=[], test_df=[])
    bundle = SimpleNamespace(mlflow_callback=None)
    monkeypatch.setattr(
        training,
        "_configure_experiment_environment",
        lambda *_args: (_ for _ in ()).throw(AssertionError("unexpected global setup")),
    )
    monkeypatch.setattr(training, "build_prepared_dataset", lambda *_args: dataset)
    monkeypatch.setattr(training, "_log_data_diagnostics", lambda *_args: None)
    monkeypatch.setattr(training, "_build_training_bundle", lambda **_kwargs: bundle)
    monkeypatch.setattr(training, "_log_mlflow_artifacts", lambda *_args: None)

    runtime = training.build_experiment_runtime(
        config=SimpleNamespace(data=SimpleNamespace(data_path="x", train_size=1)),
        environment=environment,
        create_mlflow_callback=False,
    )

    assert runtime.logger is logger
    assert runtime.effective_experiment_name == "configured"
    assert runtime.prepared_dataset is dataset
    assert runtime.training_bundle is bundle


def test_execute_configures_environment_before_runtime_construction(
    monkeypatch,
) -> None:
    calls: list[str] = []
    environment = training.ExperimentEnvironment(
        logger=SimpleNamespace(),
        effective_experiment_name="configured",
    )
    monkeypatch.setattr(runner, "run_guardrail_check", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        runner,
        "_configure_experiment_environment",
        lambda *_args: calls.append("configure") or environment,
    )

    def build_runtime(**kwargs):
        calls.append("build")
        assert kwargs["environment"] is environment
        raise RuntimeError("stop after construction boundary")

    config = SimpleNamespace(profiling=SimpleNamespace(level=0))

    try:
        runner.execute_single_experiment(
            config=config,
            build_experiment_runtime_fn=build_runtime,
        )
    except RuntimeError as exc:
        assert str(exc) == "stop after construction boundary"

    assert calls == ["configure", "build"]
