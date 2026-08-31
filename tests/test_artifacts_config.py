from trading_rl import ExperimentConfig
from trading_rl.callbacks.artifacts_config import log_training_parameters


def test_log_training_parameters_supports_unbounded_train_size(monkeypatch):
    logged = {}
    monkeypatch.setattr(
        "trading_rl.callbacks.artifacts_config.mlflow.log_param",
        lambda key, value: logged.__setitem__(key, value),
    )
    config = ExperimentConfig()
    config.seed = 42
    config.data.train_size = None

    log_training_parameters(config)

    assert logged["data_train_size"] == "all"
