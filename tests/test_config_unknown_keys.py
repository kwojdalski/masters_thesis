import pytest

from trading_rl.config import ExperimentConfig


@pytest.mark.parametrize(
    "training_config,expected_path",
    [
        ({"ppo_epochs": 1}, "training.ppo_epochs"),
        ({"eval_steps": 4}, "training.eval_steps"),
    ],
)
def test_removed_flat_training_keys_are_rejected(
    training_config: dict[str, int], expected_path: str
) -> None:
    with pytest.raises(ValueError, match=expected_path.replace(".", r"\.")):
        ExperimentConfig.from_dict({"training": training_config})
