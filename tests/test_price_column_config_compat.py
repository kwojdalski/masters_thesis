from trading_rl.config import ExperimentConfig


def test_from_dict_maps_legacy_price_columns_to_price_column():
    config = ExperimentConfig.from_dict(
        {
            "env": {
                "backend": "tradingenv",
                "price_columns": ["close"],
            }
        }
    )
    assert config.env.price_column == "close"


def test_to_dict_emits_price_column_only():
    config = ExperimentConfig()
    config.env.backend = "tradingenv"
    config.env.price_column = "close"

    config_dict = config.to_dict()
    env = config_dict["environment"]
    assert env["price_column"] == "close"
    assert "price_columns" not in env
