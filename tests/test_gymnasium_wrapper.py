"""Unit tests for GymnasiumTradingEnvWrapper, focusing on the bankrupt handler."""

from __future__ import annotations

from unittest.mock import MagicMock

import gymnasium as gym
import numpy as np
import pytest
from tradingenv.broker.broker import EndOfEpisodeError

from trading_rl.envs.tradingenvxy_wrapper import GymnasiumTradingEnvWrapper


def _wrapper_dict_obs() -> GymnasiumTradingEnvWrapper:
    """Wrapper whose observation_space is a Dict (shape is None) — like TradingEnv."""
    inner = MagicMock()
    inner.observation_space = gym.spaces.Dict(
        {"feat": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)}
    )
    inner.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
    return GymnasiumTradingEnvWrapper(inner)


def _wrapper_box_obs() -> GymnasiumTradingEnvWrapper:
    """Wrapper whose observation_space is a Box (shape is not None)."""
    inner = MagicMock()
    inner.observation_space = gym.spaces.Box(
        low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32
    )
    inner.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
    return GymnasiumTradingEnvWrapper(inner)


class TestBankruptHandler:
    def test_dict_obs_returns_zero_dict(self):
        """Regression: Dict space (shape=None) must not call np.zeros(None)."""
        wrapper = _wrapper_dict_obs()
        wrapper._env.step.side_effect = EndOfEpisodeError("bankrupt")
        obs, reward, terminated, truncated, info = wrapper.step(np.array([0.0]))

        assert isinstance(obs, dict), "observation must be a dict for Dict space"
        assert "feat" in obs
        np.testing.assert_allclose(obs["feat"], np.zeros(3, dtype=np.float32))
        assert reward == pytest.approx(-1.0)
        assert terminated is True
        assert truncated is False
        assert info.get("bankrupt") is True

    def test_box_obs_returns_zero_array(self):
        wrapper = _wrapper_box_obs()
        wrapper._env.step.side_effect = EndOfEpisodeError("bankrupt")
        obs, reward, terminated, truncated, info = wrapper.step(np.array([0.0]))

        assert isinstance(obs, np.ndarray)
        assert obs.shape == (4,)
        np.testing.assert_allclose(obs, np.zeros(4, dtype=np.float32))
        assert reward == pytest.approx(-1.0)
        assert terminated is True
        assert truncated is False
        assert info.get("bankrupt") is True

    def test_normal_step_passes_through(self):
        wrapper = _wrapper_box_obs()
        fake_obs = np.ones(4, dtype=np.float32) * 0.5
        wrapper._env.step.return_value = (fake_obs, 0.3, False, {"info_key": 1})
        obs, reward, terminated, truncated, info = wrapper.step(np.array([0.5]))

        np.testing.assert_allclose(obs, fake_obs)
        assert reward == pytest.approx(0.3)
        assert terminated is False
        assert truncated is False

    def test_reset_returns_obs_and_info(self):
        wrapper = _wrapper_box_obs()
        wrapper._env.reset.return_value = np.ones(4, dtype=np.float32)
        obs, info = wrapper.reset()

        assert isinstance(obs, np.ndarray)
        assert isinstance(info, dict)
