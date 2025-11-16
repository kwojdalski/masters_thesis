"""
Test module for action probabilities plotting functionality.
"""

import os

# Import the function we want to test
import sys
from unittest.mock import Mock

import pandas as pd
import pytest
import torch
from tensordict import TensorDict

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from trading_rl.train_trading_agent import _create_action_probabilities_plot


class TestActionProbabilitiesPlot:
    """Test cases for the action probabilities plotting function."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        # Create a mock environment that returns proper TensorDict observations
        self.mock_env = Mock()

        # Create initial observation TensorDict
        self.initial_obs = TensorDict(
            {
                "observation": torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]),
                "done": torch.tensor([False]),
            },
            batch_size=(1,),
        )

        # Mock environment reset
        self.mock_env.reset.return_value = self.initial_obs

        # Create a sequence of step responses with varying observations
        self.step_responses = []
        for i in range(10):
            # Create varying observations to ensure policy gets different inputs
            obs_values = torch.tensor(
                [
                    [
                        1.0 + i,
                        2.0 + i * 0.5,
                        3.0 - i * 0.3,
                        4.0 + i * 0.2,
                        5.0 - i * 0.1,
                        6.0 + i * 0.4,
                    ]
                ]
            )
            step_response = TensorDict(
                {
                    "observation": obs_values,
                    "done": torch.tensor([i >= 8]),  # Episode ends after 8 steps
                },
                batch_size=(1,),
            )
            self.step_responses.append(step_response)

        # Mock environment step method
        self.mock_env.step.side_effect = self.step_responses

        # Create a mock PPO actor that returns varying probabilities
        self.mock_actor = Mock()

        # Create varying logits to ensure probabilities change over time
        self.logits_sequence = []
        for i in range(10):
            # Create logits that favor different actions at different times
            if i % 3 == 0:
                logits = torch.tensor([[2.0, 0.5, 0.3]])  # Favor action 0 (Short)
            elif i % 3 == 1:
                logits = torch.tensor([[0.3, 2.0, 0.5]])  # Favor action 1 (Hold)
            else:
                logits = torch.tensor([[0.5, 0.3, 2.0]])  # Favor action 2 (Long)

            mock_output = Mock()
            mock_output.logits = logits
            mock_output.sample.return_value = torch.tensor(
                [i % 3]
            )  # Return corresponding action
            self.logits_sequence.append(mock_output)

        self.mock_actor.side_effect = self.logits_sequence

    def test_action_probabilities_plot_creation(self):
        """Test that the function creates a plot without errors."""
        max_steps = 10

        # Call the function
        plot = _create_action_probabilities_plot(
            self.mock_env, self.mock_actor, max_steps
        )

        # Verify that a plot object is returned
        assert plot is not None

        # Verify that the environment was reset
        self.mock_env.reset.assert_called_once()

        # Verify that the actor was called multiple times
        assert self.mock_actor.call_count > 0

        # Verify that environment step was called
        assert self.mock_env.step.call_count > 0

    def test_probabilities_vary_over_time(self):
        """Test that action probabilities actually vary over time."""
        max_steps = 9  # Test with 9 steps

        # Call the function
        plot = _create_action_probabilities_plot(
            self.mock_env, self.mock_actor, max_steps
        )

        # Extract the data from the plot
        # Note: This is a bit hacky but necessary to test the actual data
        plot_data = plot.data

        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(plot_data)

        # Verify we have data for all three actions
        actions = df["Action"].unique()
        assert set(actions) == {"Short", "Hold", "Long"}

        # Verify probabilities sum to 1 for each step
        for step in df["Step"].unique():
            step_data = df[df["Step"] == step]
            prob_sum = step_data["Probability"].sum()
            assert abs(prob_sum - 1.0) < 1e-6, (
                f"Probabilities don't sum to 1 at step {step}: {prob_sum}"
            )

        # Verify that probabilities actually vary across steps
        # Get probabilities for action 'Short' across all steps
        short_probs = df[df["Action"] == "Short"]["Probability"].values

        # Check that not all probabilities are the same (within small tolerance)
        if len(short_probs) > 1:
            prob_variance = torch.var(torch.tensor(short_probs))
            assert prob_variance > 1e-6, (
                "Action probabilities appear to be constant across time steps"
            )

    def test_proper_observation_extraction(self):
        """Test that observations are properly extracted from TensorDict."""
        max_steps = 5

        # Call the function
        _create_action_probabilities_plot(self.mock_env, self.mock_actor, max_steps)

        # Verify that actor was called with proper observation tensors
        for call in self.mock_actor.call_args_list:
            obs_arg = call[0][0]  # First argument to actor call

            # Verify it's a tensor (not a TensorDict)
            assert isinstance(obs_arg, torch.Tensor)

            # Verify it has the expected shape (should be observation data)
            assert obs_arg.shape[-1] == 6  # Should be 6 features

    def test_episode_termination_handling(self):
        """Test that the function properly handles episode termination."""
        max_steps = 15  # More than our mock episode length

        # Call the function
        plot = _create_action_probabilities_plot(
            self.mock_env, self.mock_actor, max_steps
        )

        # Verify the function stopped before max_steps due to episode termination
        plot_data = plot.data
        df = pd.DataFrame(plot_data)

        # Should have stopped when episode ended (at step 8 in our mock)
        max_step_in_data = df["Step"].max()
        assert max_step_in_data <= 8, (
            f"Function didn't stop at episode termination, max step: {max_step_in_data}"
        )

    def test_fallback_behavior(self):
        """Test that the function handles errors gracefully with fallback."""
        # Create a mock actor that raises an exception
        broken_actor = Mock(side_effect=Exception("Mock error"))

        max_steps = 10

        # Call the function - should not raise but return fallback plot
        plot = _create_action_probabilities_plot(self.mock_env, broken_actor, max_steps)

        # Verify that a plot is still returned (fallback behavior)
        assert plot is not None

    def test_step_limiting(self):
        """Test that the function properly limits visualization steps."""
        max_steps = 500  # Large number

        # Call the function
        plot = _create_action_probabilities_plot(
            self.mock_env, self.mock_actor, max_steps
        )

        # Extract data
        plot_data = plot.data
        df = pd.DataFrame(plot_data)

        # Should be limited to 200 steps for visualization
        max_step_in_data = df["Step"].max()
        assert max_step_in_data < 200, (
            f"Steps not properly limited for visualization: {max_step_in_data}"
        )


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])
