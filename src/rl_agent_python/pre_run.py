"""
A Reinforcement Learning Application for Statistical Arbitrage
ATRGStrategies

This module sets up the environment and imports all necessary dependencies
for the reinforcement learning application.
"""

from pathlib import Path

# Add required packages to requirements.txt:
# numpy
# pandas
# statsmodels
# scikit-learn
# matplotlib
# seaborn
# tqdm
# typing

# Set up data path
DATA_PATH = Path(__file__).parent.parent / "data" / "raw" / "1min"

# Import all required modules
from . import mcc  # renamed from MCC.py to follow Python naming conventions
from . import (
    attributes,
    cointegration,
    discretization,
    financial_functions,
    getdata,
    q_learning,
    reward,
    rl_utils,
    state_space,
    tdva,
    utils,
)

# Make all modules available at package level
__all__ = [
    # "object_utils",
    "utils",
    "getdata",
    "cointegration",
    "attributes",
    "financial_functions",
    "discretization",
    "state_space",
    "reward",
    "mcc",
    "q_learning",
    "tdva",
    "rl_utils",
]
