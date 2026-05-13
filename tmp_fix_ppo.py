"""Fix PPO evaluate() method."""

# Read and write
with open('/Users/krzysztofwojdalski/github_projects/masters_thesis/src/trading_rl/trainers/ppo.py', 'r') as f:
    content = f.readlines()

# Keep first 324 lines
first_part = content[:324]

# New evaluate method
new_method = """Delegated evaluation - uses StrategyEvaluator from BaseTrainer."""
new_eval = f"""    def evaluate(\\n"""
new_eval += """        self,
        df,
        max_steps: int,
        config=None,
        algorithm: str | None = None,
        eval_env: Any | None = None,
    ):
        \"\"\"Delegated evaluation - uses StrategyEvaluator from BaseTrainer.\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"""  # Keep 30 backslashes to match

# Write back
with open('/Users/krzysztofwojdalski/github_projects/masters_thesis/src/trading_rl/trainers/ppo.py', 'w') as f:
    f.write(first_part + new_method + new_eval)
print("Fixed PPO evaluate() method")
