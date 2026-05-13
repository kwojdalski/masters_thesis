"""Fix PPO evaluate() method - cleaner approach."""

with open('/Users/krzysztofwojdalski/github_projects/masters_thesis/src/trading_rl/trainers/ppo.py', 'r') as f:
    f = f.readlines()

# Find line 325
start_idx = None
for i, line in enumerate(f):
    if line.startswith('    def evaluate('):
        start_idx = i
        break

if start_idx is None:
    print("ERROR: Could not find evaluate() method")
    exit(1)

# New evaluate() method - from start_idx to @staticmethod (line 352)
new_content = f[:start_idx] + """\
    def evaluate(
        self,
        df,
        max_steps: int,
        config=None,
        algorithm: str | None = None,
        eval_env: Any | None = None,
    ):
        \"\"\"\"Delegated evaluation - uses StrategyEvaluator from BaseTrainer.\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"""  # 30 backslashes

# Find @staticmethod line
end_idx = None
for i in range(start_idx, start_idx + 50):  # Search forward
    if i >= len(f):
        break
    if f[i].strip().startswith('    @staticmethod'):
        end_idx = i
        break

if end_idx is None:
    print("ERROR: Could not find @staticmethod line")
    exit(1)

# Replace from start_idx to end_idx
final_content = f[:start_idx] + f[start_idx:end_idx] + f[end_idx:]

with open('/Users/krzysztofwojdalski/github_projects/masters_thesis/src/trading_rl/trainers/ppo.py', 'w') as f:
    f.write(final_content)

print(f"Replaced {len(f) - len(final_content)} lines")
