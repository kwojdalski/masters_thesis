"""Fix TD3 evaluate() method - delegate to super().evaluate()."""

# Read the file
with open('src/trading_rl/trainers/td3.py', 'r') as f:
    lines = f.readlines()

# Find where evaluate() starts (line 138)
start_idx = None
for i, line in enumerate(lines):
    if '    def evaluate(' in line:
        start_idx = i
        break

if start_idx is None:
    print("ERROR: Could not find evaluate() method")
    exit(1)

# Find where evaluate() ends (before the next method)
# Look for the next method definition or class definition
end_idx = None
for i in range(start_idx + 1, len(lines)):
    stripped = lines[i].strip()
    if stripped.startswith('def ') and not stripped.startswith('def __'):  # Next method
        end_idx = i
        break
    if stripped.startswith('@'):  # Next decorator (method with decorator)
        end_idx = i
        break

if end_idx is None:
    print("ERROR: Could not find end of evaluate() method")
    exit(1)

print(f"Found evaluate() method from line {start_idx + 1} to {end_idx}")

# Create new evaluate method
new_evaluate = '''    def evaluate(
        self,
        df,
        max_steps: int,
        config=None,
        algorithm: str | None = None,
        eval_env: Any | None = None,
    ):
        """Delegated evaluation - uses StrategyEvaluator from BaseTrainer."""
        return super().evaluate(
            df=df,
            max_steps=max_steps,
            config=config,
            algorithm=algorithm,
            eval_env=eval_env,
        )

'''

# Replace the method
new_lines = lines[:start_idx] + [new_evaluate] + lines[end_idx:]

# Write back
with open('src/trading_rl/trainers/td3.py', 'w') as f:
    f.writelines(new_lines)

print(f"Replaced evaluate() method (removed {end_idx - start_idx} lines, added {len(new_evaluate.split(chr(10)))} lines)")
