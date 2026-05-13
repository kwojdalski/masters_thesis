"""Fix PPO evaluate() - simplest approach."""

import sys

# Read
with open('ppo.py', 'r') as f:
    lines = f.readlines()

# Find line 325 (start of evaluate method)
start_idx = None
for i, line in enumerate(lines):
    if '    def evaluate(' in line:
        start_idx = i
        break

if start_idx is None:
    print("ERROR")
    sys.exit(1)

# Find next @staticmethod (line 352)
# Count forward to next method
end_idx = None
for i in range(start_idx, start_idx + 50):
    if i >= len(lines):
        break
    if lines[i].strip().startswith('    @staticmethod'):
        end_idx = i
        break

# Build new file: keep up to line 324, then new evaluate, then rest from after @staticmethod
new_lines = lines[:start_idx] + [
    "    def evaluate(",
    "\\n",
    "        self,",
    "        df,",
    "        max_steps: int,",
    "        config=None,",
    "        algorithm: str | None = None,
    "        eval_env: Any | None = None,",
    "    ):",
    '        """Delegated evaluation - uses StrategyEvaluator from BaseTrainer."""',
    "        return super().evaluate(',
    '            df=df,',
    '            max_steps=max_steps,',
    '            config=config,',
    '            algorithm=algorithm,',
    '            eval_env=eval_env,',
    '        )',
]

# Write remaining lines (should include rest after new evaluate and build_models)
remaining_lines = lines[end_idx:] if end_idx else []

with open('ppo.py', 'w') as f:
    f.writelines(new_lines + remaining_lines)

print("Fixed PPO evaluate() method")
