"""Helper to replace PPO evaluate() method."""
import sys

# Read the file
with open('/Users/krzysztofwojdalski/github_projects/masters_thesis/src/trading_rl/trainers/ppo.py', 'r') as f:
    content = f.readlines()

# Find where evaluate() starts (line 325)
start_line = None
for i, line in enumerate(content):
    if '    def evaluate(' in line:
        start_line = i
        break

if start_line is None:
    print("ERROR: Could not find evaluate() method")
    sys.exit(1)

# Find where evaluate() body ends (before build_models)
end_line = None
for i, line in enumerate(content[start_line:], start=start_line):
    if '    @staticmethod' in line:
        end_line = i
        break

if end_line is None:
    print("ERROR: Could not find end of evaluate() method")
    sys.exit(1)

# Extract line 325 (method signature) and everything up to @staticmethod
line_325 = content[start_line]
method_sig = line_325[:30]  # First 30 chars
method_body = line_325[30:]  # Rest

# Create new method - just delegates to super()
new_line = f"    def evaluate{method_sig}\n"
new_method = f"""Delegated evaluation - uses StrategyEvaluator from BaseTrainer."""\n"

# Extract @staticmethod and build_models lines
remaining_lines = content[end_line:]

# Write back
with open('/Users/krzysztofwojdalski/github_projects/masters_thesis/src/trading_rl/trainers/ppo.py', 'w') as f:
    f.write(''.join([new_line, new_method, *remaining_lines]))

print(f"Replaced PPO evaluate() method (kept {len(content) - len(remaining_lines)} lines)")
