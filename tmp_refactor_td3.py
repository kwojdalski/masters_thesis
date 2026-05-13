"""Helper to remove TD3 evaluate() method body."""
import sys

# Read the file
with open('/Users/krzysztofwojdalski/github_projects/masters_thesis/src/trading_rl/trainers/td3.py', 'r') as f:
    content = f.readlines()

# Find where evaluate() starts
start_line = None
for i, line in enumerate(content):
    if '    def evaluate(' in line:
        start_line = i
        break

if start_line is None:
    print("ERROR: Could not find evaluate() method")
    sys.exit(1)

# Find where evaluate() ends (return statement)
end_line = None
for i, line in enumerate(content[start_line:], start=start_line):
    if line.startswith('    @staticmethod') and i > start_line:
        # Found the decorator for the next method
        end_line = i
        break

if end_line is None:
    print("ERROR: Could not find end of evaluate() method")
    sys.exit(1)

# Remove everything from start_line to end_line (exclusive)
new_content = content[:start_line] + content[end_line:]

# Write back
with open('/Users/krzysztofwojdalski/github_projects/masters_thesis/src/trading_rl/trainers/td3.py', 'w') as f:
    f.write(''.join(new_content))

print(f"Removed {end_line - start_line} lines from TD3 evaluate() method")
