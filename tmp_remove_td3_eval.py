"""Helper to remove old TD3 evaluate() method body."""
import sys

# Read the file
with open('/Users/krzysztofwojdalski/github_projects/masters_thesis/src/trading_rl/trainers/td3.py', 'r') as f:
    content = f.readlines()

# Find where new evaluate() ends (after imports)
start_line = None
for i, line in enumerate(content):
    if '    """Delegated' in line:
        start_line = i + 1  # Skip the docstring
        break

if start_line is None:
    print("ERROR: Could not find new evaluate() method")
    sys.exit(1)

# Find where @staticmethod for build_models starts
build_models_line = None
for i, line in enumerate(content[start_line:], start=start_line):
    if '    @staticmethod' in line:
        build_models_line = i
        break

if build_models_line is None:
    print("ERROR: Could not find build_models() method")
    sys.exit(1)

# Remove everything from after docstring to before build_models
new_content = content[:start_line] + content[build_models_line:]

# Write back
with open('/Users/krzysztofwojdalski/github_projects/masters_thesis/src/trading_rl/trainers/td3.py', 'w') as f:
    f.write(''.join(new_content))

print(f"Removed {len(content) - len(new_content)} lines from TD3 (kept docstring, new evaluate(), and build_models)")
