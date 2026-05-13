"""DSR Workflow Diagram skill implementation."""

import re
from pathlib import Path


def add_workflow_diagram_to_dsr(dsr_file_path: str) -> None:
    """Add visual workflow diagram to Differential Sharpe Ratio documentation.

    Args:
        dsr_file_path: Path to differential_sharpe.py file

    Returns:
        None (modifies file in place)
    """
    dsr_path = Path(dsr_file_path)

    if not dsr_path.exists():
        print(f"Error: File not found: {dsr_file_path}")
        return

    # Read the file content
    with open(dsr_path, 'r') as f:
        content = f.read()

    # Check if workflow diagram already exists
    has_workflow_diagram = "Workflow Diagram:" in content

    # Check if artifact consumption step already exists
    has_artifact_consumption = "Artifact Consumption" in content

    # Allow adding artifact consumption if workflow diagram exists (but prevent duplicates)
    if has_artifact_consumption:
        print("Artifact consumption step already exists in the file.")
        return

    # Find the position to insert the workflow diagram
    # We want to add it after the performance section but before the closing quotes
    performance_section_end = content.find("Memory footprint: O(1)")

    if performance_section_end == -1:
        print("Error: Could not find Performance Characteristics section")
        return

    # Workflow diagram to insert
    workflow_diagram = """
    Workflow Diagram:
    ──────────────────────────────────────────────────────────────────────────────────────────

    Step 1: Initialization
    ┌─────────────────────────────────────────────────────────────────────────────────────┐
    │  • Set EMA learning rate (η) and stability constant (ε)                   │
    │  • Initialize first moment EMA: A_t = 0.0 (mean estimate)               │
    │  • Initialize second moment EMA: B_t = 0.0 (squared returns)             │
    │  • Set previous net liquidation value: _prev_nlv = None                  │
    └─────────────────────────────────────────────────────────────────────────────────────┘

    Step 2: First Call Handling (when _prev_nlv is None)
    ┌─────────────────────────────────────────────────────────────────────────────────────┐
    │  • Get current net liquidation value: nlv_now                            │
    │  • Store as previous: _prev_nlv = nlv_now                                 │
    │  • Return reward: 0.0 (no calculation yet)                                │
    └─────────────────────────────────────────────────────────────────────────────────────┘

    Step 3: Normal Call Handling
    ┌─────────────────────────────────────────────────────────────────────────────────────┐
    │  INPUT: nlv_now (current portfolio value)                                  │
    │  INPUT: _prev_nlv (previous portfolio value)                                  │
    │  INPUT: A_t, B_t (previous EMA estimates at t-1)                           │
    │                                                                              │
    │  3.1 Calculate Log Return:                                                   │
    │      R_t = log(nlv_now / _prev_nlv)                                         │
    │                                                                              │
    │  3.2 Calculate Deltas (using OLD EMA values at t-1):                        │
    │      ΔA_t = R_t - A_t                    (change in mean estimate)            │
    │      ΔB_t = R_t² - B_t                   (change in second moment)         │
    │                                                                              │
    │  3.3 Calculate Variance:                                                     │
    │      Var = B_t - A_t²             (variance estimate)                      │
    │                                                                              │
    │  3.4 Calculate DSR (using OLD EMA values):                                     │
    │      numerator = B_t * ΔA_t - A_t * ΔB_t / 2                                  │
    │      denominator = max(Var, 0.0)^1.5 + ε                                        │
    │      D_t = numerator / denominator                                                   │
    │                                                                              │
    │  3.5 Update EMAs for NEXT step (t):                                          │
    │      A_t = (1 - η) * A_t + η * R_t        (update first moment)   │
    │      B_t = (1 - η) * B_t + η * R_t²        (update second moment)  │
    │                                                                              │
    │  3.6 Update Previous NLV:                                                    │
    │      _prev_nlv = nlv_now                                                        │
    │                                                                              │
    │  OUTPUT: D_t (Differential Sharpe Ratio reward)                                │
    └─────────────────────────────────────────────────────────────────────────────────────┘

    Step 4: Episode Reset Handling
    ┌─────────────────────────────────────────────────────────────────────────────────────┐
    │  • Reset first moment EMA: A_t = 0.0                                          │
    │  • Reset second moment EMA: B_t = 0.0                                        │
    │  • Clear previous NLV: _prev_nlv = None                                        │
    │                                                                              │
    │  Ready for new episode with fresh EMA state                                    │
    └─────────────────────────────────────────────────────────────────────────────────────┘

    Step 5: Artifact Consumption by Documentation
    ┌─────────────────────────────────────────────────────────────────────────────────────┐
    │  • Training artifacts (checkpoints, replay buffers) logged to MLflow                 │
    │  • Evaluation plots (rewards, actions, returns) saved as artifacts            │
    │  • Training logs (trading_env_debug.log) contain DSR calculation traces          │
    │  • Final metrics and evaluation reports consumed for documentation                       │
    │  • MLflow tracking provides artifact links for paper/dataset integration             │
    │                                                                              │
    │  Documentation Process:                                                       │
    │      1. Run training with DSR reward → MLflow artifacts                        │
    │      2. Generate evaluation plots → artifact storage                                 │
    │      3. Export metrics/reports → documentation consumption                           │
    │      4. Link artifacts in paper via MLflow UI or programmatic access               │
    └─────────────────────────────────────────────────────────────────────────────────────┘

    Key Design Decisions:
    ┌─────────────────────────────────────────────────────────────────────────────────────┐
    │  • Calculate DSR BEFORE updating EMAs (uses t-1 values)                    │
    │  • Use max(Var, 0.0) to prevent numerical issues with negative variance    │
    │  • Add ε (epsilon) to denominator for numerical stability                       │
    │  • EMA update formula: new = (1 - η) * old + η * current                      │
    │  • Initialize _prev_nlv to None to detect first call                              │
    └─────────────────────────────────────────────────────────────────────────────────────────────┘

    Performance Characteristics:
    ┌─────────────────────────────────────────────────────────────────────────────────────┐
    │  η (eta) = 0.01  → Moderate adaptation (balanced variance/smoothness)          │
    │  η (eta) = 0.1   → Fast adaptation (higher variance, quicker response)       │
    │  η (eta) = 0.001 → Slow adaptation (smoother signal, stable variance)        │
    │                                                                              │
    │  Computational cost per step: O(1) constant time                               │
    │  Memory footprint: O(1) - stores only A_t, B_t, _prev_nlv                    │
    └─────────────────────────────────────────────────────────────────────────────────────┘
"""

    # Insert the workflow diagram and additional steps
    new_content = content[:performance_section_end] + workflow_diagram + content[performance_section_end:]

    # Write back to file
    with open(dsr_path, 'w') as f:
        f.write(new_content)

    print(f"Workflow diagram added to: {dsr_file_path}")


def main():
    """Main entry point for the skill."""
    import sys

    if len(sys.argv) > 1:
        dsr_file = sys.argv[1]
    else:
        # Default path
        dsr_file = "src/trading_rl/rewards/differential_sharpe.py"

    add_workflow_diagram_to_dsr(dsr_file)


if __name__ == "__main__":
    main()
