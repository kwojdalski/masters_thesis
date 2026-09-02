#!/usr/bin/env python3
"""Export all hypothesis scenarios to thesis result snapshots.

Calls export_eval_to_thesis.py for every scenario in the specified hypothesis
(or all hypotheses).  Skips scenarios whose results.json is missing and prints
a summary at the end.

Usage:
    uv run python scripts/export_all_to_thesis.py              # all hypotheses
    uv run python scripts/export_all_to_thesis.py --hypothesis h1
    uv run python scripts/export_all_to_thesis.py --hypothesis h2
    uv run python scripts/export_all_to_thesis.py --hypothesis h4
    uv run python scripts/export_all_to_thesis.py --hypothesis h1 h2
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from logger import get_logger, setup_logging

logger = get_logger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[1]

# Canonical scenario lists per hypothesis (order matches the run scripts).
SCENARIOS: dict[str, list[str]] = {
    "h1": [
        "pooled/td3_hft_lob_state_space_pooled_streaming_selected_dsr",
        "pooled/ddpg_hft_lob_state_space_pooled_streaming_selected_dsr",
        "pooled/ppo_hft_lob_state_space_pooled_streaming_selected_dsr",
        "pooled/random_hft_lob_state_space_pooled_streaming_selected_dsr",
    ],
    "h2": [  # transaction-cost sensitivity (0 bp arm is the H1 agent)
        "pooled/td3_h2_fees_0_dsr",
        "pooled/td3_h2_fees_1e6_dsr",
        "pooled/td3_h2_fees_1e5_dsr",
        "pooled/td3_h2_fees_1e4_dsr",
    ],
    "h3": [  # feature specification
        "pooled/td3_h3_features_minimal",
        "pooled/td3_h3_features_selected",
        "pooled/td3_h3_features_full",
    ],
    "h4": [  # reward-function design
        "pooled/td3_h4_reward_logreturn",
        "pooled/td3_h4_reward_dsr",
    ],
    "h5": [  # execution latency
        "pooled/td3_h5_latency_0_dsr",
        "pooled/td3_h5_latency_10us_dsr",
        "pooled/td3_h5_latency_100us_dsr",
        "pooled/td3_h5_latency_1ms_dsr",
        "pooled/td3_h5_latency_5ms_dsr",
    ],
}

_EXPORT_SCRIPT = _REPO_ROOT / "scripts" / "export_eval_to_thesis.py"


def _export_scenario(scenario: str) -> bool:
    """Run export_eval_to_thesis.py for one scenario. Returns True on success."""
    result = subprocess.run(  # noqa: S603 -- fixed interpreter path + repo-local script, no untrusted input
        [sys.executable, str(_EXPORT_SCRIPT), "--scenario", scenario],
        cwd=_REPO_ROOT,
        env=os.environ,
    )
    return result.returncode == 0


def main() -> int:
    p = argparse.ArgumentParser(
        description="Export all hypothesis scenarios to thesis snapshots.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--hypothesis",
        nargs="+",
        choices=["h1", "h2", "h3", "h4", "h5"],
        default=["h1", "h2", "h3", "h4", "h5"],
        metavar="HYP",
        help="Which hypotheses to export (h1 h2 h3 h4 h5). Defaults to all.",
    )
    p.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable DEBUG logging.",
    )
    args = p.parse_args()

    env_level = os.environ.get("LOG_LEVEL", "INFO").upper()
    if env_level not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
        env_level = "INFO"
    level = "DEBUG" if args.verbose else env_level
    setup_logging(level=level)
    os.environ["LOG_LEVEL"] = level

    # Deduplicate while preserving order.
    seen: set[str] = set()
    scenarios: list[tuple[str, str]] = []
    for hyp in args.hypothesis:
        for s in SCENARIOS[hyp]:
            if s not in seen:
                seen.add(s)
                scenarios.append((hyp, s))

    hypotheses_label = " ".join(h.upper() for h in args.hypothesis)
    logger.info("exporting {} scenario(s) for {}", len(scenarios), hypotheses_label)

    ok: list[str] = []
    skipped: list[str] = []

    for hyp, scenario in scenarios:
        logger.info("[{}] {}", hyp.upper(), scenario)
        if _export_scenario(scenario):
            ok.append(scenario)
        else:
            skipped.append(scenario)
            logger.warning("[{}] export failed or skipped: {}", hyp.upper(), scenario)

    logger.info(
        "export complete  exported={}  skipped/failed={}",
        len(ok),
        len(skipped),
    )

    if skipped:
        logger.warning(
            "skipped scenarios (results.json missing or evaluate not yet run):"
        )
        for s in skipped:
            logger.warning("  {}", s)
        logger.warning(
            "to evaluate a missing scenario: "
            "uv run python src/cli.py evaluate -c <scenario> "
            "--output-dir logs/<log-name> --only metrics --only benchmarks --only plots"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
