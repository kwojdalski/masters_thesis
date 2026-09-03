#!/usr/bin/env python3
"""Emit LaTeX macros for result values quoted in thesis prose.

A number typed into a sentence -- "$312.8\\%$" -- is frozen the moment it is
written. The table beside it recomputes on every render; the sentence does
not, so a re-run silently leaves the prose asserting a stale figure. An audit
on 2026-09-03 found roughly ninety such values across chapters 4-7.

The obvious fix, a Quarto inline expression, does not work here: this
document wraps prose numbers in ``$...$`` and an inline expression inside
math mode is *silently corrupted* rather than rejected -- a rendered test
turned "49.8%" into "498", losing the decimal point with no error and no
failed build. Dropping the math delimiters avoids that but shifts every
wired number from the math font to the text font, a visible change across
the whole results chapter.

This script takes the third route. It reads the exported artifacts and
writes ``\\newcommand`` definitions to a generated ``.tex`` file that is
pulled in through ``include-in-header``. Prose then reads
``$\\valDdpgReturn$``: still math mode, still the same glyphs, but the value
is regenerated from the artifact on every render and cannot drift. Because
the file is generated during ``pre-render``, an artifact that changes is
reflected without touching a single ``.qmd``.

An undefined macro is a hard LaTeX error, which is the intended behaviour --
a missing artifact should stop the build loudly rather than silently render
a gap where a number belongs.

Naming: TeX command names admit letters only, so digits are spelled out
(``td3`` -> ``TdThree``). Every macro is prefixed ``val`` to keep the
generated namespace clearly separate from the class file's own commands.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "thesis" / "qmd" / "src"
sys.path.insert(0, str(_SRC))

from thesis_mlflow_results import load_scenario_metrics  # noqa: E402

_OUT = _SRC / "_generated_values.tex"

# (scenario directory, macro infix). Digits are spelled out because a TeX
# command name cannot contain them.
_H1_AGENTS = [
    ("pooled_td3_hft_lob_state_space_pooled_streaming_selected_dsr", "TdThree"),
    ("pooled_ddpg_hft_lob_state_space_pooled_streaming_selected_dsr", "Ddpg"),
    ("pooled_ppo_hft_lob_state_space_pooled_streaming_selected_dsr", "Ppo"),
    ("pooled_random_hft_lob_state_space_pooled_streaming_selected_dsr", "Random"),
]

# (metrics key, macro suffix, format spec). The format spec must reproduce
# the digits the prose already prints -- this file replaces the mechanism
# behind a number, never the number's appearance. A percentage renders
# through _pct so the '%' arrives escaped for LaTeX.
_METRICS = [
    ("total_return", "Return", "pct1"),
    ("profit_factor", "ProfitFactor", ".2f"),
    ("sharpe_ratio", "Sharpe", ".2f"),
    ("turnover", "Turnover", ".2f"),
    ("pct_long", "PctLong", "pct1"),
    ("max_drawdown", "MaxDrawdown", "pct2"),
]

# Hyperparameters of the main TD3 run, quoted throughout the design chapter.
# These are read from the run's own logged hyperparams rather than from the
# scenario YAML, so the prose states what the reported agent was actually
# trained with -- a config edit after the fact cannot silently desynchronise
# the sentence from the result it describes.
_HYPERPARAM_SOURCE = "pooled_td3_hft_lob_state_space_pooled_streaming_selected_dsr"

# (hyperparams key, macro suffix, format spec). "g" prints the shortest form
# that round-trips (0.9, 0.005, 0.0001) and matches the digits already in the
# prose; "latexsci" renders 2e-06 as the LaTeX "2 \times 10^{-6}" the text
# uses. Note fmt_scientific() in thesis_tables.py is NOT usable here -- it
# emits HTML <sup> for pandas tables, which would pass through literally.
_HYPERPARAMS = [
    ("gamma", "Gamma", "g"),
    ("tau", "Tau", "g"),
    ("reward_eta", "RewardEta", "g"),
    ("actor_lr", "ActorLr", "g"),
    ("exploration_noise_std", "ExplorationNoise", "g"),
    ("policy_noise", "PolicyNoise", "g"),
    ("noise_clip", "NoiseClip", "g"),
    ("policy_delay", "PolicyDelay", "d"),
    ("actor_weight_decay", "WeightDecay", "latexsci"),
]


def _latex_sci(value: float) -> str:
    """Render a float as LaTeX scientific notation, e.g. 2e-06 -> '2 \\times 10^{-6}'.

    Kept separate from thesis_tables.fmt_scientific, which emits an HTML
    <sup> tag for embedding in a pandas-generated table and would appear
    literally if it reached prose.
    """
    import math

    exponent = math.floor(math.log10(abs(value)))
    mantissa = value / (10**exponent)
    rounded = round(mantissa)
    mantissa_str = str(rounded) if abs(mantissa - rounded) < 1e-9 else f"{mantissa:g}"
    if mantissa_str == "1":
        return rf"10^{{{exponent}}}"
    return rf"{mantissa_str} \times 10^{{{exponent}}}"


def _render(value: object, spec: str) -> str | None:
    """Format one value, returning None when it is absent or non-finite."""
    if not isinstance(value, int | float):
        return None
    if value != value or value in (float("inf"), float("-inf")):  # NaN / inf
        return None
    if spec == "pct1":
        return f"{value:.1%}".replace("%", r"\%")
    if spec == "pct2":
        return f"{value:.2%}".replace("%", r"\%")
    if spec == "latexsci":
        return _latex_sci(float(value))
    if spec == "d":
        return f"{int(value):d}"
    return f"{value:{spec}}"


def build_definitions() -> tuple[list[str], list[str]]:
    """Return (macro lines, names of metrics that could not be resolved)."""
    lines: list[str] = []
    missing: list[str] = []

    for scenario, infix in _H1_AGENTS:
        metrics = load_scenario_metrics(scenario)
        for key, suffix, spec in _METRICS:
            name = f"val{infix}{suffix}"
            body = _render(metrics.get(key), spec)
            if body is None:
                missing.append(f"{scenario}:{key}")
                continue
            lines.append(rf"\newcommand{{\{name}}}{{{body}}}")

    hyper_path = (
        _SRC.parent
        / "results"
        / _HYPERPARAM_SOURCE
        / "latest_finished"
        / "hyperparams.json"
    )
    if hyper_path.exists():
        hyper = json.loads(hyper_path.read_text())
        for key, suffix, spec in _HYPERPARAMS:
            name = f"val{suffix}"
            body = _render(hyper.get(key), spec)
            if body is None:
                missing.append(f"{_HYPERPARAM_SOURCE}:{key}")
                continue
            lines.append(rf"\newcommand{{\{name}}}{{{body}}}")
    else:
        missing.append(f"{hyper_path} (not exported)")

    return lines, missing


def main() -> int:
    lines, missing = build_definitions()

    header = [
        "% Generated by scripts/generate_thesis_value_macros.py -- do not edit.",
        "% Regenerated on every render via the pre-render hook in _quarto.yml,",
        "% so each value below is read fresh from thesis/qmd/results/**.",
        "",
    ]
    _OUT.write_text("\n".join(header + lines) + "\n")

    print(f"generate_thesis_value_macros: wrote {len(lines)} macro(s) to {_OUT}")
    if missing:
        # Not fatal: a scenario may legitimately not be exported yet. The
        # macro is simply absent, and any prose referencing it fails loudly
        # at the LaTeX stage rather than rendering a wrong number.
        print(
            f"generate_thesis_value_macros: {len(missing)} metric(s) unresolved "
            "(prose referencing them will fail the LaTeX pass):"
        )
        for m in missing:
            print(f"  {m}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
