"""The concrete export stages, in the order the snapshot needs them.

Importing this module populates :class:`ThesisExportRegistry`. Each stage
wraps a script that already exists; nothing here reimplements an export, so
the per-artifact logic stays in the script that owns it and this file only
records what that script needs and when it has to run.

Order groups:

* **10 -- run artifacts.** Needs the MLflow store. Writes the
  ``latest_finished/`` snapshot every results table reads.
* **20 -- dataset artifacts.** Needs the prepared parquet data. Writes
  ``peek/*``: split sizes, the raw file inventory, feature statistics and
  correlations, the tick break-even table, the observation sample.
* **30 -- figures.** Needs the MLflow store, because the rollout parquets
  live in the artifact directory. Writes downsampled copies under
  ``evaluation_plots*/`` so a CI render -- which has no ``mlruns/`` -- can
  still draw the figures.
* **90 -- derived from the snapshot.** Needs only what the stages above
  wrote, so this is the one group that runs anywhere, CI included. Currently
  the LaTeX value macros.

Two scripts were deliberately not given stages. ``export_thesis_results.py``
is superseded by ``export_eval_to_thesis.py`` and referenced by nothing, and
``generate_transformed_features_table.py`` printed a markdown table to stdout
for a human to paste into a chapter -- the exact frozen-value pattern the
macro stage exists to remove, and already superseded by the wired
``@tbl-transformed-features``. Both are deleted rather than wrapped.
"""

from __future__ import annotations

from masters_thesis.export.registry import Requirement, Stage, ThesisExportRegistry

_UV = ("uv", "run", "python")


def _script(name: str, *args: str) -> tuple[str, ...]:
    return (*_UV, f"scripts/{name}", *args)


ThesisExportRegistry.register(
    Stage(
        name="eval",
        description="Per-scenario run snapshot (metrics, hyperparameters, statistical tests)",
        command=_script("export_all_to_thesis.py"),
        requires=frozenset({Requirement.MLFLOW}),
        order=10,
    )
)

# Order 15, ahead of the order-20 group, and the reason is a genuine overlap.
# This stage does not compute anything: it copies four files out of
# reports/peek/<scenario>/, the gitignored scratch that `cli.py peek dataset
# --export` writes. Two of those four -- correlations.csv and
# feature_stats.csv -- are also produced, from the memmaps directly, by the
# feature-correlations and feature-stats stages below. Sorting inside a group
# is alphabetical, so at equal order "peek" would run last and overwrite the
# freshly computed pair with whatever the scratch happened to hold; the
# checked-in correlations.csv is 2,370 bytes against the scratch copy's 740,
# because the computed version is the current one. Running first makes the
# computed stages authoritative and leaves peek owning only the two files
# nothing else writes: raw_file_inventory.json and splits.json.
#
# It also means this stage is only as fresh as the last `peek dataset
# --export`, which is not part of the pipeline. That is the remaining
# staleness hole a manifest would surface.
ThesisExportRegistry.register(
    Stage(
        name="peek",
        description="Split sizes, timestamp ranges and the raw file inventory",
        command=_script("export_peek_to_thesis.py"),
        requires=frozenset({Requirement.PEEK_SCRATCH}),
        order=15,
    )
)

ThesisExportRegistry.register(
    Stage(
        name="feature-stats",
        description="Per-feature distribution statistics",
        command=_script("export_streaming_feature_stats_to_thesis.py"),
        requires=frozenset({Requirement.PREPARED_DATA}),
        order=20,
    )
)

ThesisExportRegistry.register(
    Stage(
        name="feature-correlations",
        description="Feature-return correlations",
        command=_script("export_streaming_feature_correlations_to_thesis.py"),
        requires=frozenset({Requirement.PREPARED_DATA}),
        order=20,
    )
)

ThesisExportRegistry.register(
    Stage(
        name="tick-breakeven",
        description="Tick-size break-even fee per instrument",
        command=_script("export_tick_breakeven_to_thesis.py"),
        requires=frozenset({Requirement.PREPARED_DATA}),
        order=20,
    )
)

ThesisExportRegistry.register(
    Stage(
        name="observation-sample",
        description="Worked observation sample behind the transformed-features table",
        command=_script("export_observation_sample_to_thesis.py"),
        requires=frozenset({Requirement.PREPARED_DATA}),
        order=20,
    )
)

ThesisExportRegistry.register(
    Stage(
        name="algo-comparison-plots",
        description="Multi-algorithm rollout figures for the results chapter",
        command=_script("export_algo_comparison_plots.py"),
        requires=frozenset({Requirement.MLFLOW}),
        order=30,
    )
)

# The single-policy rollout is the fallback 06-03 uses when the multi-algorithm
# comparison is unavailable, so it is exported for the same H1 experiment the
# results tables report. Unlike the other scripts this one has no default
# experiment, hence the explicit argument.
_H1_EXPERIMENT = "pooled_td3_hft_lob_state_space_pooled_streaming_selected_dsr"

ThesisExportRegistry.register(
    Stage(
        name="rollout-plots",
        description="Single-policy rollout figures (fallback when the comparison is absent)",
        command=_script(
            "export_rollout_plots_to_thesis.py", "--experiment-name", _H1_EXPERIMENT
        ),
        requires=frozenset({Requirement.MLFLOW}),
        order=30,
    )
)

ThesisExportRegistry.register(
    Stage(
        name="value-macros",
        description="LaTeX macros for the result figures quoted in prose",
        command=_script("generate_thesis_value_macros.py"),
        requires=frozenset({Requirement.SNAPSHOT}),
        order=90,
    )
)
