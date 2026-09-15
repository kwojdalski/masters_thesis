from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from scripts import export_eval_to_thesis


def _touch(path: Path, timestamp_ns: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    os.utime(path, ns=(timestamp_ns, timestamp_ns))


def _args(
    output_dir: Path, results_root: Path, *, allow_stale: bool
) -> argparse.Namespace:
    return argparse.Namespace(
        scenario=None,
        output_dir=output_dir,
        experiment_name="test_experiment",
        plots_dir=None,
        thesis_results_root=results_root,
        allow_split_fallback=False,
        allow_stale_results=allow_stale,
        verbose=False,
    )


def test_newer_checkpoint_ignores_older_and_missing_checkpoints(tmp_path: Path) -> None:
    results_file = tmp_path / "results.json"
    _touch(results_file, 2_000_000_000)

    assert export_eval_to_thesis._newer_checkpoint(results_file) is None

    _touch(tmp_path / "run_checkpoint_step_1.pt", 1_000_000_000)
    assert export_eval_to_thesis._newer_checkpoint(results_file) is None


def test_main_refuses_results_older_than_newest_checkpoint(
    tmp_path: Path, monkeypatch
) -> None:
    eval_dir = tmp_path / "logs" / "experiment"
    eval_dir.mkdir(parents=True)
    results_file = eval_dir / "results.json"
    results_file.write_text(json.dumps({"test": {"metrics": {"total_return": 0.1}}}))
    _touch(results_file, 1_000_000_000)
    _touch(eval_dir / "run_checkpoint_step_2.pt", 2_000_000_000)
    results_root = tmp_path / "thesis-results"
    monkeypatch.setattr(
        export_eval_to_thesis,
        "_parse_args",
        lambda: _args(eval_dir, results_root, allow_stale=False),
    )

    assert export_eval_to_thesis.main() == 1
    assert not results_root.exists()


def test_main_allows_explicit_stale_results_override(
    tmp_path: Path, monkeypatch
) -> None:
    eval_dir = tmp_path / "logs" / "experiment"
    eval_dir.mkdir(parents=True)
    results_file = eval_dir / "results.json"
    results_file.write_text(json.dumps({"test": {"metrics": {"total_return": 0.1}}}))
    _touch(results_file, 1_000_000_000)
    _touch(eval_dir / "run_checkpoint_step_2.pt", 2_000_000_000)
    results_root = tmp_path / "thesis-results"
    monkeypatch.setattr(
        export_eval_to_thesis,
        "_parse_args",
        lambda: _args(eval_dir, results_root, allow_stale=True),
    )

    assert export_eval_to_thesis.main() == 0
    run_json_path = results_root / "test_experiment" / "latest_finished" / "run.json"
    assert run_json_path.exists()

    run_json = json.loads(run_json_path.read_text())
    # #11: do not synthesise an unverifiable FINISHED status or a fake run start.
    assert run_json["status"] != "FINISHED"
    assert run_json["start_time"] is None
    # end_time is results.json's mtime, not the export time.
    assert run_json["end_time"] != run_json["source"]["exported_at_utc"]
    assert run_json["source"]["results_file_mtime_utc"] == run_json["end_time"]


def _write_scenario_yaml(repo_root: Path, scenario: str, body: str) -> None:
    path = repo_root / "src" / "configs" / "scenarios" / scenario / "train.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")


def test_load_hyperparams_reads_nested_algo_block(tmp_path: Path) -> None:
    # #774: TD3 knobs live under training.td3.* in the scenario schema; a flat
    # training.get("policy_delay") returns None and the appendix then silently
    # substitutes code defaults.
    _write_scenario_yaml(
        tmp_path,
        "pooled/td3_example",
        """
training:
  algorithm: TD3
  gamma: 0.9
  td3:
    policy_delay: 2
    policy_noise: 0.2
    noise_clip: 0.3
    exploration_noise_std: 0.3
""",
    )

    hp = export_eval_to_thesis._load_scenario_hyperparams(
        "pooled/td3_example", tmp_path
    )

    assert hp["policy_delay"] == 2
    assert hp["policy_noise"] == 0.2
    assert hp["noise_clip"] == 0.3
    assert hp["exploration_noise_std"] == 0.3


def test_load_hyperparams_falls_back_to_flat_layout(tmp_path: Path) -> None:
    _write_scenario_yaml(
        tmp_path,
        "pooled/td3_flat",
        """
training:
  algorithm: TD3
  policy_delay: 3
  policy_noise: 0.1
""",
    )

    hp = export_eval_to_thesis._load_scenario_hyperparams("pooled/td3_flat", tmp_path)

    assert hp["policy_delay"] == 3
    assert hp["policy_noise"] == 0.1
    assert hp["noise_clip"] is None


def test_trained_steps_reads_highest_checkpoint(tmp_path: Path) -> None:
    # #797: the H2/H3/H4 sweeps ran at 500k via a launch-time override while
    # their configs still say 3,000,000. Checkpoint filenames are the only
    # on-disk record of how far a run actually got.
    for step in (100_000, 500_000, 300_000):
        (tmp_path / f"run_checkpoint_step_{step}.pt").touch()

    assert export_eval_to_thesis._trained_steps_from_checkpoints(tmp_path) == 500_000


def test_trained_steps_is_none_without_step_stamped_checkpoints(
    tmp_path: Path,
) -> None:
    # A bare "_checkpoint.pt" carries no step, so there is nothing to report and
    # the caller must keep the configured value rather than guess.
    (tmp_path / "run_checkpoint.pt").touch()

    assert export_eval_to_thesis._trained_steps_from_checkpoints(tmp_path) is None


def test_export_records_trained_steps_over_configured_max_steps(
    tmp_path: Path, monkeypatch
) -> None:
    # With --scenario set, the eval dir is derived from the scenario's last
    # component, so it must be named to match.
    eval_dir = tmp_path / "logs" / "td3_swept"
    eval_dir.mkdir(parents=True)
    results_file = eval_dir / "results.json"
    results_file.write_text(json.dumps({"test": {"metrics": {"total_return": 0.1}}}))
    _touch(results_file, 2_000_000_000)
    _touch(eval_dir / "run_checkpoint_step_500000.pt", 1_000_000_000)
    _write_scenario_yaml(
        tmp_path,
        "pooled/td3_swept",
        """
training:
  algorithm: TD3
  max_steps: 3000000
""",
    )
    results_root = tmp_path / "thesis-results"

    args = _args(eval_dir, results_root, allow_stale=False)
    args.scenario = "pooled/td3_swept"
    args.output_dir = None
    monkeypatch.setattr(export_eval_to_thesis, "_parse_args", lambda: args)
    monkeypatch.setattr(export_eval_to_thesis, "_repo_root", lambda: tmp_path)

    assert export_eval_to_thesis.main() == 0

    hp = json.loads(
        (
            results_root / "test_experiment" / "latest_finished" / "hyperparams.json"
        ).read_text()
    )
    # max_steps reports what ran; the YAML figure survives alongside it.
    assert hp["max_steps"] == 500_000
    assert hp["trained_steps"] == 500_000
    assert hp["max_steps_configured"] == 3_000_000
