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


def test_main_exports_per_symbol_statistical_tests_without_averaging(
    tmp_path: Path, monkeypatch
) -> None:
    eval_dir = tmp_path / "logs" / "experiment"
    eval_dir.mkdir(parents=True)
    test_result = {
        "enabled": True,
        "tests_configured": ["t_test"],
        "baselines": [
            {
                "baseline": "buy_and_hold",
                "t_test": {"p_value": 0.01, "significant": True},
            }
        ],
    }
    results_file = eval_dir / "results.json"
    results_file.write_text(
        json.dumps(
            {
                "test_AAPL": {
                    "metrics": {"total_return": 0.1},
                    "statistical_tests": test_result,
                },
                "test_MSFT": {
                    "metrics": {"total_return": 0.2},
                    "statistical_tests": {
                        **test_result,
                        "baselines": [
                            {
                                "baseline": "buy_and_hold",
                                "t_test": {"p_value": 0.03, "significant": True},
                            }
                        ],
                    },
                },
            }
        )
    )
    results_root = tmp_path / "thesis-results"
    monkeypatch.setattr(
        export_eval_to_thesis,
        "_parse_args",
        lambda: _args(eval_dir, results_root, allow_stale=False),
    )

    assert export_eval_to_thesis.main() == 0

    snapshot_dir = results_root / "test_experiment" / "latest_finished"
    exported = json.loads((snapshot_dir / "statistical_tests.json").read_text())
    assert exported["__source_split__"] == "test"
    assert exported["__source_keys__"] == ["test_AAPL", "test_MSFT"]
    assert [(row["symbol"], row["t_test"]["p_value"]) for row in exported["baselines"]] == [
        ("AAPL", 0.01),
        ("MSFT", 0.03),
    ]

    run_json = json.loads((snapshot_dir / "run.json").read_text())
    assert run_json["files"]["statistical_tests"] == "statistical_tests.json"
    assert run_json["files"]["benchmark_comparison"] is None


def test_main_does_not_label_benchmark_only_payload_as_statistical_tests(
    tmp_path: Path, monkeypatch
) -> None:
    eval_dir = tmp_path / "logs" / "experiment"
    eval_dir.mkdir(parents=True)
    (eval_dir / "results.json").write_text(
        json.dumps({"test_AAPL": {"metrics": {"total_return": 0.1}}})
    )
    (eval_dir / "test_AAPL_benchmark_table.json").write_text(
        json.dumps(
            {
                "rows": [
                    {"name": "agent", "is_strategy": True, "total_return": 0.1},
                    {
                        "name": "buy_and_hold",
                        "is_strategy": False,
                        "total_return": 0.01,
                    },
                ]
            }
        )
    )
    results_root = tmp_path / "thesis-results"
    monkeypatch.setattr(
        export_eval_to_thesis,
        "_parse_args",
        lambda: _args(eval_dir, results_root, allow_stale=False),
    )

    assert export_eval_to_thesis.main() == 0

    snapshot_dir = results_root / "test_experiment" / "latest_finished"
    assert (snapshot_dir / "benchmark_comparison.json").exists()
    assert not (snapshot_dir / "statistical_tests.json").exists()
    run_json = json.loads((snapshot_dir / "run.json").read_text())
    assert run_json["files"]["benchmark_comparison"] == "benchmark_comparison.json"
    assert run_json["files"]["statistical_tests"] is None
