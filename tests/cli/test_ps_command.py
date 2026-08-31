"""Tests for the ps/attach CLI commands (live trainer IPC introspection)."""

from __future__ import annotations

import shutil
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest
import typer
from rich.console import Console

from cli.commands import ps_command as ps_command_module
from cli.commands.ps_command import AttachCommand, AttachParams, PsCommand, PsParams
from trading_rl import ipc as ipc_module
from trading_rl.ipc import IpcServer


@pytest.fixture(autouse=True)
def _isolated_ipc_dir(monkeypatch):
    """Short /tmp-rooted IPC_DIR; see tests/test_ipc.py for why not tmp_path.

    Patched in *two* places: ``ipc_module.IPC_DIR`` (read by functions
    defined in trading_rl.ipc, e.g. list_registered/IpcServer) and
    ``ps_command_module.IPC_DIR`` (ps_command.py did `from trading_rl.ipc
    import IPC_DIR`, a value import that copies the reference at import
    time -- patching only the origin module would leave that copy pointing
    at the real /tmp/thesis_rl_ipc).
    """
    test_dir = Path(f"/tmp/thesis_rl_ipc_test_{uuid.uuid4().hex[:8]}")  # noqa: S108
    monkeypatch.setattr(ipc_module, "IPC_DIR", test_dir)
    monkeypatch.setattr(ps_command_module, "IPC_DIR", test_dir)
    yield test_dir
    shutil.rmtree(test_dir, ignore_errors=True)


def _console() -> Console:
    return Console(record=True, width=200)


def _running_server(
    run_id: str, *, algorithm: str = "TD3", label: str = "run-label"
) -> IpcServer:
    server = IpcServer(
        SimpleNamespace(
            config=SimpleNamespace(algorithm=algorithm), checkpoint_prefix=label
        ),
        run_id=run_id,
    )
    server.start()
    return server


class TestPsCommand:
    def test_no_registered_processes_prints_hint_and_exits_zero(self):
        console = _console()
        command = PsCommand(console)

        with pytest.raises(typer.Exit) as exc_info:
            command.execute(PsParams())

        assert exc_info.value.exit_code == 0
        assert "No registered processes" in console.export_text()

    def test_registered_process_appears_in_the_table(self):
        server = _running_server("abcdef0001", algorithm="TD3", label="my-run")
        try:
            console = _console()
            PsCommand(console).execute(PsParams())

            output = console.export_text()
            assert "abcdef0001" in output
            assert "TD3" in output
            assert "my-run" in output
        finally:
            server.stop()


class TestAttachResolveClient:
    def test_resolves_by_exact_run_id(self):
        server = _running_server("exact00001")
        try:
            client = AttachCommand(_console())._resolve_client("exact00001")
            assert client.status()["pid"] is not None
        finally:
            server.stop()

    def test_resolves_by_run_id_prefix(self):
        server = _running_server("prefixabcd")
        try:
            client = AttachCommand(_console())._resolve_client("prefix")
            assert client.status()["pid"] is not None
        finally:
            server.stop()

    def test_resolves_by_label_substring_case_insensitive(self):
        server = _running_server("labelrun01", label="MyExperiment")
        try:
            client = AttachCommand(_console())._resolve_client("myexperiment")
            assert client.status()["pid"] is not None
        finally:
            server.stop()

    def test_ambiguous_prefix_exits_with_error(self):
        s1 = _running_server("dupe000001")
        s2 = _running_server("dupe000002")
        try:
            console = _console()
            with pytest.raises(typer.Exit) as exc_info:
                AttachCommand(console)._resolve_client("dupe")

            assert exc_info.value.exit_code == 1
            assert "Ambiguous" in console.export_text()
        finally:
            s1.stop()
            s2.stop()

    def test_no_match_exits_with_error(self):
        console = _console()
        with pytest.raises(typer.Exit) as exc_info:
            AttachCommand(console)._resolve_client("does-not-exist")

        assert exc_info.value.exit_code == 1
        assert "No registered process matching" in console.export_text()


class TestAttachExecute:
    def test_execute_without_watch_prints_status_once(self):
        server = _running_server("execone001", algorithm="PPO")
        try:
            console = _console()
            AttachCommand(console).execute(AttachParams(run_id="execone001"))

            output = console.export_text()
            assert "algorithm" in output
            assert "PPO" in output
        finally:
            server.stop()

    def test_execute_with_path_prints_only_that_value(self):
        server = _running_server("execpath01", algorithm="SAC")
        try:
            console = _console()
            AttachCommand(console).execute(
                AttachParams(run_id="execpath01", path="config.algorithm")
            )

            output = console.export_text()
            assert "config.algorithm" in output
            assert "SAC" in output
        finally:
            server.stop()

    def test_execute_against_no_match_exits_rather_than_crashing(self):
        # No socket file at all for this run_id: _resolve_client can't even
        # find a candidate, so this is the typer.Exit(1) path, not
        # execute()'s handle_error path (covered next).
        console = _console()

        with pytest.raises(typer.Exit):
            AttachCommand(console).execute(AttachParams(run_id="never-existed"))

    def test_execute_against_orphaned_socket_file_reports_failure_not_crash(self):
        # Simulates a crashed trainer that left its socket path behind (a
        # regular file, not a live listener): _resolve_client's exact-match
        # branch only checks existence, so it finds this (no "no match"
        # typer.Exit), but connecting to a non-socket file fails at the OS
        # level -- execute()'s try/except must translate that into
        # BaseCommand's standard handle_error path (print + typer.Exit(1))
        # rather than letting a raw OSError propagate uncaught.
        ipc_module.IPC_DIR.mkdir(parents=True, exist_ok=True)
        sock_path = ipc_module.IPC_DIR / "orphaned001.sock"
        sock_path.touch()
        try:
            console = _console()

            with pytest.raises(typer.Exit) as exc_info:
                AttachCommand(console).execute(AttachParams(run_id="orphaned001"))

            assert exc_info.value.exit_code == 1
            assert "attach failed" in console.export_text()
        finally:
            sock_path.unlink(missing_ok=True)
