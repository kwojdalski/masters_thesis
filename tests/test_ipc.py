"""Tests for the read-only trainer IPC module (trading_rl.ipc)."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest

from trading_rl import ipc as ipc_module
from trading_rl.ipc import (
    IpcClient,
    IpcServer,
    _pid_alive,
    _resolve_path,
    _to_picklable,
    list_registered,
    status_snapshot,
)


@pytest.fixture(autouse=True)
def _isolated_ipc_dir(monkeypatch):
    """Point IPC_DIR at a throwaway *short* directory under /tmp.

    Deliberately not pytest's ``tmp_path``: that resolves to a long
    per-test path (macOS: /private/var/folders/.../pytest-of-.../...),
    which blows the ~104-byte AF_UNIX path limit the module itself works
    around by hardcoding /tmp (see ipc.py's IPC_DIR comment) -- using
    tmp_path here would fail for the same reason production code avoids it.
    """
    test_dir = Path(f"/tmp/thesis_rl_ipc_test_{uuid.uuid4().hex[:8]}")  # noqa: S108
    monkeypatch.setattr(ipc_module, "IPC_DIR", test_dir)
    yield test_dir
    shutil.rmtree(test_dir, ignore_errors=True)


class _FakeItem:
    """Mimics a 0-d tensor/array's .item() scalar-extraction protocol."""

    def __init__(self, value):
        self._value = value

    def item(self):
        return self._value


class _RaisingItem:
    """Mimics calling .item() on a non-scalar tensor, which raises."""

    def item(self):
        raise ValueError("not a scalar")


class TestToPicklable:
    def test_primitives_pass_through_unchanged(self):
        for value in (1, 1.5, "s", True, None):
            assert _to_picklable(value) is value or _to_picklable(value) == value

    def test_tensor_like_object_converted_via_item(self):
        assert _to_picklable(_FakeItem(3.14)) == 3.14

    def test_item_raising_falls_back_to_repr(self):
        result = _to_picklable(_RaisingItem())
        assert isinstance(result, str)

    def test_list_and_tuple_recurse_elementwise(self):
        assert _to_picklable([1, _FakeItem(2), "x"]) == [1, 2, "x"]
        assert _to_picklable((1, 2)) == [1, 2]

    def test_dict_recurses_over_values(self):
        assert _to_picklable({"a": _FakeItem(5), "b": 1}) == {"a": 5, "b": 1}

    def test_depth_limit_stringifies_instead_of_recursing_forever(self):
        nested = [0]
        for _ in range(10):
            nested = [nested]
        result = _to_picklable(nested)
        # Recursion is capped, so deep enough structure degrades to a string
        # rather than raising RecursionError.
        assert isinstance(result, list)

    def test_arbitrary_object_without_item_stringifies(self):
        class Plain:
            def __repr__(self):
                return "<plain>"

        assert _to_picklable(Plain()) == "<plain>"


class TestStatusSnapshot:
    def test_minimal_trainer_reports_step_and_pid_only(self):
        snapshot = status_snapshot(SimpleNamespace())

        assert snapshot["pid"] == os.getpid()
        assert snapshot["step"] is None
        assert "algorithm" not in snapshot

    def test_full_trainer_populates_every_optional_field(self):
        trainer = SimpleNamespace(
            total_count=42,
            config=SimpleNamespace(algorithm="TD3", max_steps=1_000_000),
            replay_buffer=list(range(10)),
            optimizer_actor=SimpleNamespace(param_groups=[{"lr": 0.001}]),
            optimizer_value=SimpleNamespace(param_groups=[{"lr": 0.002}]),
            exploration_module=SimpleNamespace(sigma=_FakeItem(0.1)),
            checkpoint_prefix="my-run",
        )

        snapshot = status_snapshot(trainer)

        assert snapshot["step"] == 42
        assert snapshot["algorithm"] == "TD3"
        assert snapshot["max_steps"] == 1_000_000
        assert snapshot["replay_buffer_size"] == 10
        assert snapshot["lr_actor"] == 0.001
        assert snapshot["lr_value"] == 0.002
        assert snapshot["exploration_sigma"] == 0.1
        assert snapshot["label"] == "my-run"

    def test_buffer_without_len_support_is_omitted_not_raised(self):
        trainer = SimpleNamespace(replay_buffer=object())

        snapshot = status_snapshot(trainer)

        assert "replay_buffer_size" not in snapshot


class TestResolvePath:
    def test_resolves_nested_dotted_attribute(self):
        trainer = SimpleNamespace(config=SimpleNamespace(algorithm="PPO"))

        assert _resolve_path(trainer, "config.algorithm") == "PPO"

    def test_empty_path_raises_value_error(self):
        with pytest.raises(ValueError):
            _resolve_path(SimpleNamespace(), "")

    def test_missing_attribute_raises_attribute_error(self):
        with pytest.raises(AttributeError):
            _resolve_path(SimpleNamespace(), "does_not_exist")

    def test_resolved_value_is_coerced_picklable(self):
        trainer = SimpleNamespace(sigma=_FakeItem(0.5))

        assert _resolve_path(trainer, "sigma") == 0.5


class TestPidAlive:
    def test_current_process_is_alive(self):
        assert _pid_alive(os.getpid()) is True

    def test_exited_process_is_not_alive(self):
        proc = subprocess.Popen([sys.executable, "-c", "pass"])
        proc.wait(timeout=5)

        assert _pid_alive(proc.pid) is False


class TestIpcServerClientRoundtrip:
    def _start_server(self, trainer) -> IpcServer:
        server = IpcServer(trainer, run_id="testrun01")
        server.start()
        return server

    def test_status_roundtrip_over_socket(self):
        trainer = SimpleNamespace(
            total_count=7, config=SimpleNamespace(algorithm="DDPG", max_steps=100)
        )
        server = self._start_server(trainer)
        try:
            client = IpcClient(server.sock_path)
            status = client.status()
            assert status["step"] == 7
            assert status["algorithm"] == "DDPG"
        finally:
            server.stop()

    def test_get_roundtrip_over_socket(self):
        trainer = SimpleNamespace(config=SimpleNamespace(algorithm="TD3"))
        server = self._start_server(trainer)
        try:
            client = IpcClient(server.sock_path)
            assert client.get("config.algorithm") == "TD3"
        finally:
            server.stop()

    def test_get_unknown_path_raises_runtime_error_client_side(self):
        server = self._start_server(SimpleNamespace())
        try:
            client = IpcClient(server.sock_path)
            with pytest.raises(RuntimeError):
                client.get("nonexistent")
        finally:
            server.stop()

    def test_unsupported_op_returns_error_without_crashing_server(self):
        server = self._start_server(SimpleNamespace())
        try:
            from multiprocessing.connection import Client as RawClient

            with RawClient(str(server.sock_path), family="AF_UNIX") as conn:
                conn.send({"op": "set", "path": "x", "value": 1})
                response = conn.recv()
            assert response["ok"] is False
            assert "read-only" in response["error"]

            # Server must still be responsive after a rejected request.
            assert IpcClient(server.sock_path).status()["pid"] == os.getpid()
        finally:
            server.stop()

    def test_stop_removes_socket_and_meta_files(self):
        server = self._start_server(SimpleNamespace())
        sock_path, meta_path = server.sock_path, server.meta_path
        assert sock_path.exists()
        assert meta_path.exists()

        server.stop()

        assert not sock_path.exists()
        assert not meta_path.exists()

    def test_stop_is_idempotent(self):
        server = self._start_server(SimpleNamespace())
        server.stop()
        server.stop()  # must not raise on a second call

    def test_run_id_defaults_to_a_short_opaque_hex(self):
        server = IpcServer(SimpleNamespace())
        assert len(server.run_id) == 10
        int(server.run_id, 16)  # is valid hex


class TestListRegistered:
    def test_no_ipc_dir_returns_empty_list(self):
        assert list_registered() == []

    def test_live_server_is_listed_as_alive(self):
        server = IpcServer(
            SimpleNamespace(checkpoint_prefix="lbl"), run_id="abc0000001"
        )
        server.start()
        try:
            entries = list_registered()
            assert len(entries) == 1
            assert entries[0]["run_id"] == "abc0000001"
            assert entries[0]["label"] == "lbl"
            assert entries[0]["alive"] is True
            assert entries[0]["sock_path"] == str(server.sock_path)
        finally:
            server.stop()

    def test_stale_entry_from_dead_pid_is_pruned(self):
        ipc_dir = ipc_module.IPC_DIR
        ipc_dir.mkdir(parents=True, exist_ok=True)
        meta_path = ipc_dir / "dead0000001.json"
        sock_path = ipc_dir / "dead0000001.sock"
        sock_path.touch()

        proc = subprocess.Popen([sys.executable, "-c", "pass"])
        proc.wait(timeout=5)
        meta_path.write_text(json.dumps({"run_id": "dead0000001", "pid": proc.pid}))

        entries = list_registered()

        assert entries == []
        assert not meta_path.exists()
        assert not sock_path.exists()

    def test_entry_missing_its_socket_file_is_pruned(self):
        ipc_dir = ipc_module.IPC_DIR
        ipc_dir.mkdir(parents=True, exist_ok=True)
        meta_path = ipc_dir / "nosock0000.json"
        meta_path.write_text(json.dumps({"run_id": "nosock0000", "pid": os.getpid()}))
        # deliberately no matching .sock file

        entries = list_registered()

        assert entries == []
        assert not meta_path.exists()

    def test_malformed_meta_file_is_skipped_without_raising(self):
        ipc_dir = ipc_module.IPC_DIR
        ipc_dir.mkdir(parents=True, exist_ok=True)
        (ipc_dir / "broken0000.json").write_text("{not valid json")

        assert list_registered() == []


class TestIpcServerStartStopViaConfig:
    """End-to-end check that IpcServer state converges under repeated
    start/stop cycles, matching how TrainerRuntimeHooks drives it across
    sequential in-process trials."""

    def test_repeated_start_stop_cycles_leave_no_stale_files(self):
        for i in range(3):
            server = IpcServer(SimpleNamespace(), run_id=f"cycle{i}0001")
            server.start()
            assert server.sock_path.exists()
            server.stop()
            assert not server.sock_path.exists()

        # A tiny settle window for the daemon accept-loop threads to notice
        # their listener was closed; list_registered() must reflect a clean
        # slate regardless.
        time.sleep(0.05)
        assert list_registered() == []
