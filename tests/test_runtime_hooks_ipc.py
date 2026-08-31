"""Tests for TrainerRuntimeHooks' IPC server lifecycle wiring."""

from __future__ import annotations

import shutil
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest

from trading_rl import ipc as ipc_module
from trading_rl.trainers.runtime_hooks import TrainerRuntimeHooks


@pytest.fixture(autouse=True)
def _isolated_ipc_dir(monkeypatch):
    """Short /tmp-rooted IPC_DIR; see tests/test_ipc.py for why not tmp_path."""
    test_dir = Path(f"/tmp/thesis_rl_ipc_test_{uuid.uuid4().hex[:8]}")  # noqa: S108
    monkeypatch.setattr(ipc_module, "IPC_DIR", test_dir)
    yield test_dir
    shutil.rmtree(test_dir, ignore_errors=True)


def _trainer(ipc_enabled: bool | None = False) -> SimpleNamespace:
    config = (
        SimpleNamespace(algorithm="TD3")
        if ipc_enabled is None
        else SimpleNamespace(algorithm="TD3", ipc_enabled=ipc_enabled)
    )
    return SimpleNamespace(config=config, checkpoint_prefix="test-run")


class TestIpcServerStartsOnInit:
    def test_ipc_enabled_true_starts_a_live_server(self):
        hooks = TrainerRuntimeHooks(_trainer(ipc_enabled=True))
        try:
            assert hooks._ipc is not None
            assert hooks._ipc.sock_path.exists()
            assert ipc_module.list_registered() != []
        finally:
            hooks.teardown()

    def test_ipc_enabled_false_starts_no_server(self):
        hooks = TrainerRuntimeHooks(_trainer(ipc_enabled=False))

        assert hooks._ipc is None
        assert ipc_module.list_registered() == []

    def test_missing_ipc_enabled_attribute_defaults_to_no_server(self):
        # Config objects that predate the ipc_enabled field (or a bare
        # SimpleNamespace in older tests) must not crash __init__.
        hooks = TrainerRuntimeHooks(_trainer(ipc_enabled=None))

        assert hooks._ipc is None

    def test_trainer_without_config_at_all_starts_no_server(self):
        hooks = TrainerRuntimeHooks(SimpleNamespace())

        assert hooks._ipc is None


class TestIpcServerStopsOnTeardown:
    def test_teardown_stops_and_clears_the_running_server(self):
        hooks = TrainerRuntimeHooks(_trainer(ipc_enabled=True))
        sock_path = hooks._ipc.sock_path

        hooks.teardown()

        assert hooks._ipc is None
        assert not sock_path.exists()
        assert ipc_module.list_registered() == []

    def test_teardown_without_a_running_server_does_not_raise(self):
        hooks = TrainerRuntimeHooks(_trainer(ipc_enabled=False))

        hooks.teardown()  # must be a silent no-op

        assert hooks._ipc is None

    def test_teardown_is_idempotent(self):
        hooks = TrainerRuntimeHooks(_trainer(ipc_enabled=True))

        hooks.teardown()
        hooks.teardown()  # second call must not raise

        assert hooks._ipc is None
