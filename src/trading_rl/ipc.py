"""Read-only IPC server for live introspection of a running trainer.

Started (opt-in, `training.ipc_enabled=true`) by `TrainerRuntimeHooks` and
polled by `thesis-experiments ps` / `attach`. v1 is read-only: a `status()`
view plus a dotted-path getter. Setters (mutating live hyperparameters) are
a deliberate v2 — they need a whitelist and a safe mutation point (the
per-step `maybe_run` cadence), not open `setattr` over the wire.

Transport is `multiprocessing.connection` over a Unix domain socket: stdlib,
already pickles Python dicts, no protocol to design. One socket per run at
`IPC_DIR/<run_id>.sock`, with a small JSON sidecar for discovery without
having to connect just to list what's running.
"""

from __future__ import annotations

import json
import os
import threading
import time
from datetime import UTC, datetime
from multiprocessing.connection import Client, Listener
from pathlib import Path
from typing import Any
from uuid import uuid4

from logger import get_logger

logger = get_logger(__name__)

# AF_UNIX socket paths are capped at ~104 bytes on macOS (108 on Linux).
# tempfile.gettempdir() resolves to a long per-user path on macOS
# (/var/folders/.../T, already 40+ chars) that leaves no room for a
# meaningful run_id, so this deliberately hardcodes /tmp rather than using
# it — the same workaround other tools reach for with Unix sockets on macOS.
IPC_DIR = Path("/tmp/thesis_rl_ipc")  # noqa: S108


def _to_picklable(value: Any, *, _depth: int = 0) -> Any:
    """Coerce a value into something safe to send over the wire.

    Tensors/arrays become python scalars via `.item()`; anything else that
    isn't a plain primitive/container is stringified rather than pickled
    as-is, since arbitrary training objects (envs, torch modules) are
    typically unpicklable and this is a read-only inspection tool, not an
    object-transfer mechanism.
    """
    if _depth > 3:
        return repr(value)
    if isinstance(value, int | float | str | bool | type(None)):
        return value
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return item()
        except Exception:  # noqa: S110 - expected for non-scalar .item(); falls through below
            pass
    if isinstance(value, list | tuple):
        return [_to_picklable(v, _depth=_depth + 1) for v in value]
    if isinstance(value, dict):
        return {k: _to_picklable(v, _depth=_depth + 1) for k, v in value.items()}
    return repr(value)


def status_snapshot(trainer: Any) -> dict[str, Any]:
    """Best-effort curated view of a trainer's live state.

    Tolerant by design: trainer subclasses (TD3/DDPG/PPO/SAC/Random) don't
    share every attribute, so each field is fetched independently and
    omitted rather than raising if it's missing.
    """
    snapshot: dict[str, Any] = {
        "step": getattr(trainer, "total_count", None),
        "pid": os.getpid(),
    }
    config = getattr(trainer, "config", None)
    if config is not None:
        snapshot["algorithm"] = getattr(config, "algorithm", None)
        snapshot["max_steps"] = getattr(config, "max_steps", None)

    buffer = getattr(trainer, "replay_buffer", None)
    if buffer is not None:
        try:
            snapshot["replay_buffer_size"] = len(buffer)
        except TypeError:
            pass

    for view_name, attr in (
        ("lr_actor", "optimizer_actor"),
        ("lr_value", "optimizer_value"),
    ):
        optimizer = getattr(trainer, attr, None)
        param_groups = getattr(optimizer, "param_groups", None)
        if param_groups:
            snapshot[view_name] = param_groups[0].get("lr")

    exploration = getattr(trainer, "exploration_module", None)
    sigma = getattr(exploration, "sigma", None)
    if sigma is not None:
        snapshot["exploration_sigma"] = _to_picklable(sigma)

    label = getattr(trainer, "checkpoint_prefix", None)
    if label:
        snapshot["label"] = label

    return snapshot


def _resolve_path(trainer: Any, path: str) -> Any:
    if not path:
        raise ValueError("empty path")
    obj: Any = trainer
    for part in path.split("."):
        obj = getattr(obj, part)
    return _to_picklable(obj)


class IpcServer:
    """Owns one Unix-domain-socket Listener exposing read-only trainer state."""

    def __init__(self, trainer: Any, run_id: str | None = None) -> None:
        self.trainer = trainer
        # Deliberately short: this becomes part of a Unix domain socket path
        # (~104-byte limit on macOS), which a descriptive label like
        # checkpoint_prefix would blow past. The human-readable label still
        # goes into the JSON sidecar for `ps` to display and `attach` to
        # match against.
        self.run_id = run_id or uuid4().hex[:10]
        self.sock_path = IPC_DIR / f"{self.run_id}.sock"
        self.meta_path = IPC_DIR / f"{self.run_id}.json"
        self._listener: Listener | None = None
        self._thread: threading.Thread | None = None
        self._started_at = time.time()

    def start(self) -> None:
        IPC_DIR.mkdir(parents=True, exist_ok=True)
        if self.sock_path.exists():
            self.sock_path.unlink()
        self._listener = Listener(str(self.sock_path), family="AF_UNIX")
        self.meta_path.write_text(
            json.dumps(
                {
                    "run_id": self.run_id,
                    "pid": os.getpid(),
                    "algorithm": getattr(
                        getattr(self.trainer, "config", None), "algorithm", None
                    ),
                    "label": getattr(self.trainer, "checkpoint_prefix", None),
                    "started_at": datetime.now(UTC).isoformat(),
                }
            )
        )
        self._thread = threading.Thread(
            target=self._serve, daemon=True, name=f"ipc-{self.run_id}"
        )
        self._thread.start()
        logger.debug(
            "ipc server started run_id={} socket={}", self.run_id, self.sock_path
        )

    def _serve(self) -> None:
        assert self._listener is not None
        while True:
            try:
                conn = self._listener.accept()
            except OSError:
                return  # listener closed by stop()
            try:
                self._handle_one(conn)
            except Exception:
                logger.debug("ipc request handling raised", exc_info=True)
            finally:
                conn.close()

    def _handle_one(self, conn: Any) -> None:
        try:
            request = conn.recv()
        except EOFError:
            return
        op = request.get("op")
        if op == "status":
            conn.send({"ok": True, "value": status_snapshot(self.trainer)})
        elif op == "get":
            try:
                value = _resolve_path(self.trainer, request.get("path", ""))
                conn.send({"ok": True, "value": value})
            except Exception as exc:
                conn.send({"ok": False, "error": str(exc)})
        else:
            conn.send(
                {"ok": False, "error": f"unsupported op {op!r} (v1 is read-only)"}
            )

    def stop(self) -> None:
        if self._listener is not None:
            try:
                self._listener.close()
            except OSError:
                pass
        for path in (self.sock_path, self.meta_path):
            try:
                path.unlink()
            except FileNotFoundError:
                pass
        logger.debug("ipc server stopped run_id={}", self.run_id)


class IpcClient:
    """Thin client for connecting to one run's IpcServer."""

    def __init__(self, sock_path: Path) -> None:
        self._sock_path = sock_path

    def _request(self, payload: dict[str, Any]) -> dict[str, Any]:
        with Client(str(self._sock_path), family="AF_UNIX") as conn:
            conn.send(payload)
            return conn.recv()

    def status(self) -> dict[str, Any]:
        response = self._request({"op": "status"})
        if not response.get("ok"):
            raise RuntimeError(response.get("error", "unknown IPC error"))
        return response["value"]

    def get(self, path: str) -> Any:
        response = self._request({"op": "get", "path": path})
        if not response.get("ok"):
            raise RuntimeError(response.get("error", "unknown IPC error"))
        return response["value"]


def list_registered() -> list[dict[str, Any]]:
    """List registered runs, pruning stale entries left by crashed processes."""
    if not IPC_DIR.exists():
        return []
    entries: list[dict[str, Any]] = []
    for meta_path in sorted(IPC_DIR.glob("*.json")):
        try:
            meta = json.loads(meta_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        pid = meta.get("pid")
        alive = pid is not None and _pid_alive(pid)
        sock_path = meta_path.with_suffix(".sock")
        if not alive or not sock_path.exists():
            meta_path.unlink(missing_ok=True)
            sock_path.unlink(missing_ok=True)
            continue
        meta["alive"] = alive
        meta["sock_path"] = str(sock_path)
        entries.append(meta)
    return entries


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists, owned by someone else
    return True
