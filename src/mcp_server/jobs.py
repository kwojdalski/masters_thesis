"""Background execution for CLI commands that can run for minutes to hours.

Training/evaluation/experiment runs are too slow for a blocking MCP tool call
(clients typically enforce call timeouts). Instead, a `*_start` tool launches
the command on a daemon thread and returns a job id; `job_status` / `job_logs`
poll it.
"""

from __future__ import annotations

import io
import threading
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from rich.console import Console


class JobStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Job:
    id: str
    kind: str
    console: Console
    buffer: io.StringIO
    status: JobStatus = JobStatus.PENDING
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    started_at: datetime | None = None
    finished_at: datetime | None = None
    outcome: dict[str, Any] | None = None
    error: str | None = None

    def logs(self) -> str:
        return self.buffer.getvalue()

    def summary(self) -> dict[str, Any]:
        return {
            "job_id": self.id,
            "kind": self.kind,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
            "error": self.error,
            "result": (self.outcome or {}).get("result") if self.outcome else None,
        }


class JobManager:
    """Runs CLI commands on background threads so MCP tool calls return immediately."""

    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def start(
        self, kind: str, run: Callable[[Console, io.StringIO], dict[str, Any]]
    ) -> str:
        job_id = uuid.uuid4().hex[:12]
        buffer = io.StringIO()
        console = Console(
            file=buffer, force_terminal=False, no_color=True, width=100, highlight=False
        )
        job = Job(id=job_id, kind=kind, console=console, buffer=buffer)
        with self._lock:
            self._jobs[job_id] = job

        def _target() -> None:
            job.status = JobStatus.RUNNING
            job.started_at = datetime.now(UTC)
            try:
                job.outcome = run(job.console, job.buffer)
                job.status = (
                    JobStatus.COMPLETED if job.outcome.get("ok") else JobStatus.FAILED
                )
            except Exception as exc:
                job.error = str(exc)
                job.status = JobStatus.FAILED
            finally:
                job.finished_at = datetime.now(UTC)

        threading.Thread(target=_target, name=f"mcp-job-{job_id}", daemon=True).start()
        return job_id

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def list(self) -> list[Job]:
        with self._lock:
            return sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)


JOBS = JobManager()
