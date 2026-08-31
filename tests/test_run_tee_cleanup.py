"""_run_tee must reap the child and close the pipe even when Ctrl-C aborts
the read loop, so the interpreter does not exit with a live subprocess and an
unclosed BufferedReader (the ResourceWarning noise on interrupted runs)."""

from __future__ import annotations

import subprocess

import pytest

from masters_thesis import experiments


class _FakeStdout:
    def __init__(self, chunks: list[bytes], *, raise_after: int | None = None) -> None:
        self._it = iter(chunks)
        self._raise_after = raise_after
        self._count = 0
        self.closed = False

    def __iter__(self) -> _FakeStdout:
        return self

    def __next__(self) -> bytes:
        if self._raise_after is not None and self._count >= self._raise_after:
            raise KeyboardInterrupt
        self._count += 1
        return next(self._it)

    def close(self) -> None:
        self.closed = True


class _FakePopen:
    def __init__(
        self,
        *,
        chunks: list[bytes],
        raise_after: int | None = None,
        hang_on_terminate: bool = False,
    ) -> None:
        self.stdout = _FakeStdout(chunks, raise_after=raise_after)
        self.returncode: int | None = None
        self.terminated = False
        self.killed = False
        self.wait_calls = 0
        self._hang_on_terminate = hang_on_terminate

    def wait(self, timeout: float | None = None) -> int:
        self.wait_calls += 1
        if self._hang_on_terminate and not self.killed:
            raise subprocess.TimeoutExpired(cmd="fake", timeout=timeout)
        if self.returncode is None:
            self.returncode = -15 if (self.terminated or self.killed) else 0
        return self.returncode

    def terminate(self) -> None:
        self.terminated = True

    def kill(self) -> None:
        self.killed = True


def _patch_popen(monkeypatch: pytest.MonkeyPatch, fake: _FakePopen) -> None:
    monkeypatch.setattr(
        experiments.subprocess, "Popen", lambda *_a, **_k: fake, raising=True
    )


def test_run_tee_happy_path_tees_output_and_closes_pipe(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _FakePopen(chunks=[b"hello\n", b"world\n"])
    _patch_popen(monkeypatch, fake)
    log = tmp_path / "run.log"

    experiments._run_tee(["echo", "hi"], log)

    assert log.read_text() == "hello\nworld\n"
    assert fake.wait_calls >= 1
    assert fake.stdout.closed is True


def test_run_tee_terminates_and_reaps_child_on_keyboard_interrupt(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _FakePopen(chunks=[b"line1\n", b"line2\n"], raise_after=1)
    _patch_popen(monkeypatch, fake)

    with pytest.raises(KeyboardInterrupt):
        experiments._run_tee(["train"], tmp_path / "run.log")

    assert fake.terminated is True
    assert fake.wait_calls >= 1  # child was reaped, not left running
    assert fake.stdout.closed is True


def test_run_tee_kills_child_when_terminate_times_out(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _FakePopen(chunks=[b"x\n"], raise_after=0, hang_on_terminate=True)
    _patch_popen(monkeypatch, fake)

    with pytest.raises(KeyboardInterrupt):
        experiments._run_tee(["train"], tmp_path / "run.log")

    assert fake.terminated is True
    assert fake.killed is True
    assert fake.stdout.closed is True
