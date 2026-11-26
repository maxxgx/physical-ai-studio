"""Which file a process worker's records land in."""

from __future__ import annotations

import asyncio
import json
import os
import signal
import stat
import subprocess
import sys
import threading
import time
from typing import TYPE_CHECKING

import psutil
import pytest
from loguru import logger
from tests.workers.fakes import PACED_HZ, FakeProcessWorker

from core.logging.log_config import LogConfig
from core.logging.setup import setup_logging
from workers.model_worker import ModelWorker
from workers.training_worker import TrainingWorker

if TYPE_CHECKING:
    from collections.abc import Generator
    from pathlib import Path


@pytest.fixture
def log_folder(tmp_path: Path) -> Generator[Path]:
    yield tmp_path
    logger.remove()
    logger.add(sys.stderr)


def _lines(path: Path) -> list[dict]:
    """The complete JSON lines of ``path``; a line still being written is skipped."""
    lines = []
    for line in path.read_text().splitlines() if path.exists() else []:
        try:
            lines.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return lines


async def _until(condition, within: float = 20.0) -> None:
    deadline = time.monotonic() + within
    while not condition():
        if time.monotonic() > deadline:
            raise TimeoutError
        await asyncio.sleep(0.05)


def _own_file(folder: Path, name: str, pid: int | None) -> Path:
    """The current file of worker ``pid`` in ``folder``: ``<name>-<pid>-<start>.log``."""
    matches = sorted(folder.glob(f"{name}-{pid}-[0-9]*[0-9].log"))
    return matches[0] if matches else folder / f"{name}-{pid}-missing.log"


def _worker_file(storage: Path, worker: FakeProcessWorker) -> Path:
    return _own_file(storage / "logs" / "workers", "FakeProcessWorker", worker.pid)


def _markers(storage: Path, name: str) -> Path:
    path = storage / name
    path.mkdir()
    return path


# In a real child process, configured from STORAGE_DIR as in Studio.


async def test_a_worker_processs_failure_lands_in_its_own_file(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("STORAGE_DIR", str(tmp_path))

    async with FakeProcessWorker(_markers(tmp_path, "markers"), mode="fail_run") as worker:
        await worker.wait_stopped()

    lines = _lines(_worker_file(tmp_path, worker))
    [failure] = [line for line in lines if line["record"]["message"] == "Worker fake-process-fail_run failed"]
    assert failure["record"]["extra"]["worker"] == "FakeProcessWorker"
    assert "ValueError: lost device" in failure["text"]
    assert {line["record"]["process"]["id"] for line in lines} == {worker.pid}


async def test_two_workers_of_one_kind_write_separate_files(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("STORAGE_DIR", str(tmp_path))
    first = FakeProcessWorker(_markers(tmp_path, "first"), mode="ticking")
    second = FakeProcessWorker(_markers(tmp_path, "second"), mode="ticking")

    async with first, second:
        await _until(lambda: all(len(_lines(_worker_file(tmp_path, w))) > 5 for w in (first, second)))

    for worker in (first, second):
        assert {line["record"]["process"]["id"] for line in _lines(_worker_file(tmp_path, worker))} == {worker.pid}


async def test_killing_one_worker_leaves_the_others_logging(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("STORAGE_DIR", str(tmp_path))
    killed = FakeProcessWorker(_markers(tmp_path, "killed"), mode="ticking")
    survivor = FakeProcessWorker(_markers(tmp_path, "survivor"), mode="ticking")

    async with survivor:
        async with killed:
            await _until(lambda: len(_lines(_worker_file(tmp_path, killed))) > 5)
            os.kill(killed.pid, signal.SIGKILL)  # type: ignore[arg-type]
            await killed.wait_stopped()
        at_kill = len(_lines(_worker_file(tmp_path, survivor)))
        await _until(lambda: len(_lines(_worker_file(tmp_path, survivor))) > at_kill + 20)

    messages = [line["record"]["message"] for line in _lines(_worker_file(tmp_path, survivor))]
    assert messages[-1] == "Stopped worker fake-process-ticking"
    assert len(_lines(_worker_file(tmp_path, killed))) > 5


NO_SUCH_PID = 900_000
"""Above every platform's pid limit, so no process has it."""


def _make_worker_file(folder: Path, name: str, age_s: float) -> Path:
    path = folder / name
    path.write_text("{}\n")
    then = time.time() - age_s
    os.utime(path, (then, then))
    return path


def test_the_backend_removes_old_worker_files_at_startup(log_folder: Path) -> None:
    workers = log_folder / "workers"
    workers.mkdir()
    old = _make_worker_file(workers, f"ModelWorker-{NO_SUCH_PID}-1700000000.log", 11 * 86400)
    recent = [
        _make_worker_file(workers, f"RobotWorker-{NO_SUCH_PID + 1 + index}-1700000000.log", index)
        for index in range(105)
    ]

    setup_logging(LogConfig(log_folder=log_folder))

    assert not old.exists()
    assert [path.exists() for path in recent] == [True] * 100 + [False] * 5


def test_a_restart_never_removes_a_file_still_being_written(log_folder: Path) -> None:
    """A worker left over from an earlier backend keeps its file, however old or many the files are."""
    workers = log_folder / "workers"
    workers.mkdir()
    writer = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        started = int(psutil.Process(writer.pid).create_time())
        live = _make_worker_file(workers, f"ModelWorker-{writer.pid}-{started}.log", 11 * 86400)
        live_rotated = _make_worker_file(
            workers, f"ModelWorker-{writer.pid}-{started}.2026-11-01_10-00-00_000000.log", 12 * 86400
        )
        dead = _make_worker_file(workers, f"ModelWorker-{NO_SUCH_PID}-{started}.log", 11 * 86400)
        # The same pid with another start time is an earlier process whose pid was reused.
        reused = _make_worker_file(workers, f"ModelWorker-{writer.pid}-{started - 3600}.log", 11 * 86400)

        setup_logging(LogConfig(log_folder=log_folder, worker_files_kept=0))

        assert live.exists()
        assert live_rotated.exists()
        assert not dead.exists()
        assert not reused.exists()
    finally:
        writer.kill()
        writer.wait()


async def test_worker_files_stay_bounded_while_the_backend_runs(tmp_path: Path, monkeypatch) -> None:
    """Each worker start prunes, so a long-running backend does not pile up worker files."""
    monkeypatch.setenv("STORAGE_DIR", str(tmp_path))
    setup_logging(LogConfig(log_folder=tmp_path / "logs", worker_files_kept=3))
    try:
        for index in range(6):
            async with FakeProcessWorker(_markers(tmp_path, f"run-{index}"), mode="ticking") as worker:
                await _until(lambda w=worker: len(_lines(_worker_file(tmp_path, w))) > 0)
            files = list((tmp_path / "logs" / "workers").glob("*.log"))
            # The 3 newest of the earlier ones, plus the one this worker just wrote.
            assert len(files) <= 4, files
        assert _worker_file(tmp_path, worker).exists()
    finally:
        logger.remove()
        logger.add(sys.stderr)


# In this process, through setup_logging(worker=...).


def test_a_worker_process_writes_its_records_with_a_plain_traceback(log_folder: Path) -> None:
    setup_logging(LogConfig(log_folder=log_folder, serialize=True, level="INFO"), worker=ModelWorker)

    with logger.contextualize(worker="ModelWorker"):
        try:
            raise RuntimeError("the export does not load")
        except RuntimeError:
            logger.exception("Worker failed")
    logger.complete()

    [failure] = _lines(_own_file(log_folder / "workers", "ModelWorker", os.getpid()))
    traceback_text = failure["record"]["extra"]["traceback"]
    assert traceback_text.startswith("Traceback (most recent call last):")
    assert traceback_text.rstrip().endswith("RuntimeError: the export does not load")


def test_untagged_records_of_a_worker_process_stay_in_app_log(log_folder: Path) -> None:
    """Threads the worker starts itself (a transport callback) log without the worker tag, as on main."""
    setup_logging(LogConfig(log_folder=log_folder, serialize=True, level="INFO"), worker=ModelWorker)

    with logger.contextualize(worker="ModelWorker"):
        logger.info("from the worker")
    threading.Thread(target=lambda: logger.error("from a callback thread")).start()
    time.sleep(0.2)
    logger.complete()

    worker_file = _own_file(log_folder / "workers", "ModelWorker", os.getpid())
    assert [line["record"]["message"] for line in _lines(worker_file)] == ["from the worker"]
    assert [line["record"]["message"] for line in _lines(log_folder / "app.log")] == ["from a callback thread"]


def test_job_workers_keep_their_shared_files(log_folder: Path) -> None:
    setup_logging(LogConfig(log_folder=log_folder, serialize=True, level="INFO"), worker=TrainingWorker)

    with logger.contextualize(worker="TrainingWorker"):
        logger.info("epoch 1")
    logger.complete()

    assert [line["record"]["message"] for line in _lines(log_folder / "training.log")] == ["epoch 1"]
    assert not (log_folder / "workers").exists()


def test_flushing_before_exit_gives_up_on_a_blocked_sink() -> None:
    """A sink stuck on a stalled disk must not hold a worker, job workers included, past the flush limit."""
    # Imported here, not at the top, so the other tests in this file still run on main.
    from core.logging.setup import flush_logs

    release = threading.Event()
    sink = logger.add(lambda _message: release.wait(30), enqueue=True)
    try:
        logger.info("stuck behind a stalled write")
        started = time.monotonic()

        assert flush_logs(within=0.5) is False
        assert time.monotonic() - started < 2.0
    finally:
        release.set()
        logger.remove(sink)


def test_a_worker_file_hides_local_values_and_is_private(log_folder: Path) -> None:
    setup_logging(LogConfig(log_folder=log_folder, serialize=True, level="INFO"), worker=ModelWorker)

    def connect() -> None:
        api_token = "s3cr3t-token-value"  # noqa: F841 - a local a traceback must not show
        raise RuntimeError("connection refused")

    with logger.contextualize(worker="ModelWorker"):
        try:
            connect()
        except RuntimeError:
            logger.exception("Worker failed")

    worker_file = _own_file(log_folder / "workers", "ModelWorker", os.getpid())
    content = worker_file.read_text()
    assert "RuntimeError: connection refused" in content
    assert "s3cr3t-token-value" not in content
    assert stat.S_IMODE(worker_file.stat().st_mode) == 0o600


def test_a_running_worker_keeps_a_bounded_number_of_rotated_files(log_folder: Path) -> None:
    config = LogConfig(
        log_folder=log_folder, serialize=True, level="INFO", rotation="2 KB", worker_rotated_files_kept=2
    )
    setup_logging(config, worker=ModelWorker)

    with logger.contextualize(worker="ModelWorker"):
        for index in range(200):
            logger.info("record {}", index)

    files = list((log_folder / "workers").glob(f"ModelWorker-{os.getpid()}-*.log"))
    assert 2 < len(files) <= 3, files  # it rotated, and keeps its current file plus 2


async def test_a_slow_disk_does_not_slow_a_workers_loop(tmp_path: Path, monkeypatch) -> None:
    """A worker logging every tick keeps its rate when each file write takes 5 ms (its loop period is 20 ms)."""
    monkeypatch.setenv("STORAGE_DIR", str(tmp_path))
    markers = _markers(tmp_path, "markers")

    async with FakeProcessWorker(markers, mode="paced", sink_delay_s=0.005):
        await asyncio.sleep(2.0)

    assert float((markers / "rate").read_text()) >= 0.9 * PACED_HZ


async def test_a_worker_stuck_writing_its_log_still_ends_within_its_stop_timeout(tmp_path: Path, monkeypatch) -> None:
    """The sink is never released: the worker's last write hangs, and the parent ends it at its stop timeout."""
    monkeypatch.setenv("STORAGE_DIR", str(tmp_path))
    worker = FakeProcessWorker(_markers(tmp_path, "markers"), mode="stuck_sink", stop_timeout_s=1.0)

    async with worker:
        await asyncio.sleep(0.5)
        stopping = time.monotonic()

    assert time.monotonic() - stopping < 5.0
    assert not worker.is_alive()
