# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Logging configuration and utilities for the application.

Provides centralized logging using loguru with:
- Worker-specific log files (training, import/export, inference, teleoperate)
- Job-specific temporary log files via context managers
- Automatic log rotation (10MB) and retention (10 days)
- JSON serialization and thread-safe async logging
"""

import logging
import os
import pathlib
import re
import sys
import threading
import time
import traceback
from typing import TYPE_CHECKING

import psutil
from loguru import logger

from core.logging.handlers import InterceptHandler
from core.logging.log_config import LogConfig

if TYPE_CHECKING:
    from loguru import Record

global_log_config = LogConfig()

# Folder, under the log folder, of the files that worker processes write.
WORKERS_DIR = "workers"

# Each logs one line per SSH channel/HTTP request at INFO. Raising the
# threshold to WARNING drops those routine lines and keeps actual problems.
_NOISY_LOGGERS = ("asyncssh", "httpx", "httpcore")


def _quiet_noisy_loggers() -> None:
    """Raise the level of chatty third-party loggers to reduce log noise."""
    for logger_name in _NOISY_LOGGERS:
        logging.getLogger(logger_name).setLevel(logging.WARNING)


def setup_logging(config: LogConfig | None = None, *, worker: type | None = None) -> None:
    """Configure application-wide logging with worker-specific log files.

    Creates separate log files for each worker type (training, inference, etc.) with
    configurable rotation and retention. Logs are filtered by the 'worker' field in
    record extras and serialized as JSON.

    A process worker of a class not in ``worker_log_info`` (ModelWorker, RobotWorker,
    DatasetWorker, ...) passes its class and writes only its own file,
    ``workers/<Class>-<pid>-<start>.log``, with every Loguru record of the process and its
    tracebacks. One process writes each such file, so rotating it is safe, and a worker
    that dies cannot hold up anyone else's logging. Records from the standard
    ``logging`` module are not captured there; they still go to stderr only.

    Args:
        config: Optional LogConfig instance. If None, uses default configuration.
        worker: The process worker class this process runs; None for the backend, and
            for job processes, which keep the shared files above.

    Important: Must be called in each child process (ProcessWorker does this
    automatically) and once at main process startup. Loguru sinks don't transfer
    to child processes.
    """
    global global_log_config  # noqa: PLW0603

    if config is None:
        config = LogConfig()

    # Overwrite global log_config
    global_log_config = config

    logger.remove()
    logger.configure(patcher=_add_traceback)
    logger.add(sys.stderr, level=global_log_config.level)

    _quiet_noisy_loggers()

    if worker is not None and not any(kind.__name__ in config.worker_log_info for kind in worker.__mro__):
        # The worker's own records (tagged with its class); its untagged threads still reach app.log below.
        _add_file_sink(
            config,
            os.path.join(WORKERS_DIR, f"{worker.__name__}-{os.getpid()}-{_process_start(os.getpid())}.log"),
            filter_worker=worker.__name__,
        )
    elif worker is None:
        prune_worker_files(config)

    for worker_name, log_file in global_log_config.worker_log_info.items():

        def worker_log_filter(record: "Record", name: str | None = worker_name) -> bool:
            return record["extra"].get("worker") == name

        log_path = os.path.join(config.log_folder, log_file)

        try:
            pathlib.Path(os.path.dirname(log_path)).mkdir(exist_ok=True, parents=True)
        except OSError as e:
            logger.warning(f"Failed to create log directory {log_path}: {e}")
            continue

        try:
            logger.add(
                log_path,
                rotation=config.rotation,
                retention=config.retention,
                level=config.level,
                filter=worker_log_filter,
                serialize=config.serialize,
                enqueue=True,
            )
        except Exception as e:
            logger.error(f"Failed to add log sink for {worker_name}: {e}")


def _add_traceback(record: "Record") -> None:
    """Put the exception's plain traceback in ``extra``, so a reader needs no particular text format."""
    exception = record["exception"]
    if exception is not None and exception.type is not None:
        record["extra"]["traceback"] = "".join(
            traceback.format_exception(exception.type, exception.value, exception.traceback)
        )


def _private_opener(path: str, flags: int) -> int:
    """Open a log file readable by its owner only: tracebacks can show values from driver and model code."""
    return os.open(path, flags, 0o600)


def _add_file_sink(config: LogConfig, log_file: str, filter_worker: str) -> None:
    """The worker's own file, written directly in the calling thread, like stderr.

    Not enqueued: Loguru's queue is a pipe, and a stalled writer thread lets it fill and then blocks
    every logging call, the worker's control loop included. A direct write takes as long as the disk
    takes, as the stderr sink does on main. No local variable values in tracebacks (``diagnose``).
    """
    log_path = os.path.join(config.log_folder, log_file)
    max_age_s = config.worker_files_max_age_days * 86400

    def retention(rotated: list[str]) -> None:
        # Runs at each rotation with this file's rotated copies: bounds a long-running worker's files.
        newest_first = sorted(rotated, key=os.path.getmtime, reverse=True)
        for index, path in enumerate(newest_first):
            if index >= config.worker_rotated_files_kept or time.time() - os.path.getmtime(path) > max_age_s:
                os.remove(path)

    try:
        pathlib.Path(os.path.dirname(log_path)).mkdir(exist_ok=True, parents=True)
        logger.add(
            log_path,
            rotation=config.rotation,
            retention=retention,
            level=config.level,
            filter=lambda record: record["extra"].get("worker") == filter_worker,
            serialize=config.serialize,
            enqueue=False,
            diagnose=False,
            opener=_private_opener,
        )
    except Exception as e:
        logger.error(f"Failed to add log sink {log_path}: {e}")


def flush_logs(within: float = 5.0) -> bool:
    """Wait up to ``within`` seconds for the enqueued sinks to write; return whether they did.

    A worker calls this before it exits, for the shared files it still writes through Loguru's
    queue (app.log and the job files); its own file is written directly. ``logger.complete()``
    alone waits without a limit.
    """
    flusher = threading.Thread(target=logger.complete, name="log-flush", daemon=True)
    flusher.start()
    flusher.join(within)
    return not flusher.is_alive()


def _process_start(pid: int) -> int | None:
    """When process ``pid`` started, in whole seconds since the epoch; None if there is no such process."""
    try:
        return int(psutil.Process(pid).create_time())
    except psutil.NoSuchProcess:
        return None


def _writer_of(path: pathlib.Path) -> tuple[int, int] | None:
    """The writer's pid and start time from ``<Class>-<pid>-<start>.log`` or a rotated ``....<time>.log``."""
    match = re.fullmatch(r"[^-]+-(\d+)-(\d+)(\..+)?\.log", path.name)
    return (int(match[1]), int(match[2])) if match else None


def _alive(pid: int, start: int) -> bool:
    """Whether the process that wrote the file still runs: same pid, same start time (a reused pid has another)."""
    try:
        return _process_start(pid) == start
    except psutil.AccessDenied:
        return True


def prune_worker_files(config: LogConfig | None = None) -> None:
    """Remove old worker log files: older than the age limit, then all but the newest ``worker_files_kept``.

    Files whose writer is still running are never removed (a removed file would keep being written,
    unreachable), whether it is this backend's worker or one left over from an earlier backend.
    The backend calls this at startup and each time it starts a process worker.
    """
    config = config or global_log_config
    folder = pathlib.Path(config.log_folder) / WORKERS_DIR
    if not folder.is_dir():
        return
    try:
        files = sorted(folder.glob("*.log"), key=lambda path: path.stat().st_mtime, reverse=True)
        oldest_kept = time.time() - config.worker_files_max_age_days * 86400
        for index, path in enumerate(files):
            if index < config.worker_files_kept and path.stat().st_mtime >= oldest_kept:
                continue
            writer = _writer_of(path)
            if writer is not None and _alive(*writer):
                continue
            path.unlink(missing_ok=True)
    except OSError as e:
        logger.warning(f"Failed to prune worker log files in {folder}: {e}")


def setup_uvicorn_logging() -> None:
    """Configure uvicorn logging to be handled by loguru.

    Intercepts all uvicorn log messages (from uvicorn.error, uvicorn.access, etc.)
    and redirects them to loguru for unified logging output. This ensures uvicorn
    logs follow the same format and routing as application logs.
    """
    # Configure the main uvicorn logger with InterceptHandler
    uvicorn_logger = logging.getLogger("uvicorn")
    uvicorn_logger.handlers = [InterceptHandler()]
    uvicorn_logger.setLevel(logging.INFO)
    uvicorn_logger.propagate = False  # Don't propagate to root to avoid duplicate logs
    # Clear handlers from child loggers and let them propagate to parent uvicorn logger
    for logger_name in ("uvicorn.access", "uvicorn.error"):
        child_logger = logging.getLogger(logger_name)
        child_logger.handlers.clear()
        child_logger.propagate = True  # Propagate to parent uvicorn logger
