"""Fake workers for the worker base tests.

Module-level and configured through plain arguments, because a process worker
is pickled into a spawned child that imports this module. They report what
happened through marker files, which is the one channel that works the same
for a thread and a process.
"""

from __future__ import annotations

import asyncio
import os
import threading
import time
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Literal

from loguru import logger

from workers.base import ProcessWorker, ThreadWorker

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path

Mode = Literal[
    "ok", "fail_acquire", "fail_run", "ignore_stop", "exit_hard", "slow_ready", "ticking", "paced", "stuck_sink"
]

PACED_HZ = 50
"""The rate a ``paced`` worker keeps, logging one record per tick, like a control loop."""


class _FakeBehaviour:
    """Acquire a pretend device, write markers, and misbehave on request."""

    marker_dir: Path
    mode: Mode
    name: str

    def _configure(self, marker_dir: Path, mode: Mode) -> None:
        self.marker_dir = marker_dir
        self.mode = mode

    def _mark(self, marker: str, content: str = "") -> None:
        (self.marker_dir / marker).write_text(content)

    @asynccontextmanager
    async def lifecycle(self) -> AsyncIterator[str]:
        self._mark("acquired")
        try:
            if self.mode == "fail_acquire":
                raise RuntimeError("no device")
            if self.mode == "exit_hard":
                os._exit(3)
            if self.mode == "slow_ready":
                await asyncio.sleep(1.0)
            yield f"device-of-{self.name}"
        finally:
            self._mark("released")

    async def run_loop(self, context: str) -> None:
        self._mark("context", context)
        if self.mode == "paced":
            await self._run_paced()
            return
        if self.mode == "stuck_sink":
            await self.sleep(3600)  # type: ignore[attr-defined]  # returns once a stop is requested
            _block_file_writes()  # the worker's last records ("Stopped worker ...") never get written
            return
        while not self.should_stop():  # type: ignore[attr-defined]
            if self.mode == "fail_run":
                raise ValueError("lost device")
            if self.mode == "ticking":
                logger.info("tick of process {}", os.getpid())
            if self.mode == "ignore_stop":
                # Blocks the loop, like a driver call that never returns.
                time.sleep(2.0)  # noqa: ASYNC251
                return
            await asyncio.sleep(0.01)

    async def _run_paced(self) -> None:
        period = 1 / PACED_HZ
        started = time.monotonic()
        ticks = 0
        while not self.should_stop():  # type: ignore[attr-defined]
            logger.info("tick {} of process {}", ticks, os.getpid())
            ticks += 1
            delay = started + ticks * period - time.monotonic()
            if delay > 0:
                await asyncio.sleep(delay)
        self._mark("rate", f"{ticks / (time.monotonic() - started):.1f}")


def _slow_file_writes(delay_s: float) -> None:
    """Make every Loguru file write in this process take ``delay_s`` longer, like a slow disk."""
    from loguru._file_sink import FileSink

    write = FileSink.write

    def slow_write(self: FileSink, message: str) -> None:
        time.sleep(delay_s)
        write(self, message)

    FileSink.write = slow_write  # type: ignore[method-assign]


def _block_file_writes() -> None:
    """Make every later Loguru file write in this process wait forever, like a hung disk."""
    from loguru._file_sink import FileSink

    never = threading.Event()
    FileSink.write = lambda _self, _message: never.wait()  # type: ignore[method-assign]


class FakeThreadWorker(_FakeBehaviour, ThreadWorker[str]):
    def __init__(self, marker_dir: Path, mode: Mode = "ok") -> None:
        super().__init__(name=f"fake-thread-{mode}")
        self._configure(marker_dir, mode)


class FakeProcessWorker(_FakeBehaviour, ProcessWorker[str]):
    def __init__(
        self, marker_dir: Path, mode: Mode = "ok", *, sink_delay_s: float = 0.0, stop_timeout_s: float | None = None
    ) -> None:
        super().__init__(name=f"fake-process-{mode}")
        self._configure(marker_dir, mode)
        self.sink_delay_s = sink_delay_s
        if stop_timeout_s is not None:
            self.stop_timeout_s = stop_timeout_s

    def run(self) -> None:
        if self.sink_delay_s:
            _slow_file_writes(self.sink_delay_s)
        super().run()
