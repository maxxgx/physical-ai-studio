"""Registry for managing camera workers."""

import asyncio
from collections.abc import Sequence
from types import TracebackType
from uuid import UUID

from loguru import logger

from .camera_worker import CameraWorker


class CameraReservation:
    """Lease for camera fingerprint locks held during camera transitions."""

    def __init__(self, fingerprints: Sequence[str], locks: Sequence[asyncio.Lock]) -> None:
        self._fingerprints = tuple(fingerprints)
        self._locks = tuple(locks)
        self._released = False

    async def release(self) -> None:
        """Release held fingerprint locks."""
        if self._released:
            return

        for lock in reversed(self._locks):
            if lock.locked():
                lock.release()
        self._released = True
        logger.debug(f"Released camera reservation for {self._fingerprints}")

    async def __aenter__(self) -> "CameraReservation":
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        await self.release()


class CameraWorkerRegistry:
    """Manages lifecycle of camera workers."""

    def __init__(self, max_workers: int = 10, shutdown_timeout_s: float = 10.0):
        self._workers: dict[UUID, CameraWorker] = {}
        self._lock = asyncio.Lock()
        self._fingerprint_locks: dict[str, asyncio.Lock] = {}
        self._recording_fingerprints: set[str] = set()
        self._max_workers = max_workers
        self._shutdown_timeout_s = shutdown_timeout_s

    async def _get_fingerprint_lock(self, fingerprint: str) -> asyncio.Lock:
        async with self._lock:
            return self._fingerprint_locks.setdefault(fingerprint, asyncio.Lock())

    async def _get_fingerprint_locks(self, fingerprints: Sequence[str]) -> list[asyncio.Lock]:
        async with self._lock:
            return [self._fingerprint_locks.setdefault(fingerprint, asyncio.Lock()) for fingerprint in fingerprints]

    async def reserve_cameras(self, fingerprints: Sequence[str]) -> CameraReservation:
        """Reserve cameras and shut down matching preview workers."""
        unique_fingerprints = tuple(sorted({fingerprint for fingerprint in fingerprints if fingerprint}))
        if not unique_fingerprints:
            return CameraReservation((), ())

        locks = await self._get_fingerprint_locks(unique_fingerprints)
        acquired: list[asyncio.Lock] = []

        try:
            for lock in locks:
                await lock.acquire()
                acquired.append(lock)
        except Exception:
            for lock in reversed(acquired):
                lock.release()
            raise

        try:
            async with self._lock:
                workers_to_shutdown = [
                    (worker_id, worker)
                    for worker_id, worker in self._workers.items()
                    if worker.config.fingerprint in unique_fingerprints
                ]
                for worker_id, _worker in workers_to_shutdown:
                    self._workers.pop(worker_id, None)

            tasks = [worker.shutdown() for _worker_id, worker in workers_to_shutdown]
            if tasks:
                try:
                    results = await asyncio.wait_for(
                        asyncio.gather(*tasks, return_exceptions=True),
                        timeout=self._shutdown_timeout_s,
                    )
                except TimeoutError:
                    msg = f"Some reserved camera workers did not shutdown within {self._shutdown_timeout_s}s"
                    logger.error(msg)
                    raise TimeoutError(msg) from None

                errors = [result for result in results if isinstance(result, BaseException)]
                if errors:
                    msg = f"{len(errors)} reserved camera worker(s) failed to shut down"
                    logger.error(f"{msg}: {errors}")
                    raise RuntimeError(msg) from errors[0]
        except Exception:
            for lock in reversed(acquired):
                if lock.locked():
                    lock.release()
            raise

        logger.info(f"Reserved cameras: {unique_fingerprints}")
        return CameraReservation(unique_fingerprints, locks)

    async def set_recording_cameras(self, fingerprints: Sequence[str]) -> None:
        """Track cameras owned by the recording/robot-control path."""
        async with self._lock:
            self._recording_fingerprints = {fingerprint for fingerprint in fingerprints if fingerprint}

    async def clear_recording_cameras(self) -> None:
        """Clear recording/robot-control camera ownership."""
        async with self._lock:
            self._recording_fingerprints.clear()

    async def has_recording_cameras(self) -> bool:
        """Return whether recording/robot-control owns any camera."""
        async with self._lock:
            return bool(self._recording_fingerprints)

    async def create_and_register(
        self,
        worker_id: UUID,
        worker: CameraWorker,
    ) -> None:
        """
        Create and register a new camera worker.

        Raises:
            ValueError: If worker_id already exists or max_workers exceeded.
        """
        fingerprint_lock = await self._get_fingerprint_lock(worker.config.fingerprint)
        async with fingerprint_lock, self._lock:
            if worker_id in self._workers:
                raise ValueError(f"Worker {worker_id} already exists")

            for existing in self._workers.values():
                if existing.config.fingerprint == worker.config.fingerprint:
                    raise ValueError(f"Camera '{worker.config.name}' is already streaming")

            if len(self._workers) >= self._max_workers:
                raise ValueError(f"Maximum number of workers ({self._max_workers}) reached")

            self._workers[worker_id] = worker
            logger.info(
                f"Camera worker registered: {worker_id} ({worker.config.name}). "
                f"Total: {len(self._workers)}/{self._max_workers}"
            )

    async def unregister(self, worker_id: UUID) -> None:
        """Unregister and shutdown a worker."""
        async with self._lock:
            worker = self._workers.pop(worker_id, None)

        if worker:
            try:
                await worker.shutdown()
            except Exception as e:
                logger.error(f"Error shutting down worker {worker_id}: {e}")
            logger.info(f"Camera worker unregistered: {worker_id}")

    async def get(self, worker_id: UUID) -> CameraWorker | None:
        """Get a worker by id."""
        return self._workers.get(worker_id)

    def list_all(self) -> list[CameraWorker]:
        """List all active workers."""
        return list(self._workers.values())

    def get_status_summary(self) -> dict:
        """Get summary of all worker statuses."""
        return {
            "total_workers": len(self._workers),
            "max_workers": self._max_workers,
            "workers": {
                str(worker_id): {
                    "name": worker.config.name,
                    "state": worker.state.value,
                    "error": worker.error_message,
                }
                for worker_id, worker in self._workers.items()
            },
            "recording_cameras": sorted(self._recording_fingerprints),
        }

    async def shutdown_all(self) -> None:
        """Gracefully shutdown all workers."""
        logger.info(f"Shutting down {len(self._workers)} camera workers...")

        async with self._lock:
            workers = list(self._workers.values())
            self._workers.clear()

        # Shutdown all concurrently
        tasks = [worker.shutdown() for worker in workers]

        if tasks:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*tasks, return_exceptions=True),
                    timeout=self._shutdown_timeout_s,
                )
            except TimeoutError:
                logger.error(f"Some workers did not shutdown within {self._shutdown_timeout_s}s")

        logger.info("All camera workers shut down")

    async def __aenter__(self):
        """Async context manager support."""
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Cleanup on context exit."""
        await self.shutdown_all()
