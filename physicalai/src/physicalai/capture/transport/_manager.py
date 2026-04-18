# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Process-local singleton registry for shared camera publishers.

:class:`CameraManager` is a **convenience layer** — it auto-spawns a
:class:`CameraPublisher` when no existing publisher is detected for a
given service name.  Users who want explicit lifecycle control should
use :class:`CameraPublisher` and :class:`SharedCamera` directly.
"""

from __future__ import annotations

import atexit
import ctypes
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from physicalai.capture.transport._publisher import CameraPublisher
    from physicalai.capture.transport._spec import CameraSpec
    from physicalai.capture.transport._subscriber import SharedCamera

instance: CameraManager | None = None


class CameraManager:
    """Process-local singleton that manages shared camera publishers.

    Tracks locally-spawned :class:`CameraPublisher` instances and
    provides a convenience :meth:`open` that auto-spawns publishers
    when needed.

    Use :meth:`get` to obtain the singleton instance.
    """

    def __init__(self) -> None:
        self._publishers: dict[str, CameraPublisher] = {}

    # ------------------------------------------------------------------
    # Singleton access
    # ------------------------------------------------------------------

    @classmethod
    def get(cls) -> CameraManager:
        """Return the process-local singleton, creating it if needed."""
        global instance  # noqa: PLW0603
        if instance is None:
            instance = cls()
            atexit.register(instance.close_all)
        return instance

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def open(self, spec: CameraSpec, service_name: str, *, zero_copy: bool = False) -> SharedCamera:
        """Return a :class:`SharedCamera` for *service_name*.

        If no publisher is running (locally or in another process), a
        new :class:`CameraPublisher` is spawned automatically.

        Args:
            spec: Camera construction specification for the publisher.
            service_name: iceoryx2 service name.
            zero_copy: If ``True``, the returned subscriber uses
                zero-copy reads (frames are read-only).

        Returns:
            A new (unconnected) :class:`SharedCamera` subscriber.
        """
        from physicalai.capture.transport._publisher import CameraPublisher  # noqa: PLC0415, PLC2701
        from physicalai.capture.transport._subscriber import SharedCamera  # noqa: PLC0415, PLC2701

        # 1. Check local registry for an already-running publisher.
        pub = self._publishers.get(service_name)
        if pub is not None and pub.is_alive:
            return SharedCamera(service_name, zero_copy=zero_copy)

        # 2. Probe for a cross-process publisher via iceoryx2 service.
        if self._probe_service(service_name):
            logger.debug(f"Found existing publisher for {service_name} in another process")
            return SharedCamera(service_name, zero_copy=zero_copy)

        # 3. No publisher anywhere — spawn one.
        logger.info(f"Spawning publisher for {service_name}")
        pub = CameraPublisher(spec, service_name)
        try:
            pub.start()
        except Exception:
            # If start fails because another process raced us (max_publishers=1),
            # that's OK — we can still subscribe.
            if self._probe_service(service_name):
                logger.debug(f"Lost publisher race for {service_name} — using existing")
                return SharedCamera(service_name, zero_copy=zero_copy)
            raise

        self._publishers[service_name] = pub
        return SharedCamera(service_name, zero_copy=zero_copy)

    def close_all(self) -> None:
        """Release all locally-managed publisher handles.

        Does **not** send SIGTERM to publisher subprocesses — they
        self-terminate via idle timeout once all subscribers disconnect.
        Use :meth:`stop_all` to force-kill publishers immediately.
        """
        self._publishers.clear()

    def stop_all(self) -> None:
        """Send SIGTERM to all locally-managed publishers and clear the registry."""
        for name, pub in list(self._publishers.items()):
            logger.debug(f"Stopping publisher for {name}")
            try:
                pub.stop()
            except Exception:  # noqa: BLE001
                logger.warning(f"Failed to stop publisher for {name}", exc_info=True)
        self._publishers.clear()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _probe_service(service_name: str) -> bool:
        """Check if a publisher is serving *service_name*.

        Returns:
            ``True`` if a publisher is reachable, ``False`` otherwise.
        """
        try:
            import iceoryx2 as iox2  # noqa: PLC0415

            node = iox2.NodeBuilder.new().create(iox2.ServiceType.Ipc)
            try:
                svc = (
                    node.service_builder(
                        iox2.ServiceName.new(service_name),
                    )
                    .publish_subscribe(iox2.Slice[ctypes.c_uint8])
                    .open()
                )
            except Exception:  # noqa: BLE001
                return False
            else:
                del svc
                return True
            finally:
                del node
        except Exception:  # noqa: BLE001
            return False
