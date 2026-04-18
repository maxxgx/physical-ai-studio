# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Shared-memory camera subscriber transport based on iceoryx2."""

from __future__ import annotations

import ctypes
import time
from importlib import import_module
from typing import TYPE_CHECKING

from loguru import logger

from physicalai.capture.camera import Camera, ColorMode
from physicalai.capture.errors import CaptureTimeoutError, NotConnectedError

from ._header import decode_header, decode_rgb

if TYPE_CHECKING:
    from typing import Any

    from physicalai.capture.frame import Frame


_SERVICE_NAME_EXPECTED_PARTS = 5


class SharedCamera(Camera):
    """Camera subscriber that reads frames from shared memory via iceoryx2.

    Connects to a publisher process that owns the physical camera device.
    Multiple SharedCamera instances can subscribe to the same publisher
    for zero-copy fan-out.

    Args:
        service_name: iceoryx2 service name to subscribe to.
        color_mode: Pixel format for returned frames.
    """

    def __init__(
        self,
        service_name: str,
        *,
        color_mode: ColorMode = ColorMode.RGB,
        zero_copy: bool = False,
    ) -> None:
        super().__init__(color_mode=color_mode)
        self._service_name = service_name
        self._zero_copy = zero_copy
        self._connected = False
        self._latest: Frame | None = None
        self._held_sample: Any = None
        self._node: Any | None = None
        self._subscriber: Any | None = None
        self._listener: Any | None = None

    def connect(self, timeout: float = 5.0) -> None:
        iox2 = import_module("iceoryx2")

        self._node = iox2.NodeBuilder.new().create(iox2.ServiceType.Ipc)

        pub_sub = (
            self._node.service_builder(iox2.ServiceName.new(self._service_name))
            .publish_subscribe(iox2.Slice[ctypes.c_uint8])
            .open()
        )
        self._subscriber = pub_sub.subscriber_builder().create()

        event_svc = self._node.service_builder(iox2.ServiceName.new(f"{self._service_name}/notify")).event().open()
        self._listener = event_svc.listener_builder().create()

        logger.debug(f"Connecting SharedCamera subscriber to {self._service_name}")
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            sample = self._subscriber.receive()
            if sample is not None:
                self._latest = self._decode_sample(sample)
                self._connected = True
                return

            remaining = deadline - time.monotonic()
            if remaining > 0:
                self._listener.timed_wait_one(
                    iox2.Duration.from_secs_f64(min(remaining, 0.5)),
                )

        logger.debug(f"SharedCamera connect timeout for {self._service_name}")
        self._do_disconnect()
        msg = "no publisher responded within timeout"
        raise CaptureTimeoutError(msg)

    def read(self, timeout: float | None = None) -> Frame:
        if not self._connected or self._subscriber is None or self._listener is None:
            msg = "shared camera is not connected"
            raise NotConnectedError(msg)

        iox2 = import_module("iceoryx2")

        wait_timeout = 3600.0 if timeout is None else timeout
        event = self._listener.timed_wait_one(iox2.Duration.from_secs_f64(wait_timeout))
        if event is None:
            msg = "timed out waiting for frame"
            raise CaptureTimeoutError(msg)

        self._held_sample = None  # release previous borrow before draining
        newest_sample = None
        while True:
            sample = self._subscriber.receive()
            if sample is None:
                break
            newest_sample = sample

        if newest_sample is not None:
            self._latest = self._decode_sample(newest_sample)

        if self._latest is None:
            msg = "no frame available"
            raise CaptureTimeoutError(msg)

        return self._latest

    def read_latest(self) -> Frame:
        if not self._connected or self._subscriber is None:
            msg = "shared camera is not connected"
            raise NotConnectedError(msg)

        self._held_sample = None  # release previous borrow before draining
        newest_sample = None
        while True:
            sample = self._subscriber.receive()
            if sample is None:
                break
            newest_sample = sample

        if newest_sample is not None:
            self._latest = self._decode_sample(newest_sample)

        if self._latest is None:
            msg = "no frame available"
            raise CaptureTimeoutError(msg)

        return self._latest

    def _decode_sample(self, sample: Any) -> Frame:  # noqa: ANN401
        import ctypes as _ct  # noqa: PLC0415

        slc = sample.payload()
        buf = (_ct.c_uint8 * slc.number_of_elements).from_address(slc.data_ptr)
        header = decode_header(bytes(buf))
        if self._zero_copy:
            from ._header import decode_rgb_view  # noqa: PLC0415

            self._held_sample = sample
            return decode_rgb_view(header, memoryview(buf))
        return decode_rgb(header, bytes(buf))

    def _do_disconnect(self) -> None:
        self._held_sample = None
        self._subscriber = None
        self._listener = None
        self._node = None
        self._connected = False
        self._latest = None

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def device_id(self) -> str:
        parts = self._service_name.split("/")
        if (
            len(parts) >= _SERVICE_NAME_EXPECTED_PARTS
            and parts[0] == "physicalai"
            and parts[1] == "camera"
            and parts[4] == "frame"
        ):
            return f"{parts[2]}/{parts[3]}"
        return self._service_name
