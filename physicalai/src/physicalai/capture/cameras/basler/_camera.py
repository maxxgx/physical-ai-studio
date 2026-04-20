# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import contextlib
import time
from typing import TYPE_CHECKING, Any

from loguru import logger
from pypylon import genicam, pylon

from physicalai.capture.camera import Camera, ColorMode
from physicalai.capture.errors import CaptureError, CaptureTimeoutError, NotConnectedError
from physicalai.capture.frame import Frame

if TYPE_CHECKING:
    import numpy as np

    from physicalai.capture.discovery import DeviceInfo


class BaslerCamera(Camera):
    """Basler camera using pypylon SDK."""

    def __init__(
        self,
        *,
        serial_number: str,
        fps: int = 30,
        width: int = 640,
        height: int = 480,
        color_mode: ColorMode = ColorMode.RGB,
    ) -> None:
        super().__init__(color_mode=color_mode)
        self._serial_number = serial_number
        self._fps = fps
        self._width = width
        self._height = height
        self._connected = False
        self._sequence = 0
        self._last_timestamp: float = 0.0
        self._camera: Any | None = None
        self._converter: Any | None = None
        self._last_frame_data: np.ndarray | None = None

    def connect(self, timeout: float = 5.0) -> None:  # noqa: PLR0915 (fix pls)
        factory = pylon.TlFactory.GetInstance()

        # Find the device by serial number using full enumeration.  For GigE
        # cameras this ensures CreateDevice receives the complete transport-
        # layer context (IP, MAC, etc.) which is required for streaming.
        dev_info = None
        for di in factory.EnumerateDevices():
            if di.GetSerialNumber() == self._serial_number:
                dev_info = di
                break

        if dev_info is None:
            msg = f"Basler camera with serial {self._serial_number} not found"
            raise CaptureError(msg)

        self._camera = pylon.InstantCamera(factory.CreateDevice(dev_info))
        self._camera.Open()
        self._camera.Width.Value = self._width
        self._camera.Height.Value = self._height
        try:
            self._camera.AcquisitionFrameRateEnable.Value = True
            self._camera.AcquisitionFrameRate.Value = float(self._fps)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Error setting FPS for basler camera {self._serial_number}: {exc}")

        self._converter = pylon.ImageFormatConverter()
        if self._color_mode == ColorMode.RGB:
            self._converter.OutputPixelFormat = pylon.PixelType_RGB8packed
        elif self._color_mode == ColorMode.BGR:
            self._converter.OutputPixelFormat = pylon.PixelType_BGR8packed
        else:
            self._converter.OutputPixelFormat = pylon.PixelType_Mono8

        self._camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

        deadline = time.monotonic() + timeout
        last_error = ""
        while time.monotonic() < deadline:
            remaining_ms = max(1, int((deadline - time.monotonic()) * 1000))
            try:
                grab_result = self._camera.RetrieveResult(remaining_ms, pylon.TimeoutHandling_ThrowException)
            except genicam.TimeoutException as err:
                self._do_disconnect()
                msg = f"Timed out waiting for first frame after {timeout}s"
                raise CaptureTimeoutError(msg) from err
            except Exception as err:
                self._do_disconnect()
                msg = "Failed to start Basler camera"
                raise CaptureError(msg) from err

            if grab_result.GrabSucceeded():
                converted = self._converter.Convert(grab_result)
                self._last_frame_data = converted.GetArray().copy()
                grab_result.Release()
                self._connected = True
                self._sequence = 0
                return

            last_error = grab_result.GetErrorDescription()
            grab_result.Release()

        self._do_disconnect()
        msg = f"First grab failed: {last_error}"
        raise CaptureError(msg)

    def _do_disconnect(self) -> None:
        if self._camera is not None:
            with contextlib.suppress(Exception):
                self._camera.StopGrabbing()
            with contextlib.suppress(Exception):
                self._camera.Close()
        self._camera = None
        self._converter = None
        self._last_frame_data = None
        self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def device_id(self) -> str:
        return f"basler:{self._serial_number}"

    def read(self, timeout: float | None = None) -> Frame:
        if not self._connected or self._camera is None or self._converter is None:
            raise NotConnectedError

        timeout_s = timeout if timeout is not None else 5.0
        deadline = time.monotonic() + timeout_s
        last_error = ""

        # GigE cameras may produce incomplete frames due to UDP packet loss.
        # Retry until we get a complete frame or the timeout expires.
        while time.monotonic() < deadline:
            remaining_ms = max(1, int((deadline - time.monotonic()) * 1000))
            try:
                grab_result = self._camera.RetrieveResult(remaining_ms, pylon.TimeoutHandling_ThrowException)
            except genicam.TimeoutException as err:
                msg = f"Timed out waiting for frame after {timeout_s}s"
                raise CaptureTimeoutError(msg) from err

            if grab_result.GrabSucceeded():
                converted = self._converter.Convert(grab_result)
                data = converted.GetArray().copy()
                grab_result.Release()

                self._last_frame_data = data
                self._sequence += 1
                self._last_timestamp = time.monotonic()
                return Frame(data=data, timestamp=self._last_timestamp, sequence=self._sequence)

            last_error = grab_result.GetErrorDescription()
            grab_result.Release()

        msg = f"Grab failed: {last_error}"
        raise CaptureError(msg)

    def read_latest(self) -> Frame:
        if not self._connected or self._camera is None or self._converter is None:
            raise NotConnectedError

        grab_result = self._camera.RetrieveResult(0, pylon.TimeoutHandling_Return)

        if grab_result is not None and grab_result.IsValid():
            if grab_result.GrabSucceeded():
                converted = self._converter.Convert(grab_result)
                data = converted.GetArray().copy()
                grab_result.Release()
                self._last_frame_data = data
                self._sequence += 1
                self._last_timestamp = time.monotonic()
                return Frame(data=data, timestamp=self._last_timestamp, sequence=self._sequence)
            grab_result.Release()

        if self._last_frame_data is not None:
            return Frame(data=self._last_frame_data, timestamp=self._last_timestamp, sequence=self._sequence)
        msg = "No frame available"
        raise CaptureError(msg)

    @classmethod
    def discover(cls) -> list[DeviceInfo]:
        from ._discover import discover_basler  # noqa: PLC0415

        return discover_basler()
