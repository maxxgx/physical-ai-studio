# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: S101, PLR2004

from __future__ import annotations

import importlib
import sys
from unittest import mock

import numpy as np
import pytest

from physicalai.capture.camera import ColorMode
from physicalai.capture.discovery import DeviceInfo
from physicalai.capture.errors import CaptureError, CaptureTimeoutError, NotConnectedError
from physicalai.capture.frame import Frame


@pytest.fixture
def basler_cls():  # noqa: ANN201
    mock_pylon = mock.MagicMock()
    mock_genicam = mock.MagicMock()

    class _TimeoutException(Exception):
        pass

    mock_genicam.TimeoutException = _TimeoutException

    mock_pypylon = mock.MagicMock()
    mock_pypylon.pylon = mock_pylon
    mock_pypylon.genicam = mock_genicam

    raw_array = np.zeros((480, 640, 3), dtype=np.uint8)
    mock_converted = mock.MagicMock()
    mock_converted.GetArray.return_value = raw_array

    mock_grab_result = mock.MagicMock()
    mock_grab_result.GrabSucceeded.return_value = True
    mock_grab_result.IsValid.return_value = True
    mock_grab_result.GetErrorDescription.return_value = ""

    mock_converter = mock.MagicMock()
    mock_converter.Convert.return_value = mock_converted
    mock_pylon.ImageFormatConverter.return_value = mock_converter

    mock_camera = mock.MagicMock()
    mock_camera.RetrieveResult.return_value = mock_grab_result
    mock_pylon.InstantCamera.return_value = mock_camera

    mock_dev = mock.MagicMock()
    mock_dev.GetSerialNumber.return_value = "123"
    mock_dev.GetModelName.return_value = "acA640-90uc"
    mock_dev.GetVendorName.return_value = "Basler"
    mock_dev.GetUserDefinedName.return_value = "TestCam"
    mock_dev.GetAddress.return_value = "192.168.1.100"
    mock_pylon.TlFactory.GetInstance.return_value.EnumerateDevices.return_value = [mock_dev]

    sys.modules["pypylon"] = mock_pypylon
    sys.modules["pypylon.pylon"] = mock_pylon
    sys.modules["pypylon.genicam"] = mock_genicam
    sys.modules.pop("physicalai.capture.cameras.basler._camera", None)
    sys.modules.pop("physicalai.capture.cameras.basler._discover", None)

    module = importlib.import_module("physicalai.capture.cameras.basler._camera")
    camera_cls = module.BaslerCamera

    yield camera_cls, mock_pylon, mock_genicam

    sys.modules.pop("pypylon", None)
    sys.modules.pop("pypylon.pylon", None)
    sys.modules.pop("pypylon.genicam", None)
    sys.modules.pop("physicalai.capture.cameras.basler._camera", None)
    sys.modules.pop("physicalai.capture.cameras.basler._discover", None)


def test_connect_opens_camera_and_starts_grabbing(basler_cls: tuple) -> None:
    camera_cls, mock_pylon, _ = basler_cls
    camera = camera_cls(serial_number="123")
    camera.connect()
    mock_pylon.InstantCamera.return_value.Open.assert_called_once()
    mock_pylon.InstantCamera.return_value.StartGrabbing.assert_called_once_with(
        mock_pylon.GrabStrategy_LatestImageOnly,
    )


def test_connect_uses_serial_number(basler_cls: tuple) -> None:
    camera_cls, mock_pylon, _ = basler_cls
    camera = camera_cls(serial_number="wrong-serial")
    with pytest.raises(CaptureError, match="not found"):
        camera.connect()
    mock_pylon.TlFactory.GetInstance.return_value.EnumerateDevices.assert_called()


def test_connect_verifies_first_frame(basler_cls: tuple) -> None:
    camera_cls, mock_pylon, _ = basler_cls
    camera = camera_cls(serial_number="123")
    camera.connect()
    mock_pylon.InstantCamera.return_value.RetrieveResult.assert_called()


def test_connect_timeout_raises_capture_timeout(basler_cls: tuple) -> None:
    camera_cls, mock_pylon, mock_genicam = basler_cls
    mock_pylon.InstantCamera.return_value.RetrieveResult.side_effect = mock_genicam.TimeoutException(
        "timeout",
    )
    camera = camera_cls(serial_number="123")
    with pytest.raises(CaptureTimeoutError):
        camera.connect()


def test_disconnect_stops_grabbing_and_closes(basler_cls: tuple) -> None:
    camera_cls, mock_pylon, _ = basler_cls
    camera = camera_cls(serial_number="123")
    camera.connect()
    camera.disconnect()
    mock_pylon.InstantCamera.return_value.StopGrabbing.assert_called()
    mock_pylon.InstantCamera.return_value.Close.assert_called()
    assert not camera.is_connected


def test_disconnect_without_connect_is_safe(basler_cls: tuple) -> None:
    camera_cls, _, _ = basler_cls
    camera = camera_cls(serial_number="123")
    camera.disconnect()


def test_read_returns_frame_with_correct_shape_and_dtype(basler_cls: tuple) -> None:
    camera_cls, _, _ = basler_cls
    camera = camera_cls(serial_number="123")
    camera.connect()
    frame = camera.read()
    assert isinstance(frame, Frame)
    assert frame.data.shape == (480, 640, 3)
    assert frame.data.dtype == np.uint8


def test_read_increments_sequence(basler_cls: tuple) -> None:
    camera_cls, _, _ = basler_cls
    camera = camera_cls(serial_number="123")
    camera.connect()
    f1 = camera.read()
    f2 = camera.read()
    f3 = camera.read()
    assert f1.sequence == 1
    assert f2.sequence == 2
    assert f3.sequence == 3


def test_read_timeout_raises_capture_timeout(basler_cls: tuple) -> None:
    camera_cls, mock_pylon, mock_genicam = basler_cls
    camera = camera_cls(serial_number="123")
    camera.connect()
    mock_pylon.InstantCamera.return_value.RetrieveResult.side_effect = mock_genicam.TimeoutException(
        "timeout",
    )
    with pytest.raises(CaptureTimeoutError):
        camera.read()


def test_read_grab_failed_raises_capture_error(basler_cls: tuple) -> None:
    camera_cls, mock_pylon, _ = basler_cls
    camera = camera_cls(serial_number="123")
    camera.connect()
    mock_pylon.InstantCamera.return_value.RetrieveResult.return_value.GrabSucceeded.return_value = False
    with pytest.raises(CaptureError):
        camera.read(timeout=0.01)


def test_read_latest_returns_frame(basler_cls: tuple) -> None:
    camera_cls, _, _ = basler_cls
    camera = camera_cls(serial_number="123")
    camera.connect()
    frame = camera.read_latest()
    assert isinstance(frame, Frame)


def test_read_latest_returns_cached_when_no_new_frame(basler_cls: tuple) -> None:
    camera_cls, mock_pylon, _ = basler_cls
    camera = camera_cls(serial_number="123")
    camera.connect()
    first = camera.read()
    mock_pylon.InstantCamera.return_value.RetrieveResult.return_value = None
    latest = camera.read_latest()
    assert latest.sequence == first.sequence


def test_discover_returns_device_info_list(basler_cls: tuple) -> None:
    _, _, _ = basler_cls
    sys.modules.pop("physicalai.capture.cameras.basler._discover", None)
    discover_module = importlib.import_module("physicalai.capture.cameras.basler._discover")
    devices = discover_module.discover_basler()
    assert len(devices) == 1
    assert isinstance(devices[0], DeviceInfo)
    assert devices[0].device_id == "123"
    assert devices[0].driver == "basler"
    assert devices[0].manufacturer == "Basler"


def test_discover_returns_empty_when_no_sdk() -> None:
    sys.modules.pop("physicalai.capture.cameras.basler._discover", None)
    with mock.patch.dict(sys.modules, {"pypylon": None, "pypylon.pylon": None, "pypylon.genicam": None}):
        module = importlib.import_module("physicalai.capture.cameras.basler._discover")
        result = module.discover_basler()
    assert result == []
    sys.modules.pop("physicalai.capture.cameras.basler._discover", None)


def test_color_mode_bgr(basler_cls: tuple) -> None:
    camera_cls, mock_pylon, _ = basler_cls
    camera = camera_cls(serial_number="123", color_mode=ColorMode.BGR)
    camera.connect()
    converter = mock_pylon.ImageFormatConverter.return_value
    assert converter.OutputPixelFormat == mock_pylon.PixelType_BGR8packed


def test_color_mode_gray(basler_cls: tuple) -> None:
    camera_cls, mock_pylon, _ = basler_cls
    gray_array = np.zeros((480, 640), dtype=np.uint8)
    mock_pylon.ImageFormatConverter.return_value.Convert.return_value.GetArray.return_value = gray_array
    camera = camera_cls(serial_number="123", color_mode=ColorMode.GRAY)
    camera.connect()
    frame = camera.read()
    assert frame.data.shape == (480, 640)
    assert frame.data.dtype == np.uint8


def test_read_not_connected_raises(basler_cls: tuple) -> None:
    camera_cls, _, _ = basler_cls
    camera = camera_cls(serial_number="123")
    with pytest.raises(NotConnectedError):
        camera.read()


def test_read_latest_not_connected_raises(basler_cls: tuple) -> None:
    camera_cls, _, _ = basler_cls
    camera = camera_cls(serial_number="123")
    with pytest.raises(NotConnectedError):
        camera.read_latest()


def test_context_manager_lifecycle(basler_cls: tuple) -> None:
    camera_cls, mock_pylon, _ = basler_cls
    with camera_cls(serial_number="123") as camera:
        assert camera.is_connected
    mock_pylon.InstantCamera.return_value.StopGrabbing.assert_called()
    assert not camera.is_connected


def test_device_id_format(basler_cls: tuple) -> None:
    camera_cls, _, _ = basler_cls
    camera = camera_cls(serial_number="ABC123")
    assert camera.device_id == "basler:ABC123"
