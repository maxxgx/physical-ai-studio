# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ctypes
import importlib.util
import pickle
import sys
from uuid import uuid4
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from physicalai.capture.camera import ColorMode
from physicalai.capture.errors import CaptureError, MissingDependencyError, NotConnectedError
from physicalai.capture.frame import Frame
from physicalai.capture.transport._header import (
    HEADER_SIZE,
    PROTOCOL_VERSION,
    FrameHeader,
    decode_depth,
    decode_header,
    decode_rgb,
    decode_rgb_view,
    encode_frame,
)
from physicalai.capture.transport._spec import CameraSpec

HAS_ICEORYX2 = importlib.util.find_spec("iceoryx2") is not None

requires_iceoryx2 = pytest.mark.skipif(not HAS_ICEORYX2, reason="iceoryx2 not installed")
requires_linux = pytest.mark.skipif(sys.platform != "linux", reason="requires Linux")


def _service_name() -> str:
    return f"physicalai/test/{uuid4().hex[:8]}/frame"


class TestCameraSpec:
    def test_picklable(self) -> None:
        spec = CameraSpec(camera_type="uvc", camera_kwargs={"device": 0, "width": 640})
        blob = pickle.dumps(spec)
        restored = pickle.loads(blob)

        assert restored.camera_type == spec.camera_type
        assert restored.camera_kwargs == spec.camera_kwargs

    def test_build_delegates_to_factory(self) -> None:
        spec = CameraSpec(camera_type="uvc", camera_kwargs={"device": 1, "fps": 30})

        with patch("physicalai.capture.factory.create_camera") as mock_create:
            spec.build()

        mock_create.assert_called_once_with("uvc", device=1, fps=30)

    def test_default_kwargs_empty_dict(self) -> None:
        spec = CameraSpec("uvc")
        assert spec.camera_kwargs == {}


class TestFrameHeader:
    def test_sizeof_is_40(self) -> None:
        assert ctypes.sizeof(FrameHeader) == 40

    def test_protocol_version(self) -> None:
        assert PROTOCOL_VERSION == 1

    def test_header_size_matches_sizeof(self) -> None:
        assert HEADER_SIZE == ctypes.sizeof(FrameHeader)


class TestEncodeDecodeRoundtrip:
    def test_rgb_roundtrip(self) -> None:
        data = np.arange(240 * 320 * 3, dtype=np.uint8).reshape((240, 320, 3))
        frame = Frame(data=data, timestamp=123.456789, sequence=7)

        header, payload = encode_frame(frame, ColorMode.RGB)
        full_payload = bytes(header) + payload

        decoded_header = decode_header(full_payload)
        decoded_frame = decode_rgb(decoded_header, full_payload)

        assert decoded_frame.data.shape == (240, 320, 3)
        assert decoded_frame.data.dtype == np.uint8
        assert decoded_frame.sequence == 7
        assert decoded_frame.timestamp == pytest.approx(frame.timestamp)

    def test_gray_roundtrip(self) -> None:
        data = np.arange(240 * 320, dtype=np.uint8).reshape((240, 320))
        frame = Frame(data=data, timestamp=1.0, sequence=3)

        header, payload = encode_frame(frame, ColorMode.GRAY)
        full_payload = bytes(header) + payload

        decoded_header = decode_header(full_payload)
        decoded_frame = decode_rgb(decoded_header, full_payload)

        assert decoded_frame.data.shape == (240, 320)
        assert decoded_frame.data.dtype == np.uint8

    def test_version_mismatch_raises(self) -> None:
        header = FrameHeader(version=PROTOCOL_VERSION + 1)
        payload = bytes(header)
        with pytest.raises(CaptureError, match="Unsupported protocol version"):
            decode_header(payload)

    def test_payload_too_small_raises(self) -> None:
        with pytest.raises(CaptureError, match="Payload too small"):
            decode_header(b"")

    def test_depth_roundtrip(self) -> None:
        rgb_data = np.zeros((240, 320, 3), dtype=np.uint8)
        depth_data = np.arange(240 * 320, dtype=np.uint16).reshape((240, 320))
        frame = Frame(data=rgb_data, timestamp=2.0, sequence=11)
        depth_frame = Frame(data=depth_data, timestamp=2.0, sequence=11)

        header, payload = encode_frame(frame, ColorMode.RGB, depth_frame=depth_frame)
        full_payload = bytes(header) + payload

        assert header.depth_offset > 0

        decoded_depth = decode_depth(header, full_payload)
        assert decoded_depth.data.shape == depth_data.shape
        assert decoded_depth.data.dtype == depth_data.dtype

    def test_rgb_view_roundtrip(self) -> None:
        data = np.arange(240 * 320 * 3, dtype=np.uint8).reshape((240, 320, 3))
        frame = Frame(data=data, timestamp=1.0, sequence=1)

        header, payload = encode_frame(frame, ColorMode.RGB)
        full_payload = memoryview(bytes(header) + payload)

        decoded_header = decode_header(full_payload)
        decoded_frame = decode_rgb_view(decoded_header, full_payload)

        assert decoded_frame.data.shape == (240, 320, 3)
        assert decoded_frame.data.dtype == np.uint8
        assert decoded_frame.sequence == 1
        assert not decoded_frame.data.flags.writeable
        with pytest.raises(ValueError, match="read-only"):
            decoded_frame.data[0, 0, 0] = 0

    def test_no_depth_raises(self) -> None:
        rgb_data = np.zeros((120, 160, 3), dtype=np.uint8)
        frame = Frame(data=rgb_data, timestamp=0.0, sequence=0)
        header, payload = encode_frame(frame, ColorMode.RGB)
        full_payload = bytes(header) + payload

        with pytest.raises(NotImplementedError, match="no depth data"):
            decode_depth(header, full_payload)


class TestImportGuard:
    def test_missing_iceoryx2_raises(self) -> None:
        from physicalai.capture.transport import _ensure_iceoryx2

        with patch.dict("sys.modules", {"iceoryx2": None}):
            with pytest.raises(MissingDependencyError) as exc_info:
                _ensure_iceoryx2()

        assert exc_info.value.package == "iceoryx2"
        assert exc_info.value.extra == "transport"

    def test_macos_fallback_returns_direct_camera(self) -> None:
        from physicalai.capture.transport import create_shared_camera

        mock_camera = MagicMock()
        with (
            patch("physicalai.capture.transport.sys.platform", "darwin"),
            patch("physicalai.capture.factory.create_camera", return_value=mock_camera) as mock_create,
        ):
            result = create_shared_camera("uvc", device=0)

        assert result is mock_camera
        mock_create.assert_called_once_with("uvc", device=0)


@requires_iceoryx2
@requires_linux
class TestCameraPublisher:
    def test_start_stop_lifecycle(self, fake_camera_spec: CameraSpec) -> None:
        from physicalai.capture.transport._publisher import CameraPublisher

        publisher = CameraPublisher(
            fake_camera_spec,
            _service_name(),
            _factory_override="tests.unit.capture.fake:FakeCamera",
        )
        publisher.start(timeout=10.0)
        assert publisher.is_alive
        publisher.stop()
        assert not publisher.is_alive

    def test_context_manager(self, fake_camera_spec: CameraSpec) -> None:
        from physicalai.capture.transport._publisher import CameraPublisher

        with CameraPublisher(
            fake_camera_spec,
            _service_name(),
            _factory_override="tests.unit.capture.fake:FakeCamera",
        ) as publisher:
            assert publisher.is_alive
        assert not publisher.is_alive

    def test_start_failure_propagates(self) -> None:
        from physicalai.capture.transport._publisher import CameraPublisher

        bad_spec = CameraSpec(camera_type="does-not-exist", camera_kwargs={})
        publisher = CameraPublisher(bad_spec, _service_name())

        with pytest.raises(CaptureError, match="failed"):
            publisher.start(timeout=2.0)


@requires_iceoryx2
@requires_linux
class TestSharedCamera:
    def test_connect_disconnect(self, publisher_service: str) -> None:
        from physicalai.capture.transport._subscriber import SharedCamera

        camera = SharedCamera(publisher_service)
        camera.connect(timeout=5.0)
        assert camera.is_connected
        camera.disconnect()
        assert not camera.is_connected

    def test_read_latest_returns_frame(self, publisher_service: str) -> None:
        from physicalai.capture.transport._subscriber import SharedCamera

        camera = SharedCamera(publisher_service)
        camera.connect(timeout=5.0)
        frame = camera.read_latest()
        camera.disconnect()

        assert isinstance(frame, Frame)

    def test_read_blocks_until_frame(self, publisher_service: str) -> None:
        from physicalai.capture.transport._subscriber import SharedCamera

        camera = SharedCamera(publisher_service)
        camera.connect(timeout=5.0)
        frame = camera.read(timeout=2.0)
        camera.disconnect()

        assert isinstance(frame, Frame)

    def test_read_not_connected(self) -> None:
        from physicalai.capture.transport._subscriber import SharedCamera

        camera = SharedCamera(_service_name())
        with pytest.raises(NotConnectedError):
            camera.read()

    def test_read_latest_not_connected(self) -> None:
        from physicalai.capture.transport._subscriber import SharedCamera

        camera = SharedCamera(_service_name())
        with pytest.raises(NotConnectedError):
            camera.read_latest()

    def test_zero_copy_read_only(self, publisher_service: str) -> None:
        from physicalai.capture.transport._subscriber import SharedCamera

        camera = SharedCamera(publisher_service, zero_copy=True)
        camera.connect(timeout=5.0)
        frame = camera.read_latest()
        camera.disconnect()

        assert isinstance(frame, Frame)
        assert not frame.data.flags.writeable


@requires_iceoryx2
@requires_linux
class TestMultiSubscriber:
    def test_two_subscribers_receive_frames(self, publisher_service: str) -> None:
        from physicalai.capture.transport._subscriber import SharedCamera

        cam_a = SharedCamera(publisher_service)
        cam_b = SharedCamera(publisher_service)
        cam_a.connect(timeout=5.0)
        cam_b.connect(timeout=5.0)

        frame_a = cam_a.read_latest()
        frame_b = cam_b.read_latest()

        cam_a.disconnect()
        cam_b.disconnect()

        assert isinstance(frame_a, Frame)
        assert isinstance(frame_b, Frame)
