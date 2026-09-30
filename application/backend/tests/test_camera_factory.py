from typing import Any

from physicalai.capture.camera import Camera

from schemas.project_camera import Camera as ProjectCamera
from schemas.project_camera import CameraAdapter
from utils.camera_factory import build_camera_config


def _usb_camera(fingerprint: dict[str, Any]) -> ProjectCamera:
    return CameraAdapter.validate_python(
        {
            "driver": "usb_camera",
            "name": "Front camera",
            "fingerprint": fingerprint,
            "hardware_name": "Innomaker-U20CAM-1080p-S1",
            "payload": {"width": 640, "height": 480, "fps": 30},
        }
    )


def test_hex_uuid_opens_the_camera_by_hardware_name() -> None:
    recipe = build_camera_config(_usb_camera({"uuid": "0x21230000c456366"}))

    assert recipe.instantiate(expected_type=Camera).device_id == "Innomaker-U20CAM-1080p-S1"


def test_non_hex_uuid_opens_the_camera_by_fingerprint() -> None:
    fingerprint = {"uuid": "6C707041-05AC-0010-0005-000000000001"}

    recipe = build_camera_config(_usb_camera(fingerprint))

    assert recipe.instantiate(expected_type=Camera).device_id == str(fingerprint)
