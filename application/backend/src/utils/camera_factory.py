"""Factory for building camera instances from backend camera configs.

Maps backend driver names to physicalai.capture CameraType and filters
per-driver kwargs so that only constructor-safe parameters reach the camera.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger
from physicalai.capture import CameraType, ColorMode, SharedCamera
from physicalai.capture.factory import create_camera

if TYPE_CHECKING:
    from physicalai.capture.camera import Camera as CameraABC

    from schemas.project_camera import Camera

MIGRATED_DRIVERS: frozenset[str] = frozenset({"usb_camera", "realsense", "basler"})

DRIVER_KEY_MAP: dict[str, str] = {
    "uvc": "usb_camera",
    "realsense": "realsense",
    "basler": "basler",
}

_DRIVER_TO_CAMERA_TYPE: dict[str, CameraType] = {
    "usb_camera": CameraType.UVC,
    "realsense": CameraType.REALSENSE,
    "basler": CameraType.BASLER,
}

# Per-driver kwargs that are safe to pass through to SharedCamera / underlying camera.
_ALLOWED_KWARGS: dict[str, frozenset[str]] = {
    "usb_camera": frozenset({"width", "height", "fps"}),
    "realsense": frozenset({"width", "height", "fps"}),
    "basler": frozenset({"width", "height", "fps"}),
}


def _camera_type_and_kwargs(config: Camera) -> tuple[CameraType, dict[str, object]]:
    camera_type = driver_to_camera_type(config.driver)
    allowed = _ALLOWED_KWARGS.get(config.driver, frozenset())

    payload = config.payload.model_dump()
    camera_kwargs: dict[str, object] = {k: v for k, v in payload.items() if k in allowed and v is not None}

    if camera_type == CameraType.UVC:
        camera_kwargs["device"] = config.fingerprint
    else:
        camera_kwargs["serial_number"] = config.fingerprint

    return camera_type, camera_kwargs


def is_migrated(driver: str) -> bool:
    """Return True if *driver* is supported by physicalai.capture."""
    return driver in MIGRATED_DRIVERS


def driver_to_camera_type(driver: str) -> CameraType:
    """Convert a backend driver name to a CameraType enum value."""
    try:
        return _DRIVER_TO_CAMERA_TYPE[driver]
    except KeyError:
        msg = f"unsupported driver {driver!r}; expected one of {sorted(MIGRATED_DRIVERS)}"
        raise ValueError(msg) from None


def build_shared_camera(
    config: Camera,
    *,
    strict: bool = False,
    overwrite_settings: bool = False,
    idle_timeout: float = 5.0,
) -> SharedCamera:
    """Build a SharedCamera from a backend Camera schema.

    Args:
        config: Backend camera configuration (discriminated union).
        strict: If ``True``, :meth:`~SharedCamera.connect` raises
            :class:`~physicalai.capture.errors.CaptureError` when an
            existing publisher's resolution does not match the requested
            ``width``/``height``.  Use ``False`` for preview streams
            (tolerates mismatch) and ``True`` for recording / inference
            (rejects wrong-resolution frames).
        overwrite_settings: If ``True``, attempt to reconfigure the publisher
            to match requested settings when a config mismatch is detected.
            Requires a publisher that supports the control channel (v2+).
        idle_timeout: Seconds with zero subscribers before the publisher
            self-exits.  Preview-class callers should use a short value
            (e.g. 0.5) for fast turnover on resolution changes;
            record-class callers should use a longer value (e.g. 5.0).

    Returns:
        A configured (but not yet connected) SharedCamera instance.
    """
    camera_type, camera_kwargs = _camera_type_and_kwargs(config)
    logger.debug(f"camera kwargs for {config.name}: {camera_kwargs}")

    return SharedCamera(
        camera_type,
        color_mode=ColorMode.RGB,
        strict=strict,
        overwrite_settings=overwrite_settings,
        idle_timeout=idle_timeout,
        **camera_kwargs,
    )


def build_direct_camera(config: Camera) -> CameraABC:
    """Build a direct Camera instance from a backend Camera schema.

    Unlike :func:`build_shared_camera`, this creates a camera that owns the
    hardware device exclusively — no publisher subprocess, no shared memory.
    Use for preview streams where exclusive access and fast startup matter.

    Args:
        config: Backend camera configuration (discriminated union).

    Returns:
        A configured (but not yet connected) Camera instance.
    """
    camera_type, camera_kwargs = _camera_type_and_kwargs(config)

    return create_camera(camera_type.value, color_mode=ColorMode.RGB, **camera_kwargs)
