# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Shared-memory camera transport via iceoryx2.

Provides :func:`create_shared_camera` for multi-process camera sharing
and the lower-level :class:`CameraPublisher` / :class:`SharedCamera`
building blocks.

Requires the ``transport`` extra::

    pip install physicalai[transport]
"""

from __future__ import annotations

import importlib.util
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from physicalai.capture.camera import Camera
    from physicalai.capture.transport._manager import CameraManager  # noqa: PLC2701
    from physicalai.capture.transport._publisher import CameraPublisher  # noqa: PLC2701
    from physicalai.capture.transport._subscriber import SharedCamera  # noqa: PLC2701

__all__ = [
    "CameraManager",
    "CameraPublisher",
    "CameraSpec",
    "SharedCamera",
    "create_shared_camera",
]

from ._spec import CameraSpec


def _ensure_iceoryx2() -> None:
    if importlib.util.find_spec("iceoryx2") is None:
        from physicalai.capture.errors import MissingDependencyError  # noqa: PLC0415

        msg = "iceoryx2"
        raise MissingDependencyError(msg, "transport")


def create_shared_camera(
    camera_type: str,
    *,
    service_name: str | None = None,
    zero_copy: bool = False,
    **kwargs: object,
) -> Camera:
    """Create a shared camera subscriber backed by iceoryx2.

    On Linux with ``iceoryx2`` installed this returns a
    :class:`SharedCamera` connected to a publisher process (spawned
    automatically if needed).  On other platforms — or when iceoryx2 is
    missing — it falls back to a direct single-reader camera with a
    warning.

    Args:
        camera_type: Camera type name (e.g. ``"uvc"``).
        service_name: Explicit iceoryx2 service name.  Derived from
            *camera_type* and the ``device`` kwarg when omitted.
        zero_copy: If ``True``, the returned subscriber uses zero-copy
            reads (frames are read-only numpy views).
        **kwargs: Forwarded to the camera constructor (e.g.
            ``device=0``, ``width=640``).

    Returns:
        A :class:`SharedCamera` subscriber, or a direct camera on
        unsupported platforms.
    """
    _ensure_iceoryx2()

    from physicalai.capture.transport._manager import CameraManager  # noqa: PLC0415, PLC2701

    spec = CameraSpec(camera_type=camera_type, camera_kwargs=dict(kwargs))

    if service_name is None:
        device = kwargs.get("device", 0)
        service_name = f"physicalai/camera/{camera_type}/{device}/frame"

    return CameraManager.get().open(spec, service_name, zero_copy=zero_copy)


def __getattr__(name: str) -> object:
    if name == "CameraPublisher":
        from physicalai.capture.transport._publisher import CameraPublisher  # noqa: PLC0415, PLC2701

        return CameraPublisher

    if name == "SharedCamera":
        from physicalai.capture.transport._subscriber import SharedCamera  # noqa: PLC0415, PLC2701

        return SharedCamera

    if name == "CameraManager":
        from physicalai.capture.transport._manager import CameraManager  # noqa: PLC0415, PLC2701

        return CameraManager

    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
