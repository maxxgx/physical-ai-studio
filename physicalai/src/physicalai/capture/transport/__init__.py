# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Shared-memory camera transport via iceoryx2.

Provides :func:`create_shared_camera` and :class:`SharedCamera`
as the public entry points for multi-process camera sharing.

Requires the ``transport`` extra::

    pip install physicalai[transport]
"""

from __future__ import annotations

import importlib.util
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from physicalai.capture.camera import Camera
    from physicalai.capture.transport._shared_camera import SharedCamera

__all__ = [
    "SharedCamera",
    "create_shared_camera",
]

from ._shared_camera import SharedCamera


def _ensure_iceoryx2() -> None:
    if importlib.util.find_spec("iceoryx2") is None:
        from physicalai.capture.errors import MissingDependencyError  # noqa: PLC0415

        msg = "iceoryx2"
        raise MissingDependencyError(msg, "transport")


def create_shared_camera(
    camera_type: str,
    *,
    zero_copy: bool = False,
    **kwargs: Any,  # noqa: ANN401
) -> Camera:
    """Create a shared camera subscriber backed by iceoryx2.

    Returns a :class:`SharedCamera` connected to a publisher process
    (spawned automatically if needed).

    Args:
        camera_type: Camera type name (e.g. ``"uvc"``).
        zero_copy: If ``True``, the returned subscriber uses zero-copy
            reads (frames are read-only numpy views).
        **kwargs: Forwarded to the camera constructor (e.g.
            ``device=0``, ``width=640``).

    Returns:
        A :class:`SharedCamera` subscriber.
    """
    _ensure_iceoryx2()

    return SharedCamera(camera_type, zero_copy=zero_copy, **kwargs)
