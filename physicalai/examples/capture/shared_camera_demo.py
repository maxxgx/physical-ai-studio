# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Demo: shared-memory camera with iceoryx2.

Interactive selection (discovers cameras automatically):

    uv run python examples/capture/shared_camera_demo.py

Skip selection (specify camera directly):

    uv run python examples/capture/shared_camera_demo.py \
        --camera-type uvc --device-id /dev/video0

    uv run python examples/capture/shared_camera_demo.py \
        --camera-type realsense --device-id 123456789

The demo starts a publisher process, spawns subscriber processes
that each print 10 frames, then shuts everything down.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
from typing import Any

from physicalai.capture.discovery import DeviceInfo, discover_all
from physicalai.capture.transport import SharedCamera


def _select_camera() -> tuple[str, dict[str, Any]]:
    """Discover cameras and let the user pick one interactively.

    Returns:
        Tuple of (camera_type, camera_kwargs).
    """
    print("Discovering cameras...")
    results = discover_all()

    all_devices: list[tuple[str, DeviceInfo]] = []
    for driver, devices in results.items():
        for dev in devices:
            all_devices.append((driver, dev))

    if not all_devices:
        raise SystemExit("No cameras found.")

    print(f"\nFound {len(all_devices)} camera(s):\n")
    for i, (driver, dev) in enumerate(all_devices):
        serial = f"  serial={dev.hardware_id}" if dev.hardware_id else ""
        print(f"  {i}: [{driver}] {dev.name or dev.device_id}{serial}")

    if len(all_devices) == 1:
        choice = 0
        print(f"\nAuto-selected the only camera: {all_devices[0][1].name or all_devices[0][1].device_id}")
    else:
        raw = input(f"\nSelect camera [0-{len(all_devices) - 1}]: ").strip()
        try:
            choice = int(raw)
        except ValueError:
            raise SystemExit(f"Invalid selection: {raw}")
        if choice < 0 or choice >= len(all_devices):
            raise SystemExit(f"Selection out of range: {choice}")

    driver, dev = all_devices[choice]
    return _device_info_to_kwargs(driver, dev)


def _device_id_kwarg(camera_type: str) -> str:
    """Return the constructor kwarg name that identifies a device for *camera_type*."""
    if camera_type in ("realsense", "basler"):
        return "serial_number"
    return "device"


def _device_info_to_kwargs(driver: str, dev: DeviceInfo) -> tuple[str, dict[str, Any]]:
    """Convert a discovered DeviceInfo into (camera_type, camera_kwargs)."""
    return driver, {_device_id_kwarg(driver): dev.device_id}


def subscriber_main(service_name: str) -> None:
    """Subscriber entry point for child processes."""
    camera = SharedCamera.from_publisher(service_name=service_name)
    camera.connect()
    try:
        for _ in range(10):
            frame = camera.read_latest()
            print(
                f"shape={frame.data.shape} sequence={frame.sequence} timestamp={frame.timestamp:.6f}",
            )
    finally:
        camera.disconnect()


def main() -> None:
    """Run the shared camera demo."""
    parser = argparse.ArgumentParser(description="Shared-memory camera demo")
    parser.add_argument("--camera-type", default=None, help="Camera backend (uvc, realsense, basler)")
    parser.add_argument("--device-id", default=None, help="Device identifier (serial number, device path, etc.)")
    parser.add_argument("--service-name", help="Override iceoryx2 service name")
    parser.add_argument("--subscribers", type=int, default=2, help="Number of subscriber processes")
    args = parser.parse_args()

    if args.camera_type is not None:
        camera_type = args.camera_type
        if not args.device_id:
            raise SystemExit("--device-id is required when using --camera-type")
        camera_kwargs: dict[str, Any] = {_device_id_kwarg(camera_type): args.device_id}
    else:
        camera_type, camera_kwargs = _select_camera()

    cam = SharedCamera(
        camera_type,
        **camera_kwargs,
        service_name=args.service_name,
    )
    cam.connect()
    print(f"Publisher started on {cam._service_name}")

    subscribers = [mp.Process(target=subscriber_main, args=(cam._service_name,)) for _ in range(args.subscribers)]

    try:
        for process in subscribers:
            process.start()

        for process in subscribers:
            process.join()
    finally:
        cam.disconnect()

    print("Done")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user, exiting.")
