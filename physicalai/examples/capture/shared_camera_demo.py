# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Demo: shared-memory camera with iceoryx2.

Interactive selection (discovers cameras automatically):

    uv run python examples/capture/shared_camera_demo.py

Skip selection (specify camera directly):

    uv run python examples/capture/shared_camera_demo.py \
        --camera-type uvc --device 0

    uv run python examples/capture/shared_camera_demo.py \
        --camera-type realsense --serial-number 123456789

The demo starts a publisher process, spawns subscriber processes
that each print 10 frames, then shuts everything down.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp

from physicalai.capture.discovery import DeviceInfo, discover_all
from physicalai.capture.transport import CameraPublisher, CameraSpec, SharedCamera


def _select_camera() -> tuple[str, dict]:
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


def _device_info_to_kwargs(driver: str, dev: DeviceInfo) -> tuple[str, dict]:
    """Convert a discovered DeviceInfo into (camera_type, camera_kwargs)."""
    if driver == "realsense":
        if not dev.hardware_id:
            raise SystemExit(f"RealSense device {dev.device_id} has no serial number.")
        return "realsense", {"serial_number": dev.hardware_id}
    return driver, {"device": dev.index}


def _service_name(camera_type: str, camera_kwargs: dict) -> str:
    """Derive a service name from camera type and identifier."""
    device_id = camera_kwargs.get("serial_number", camera_kwargs.get("device", 0))
    return f"physicalai/camera/{camera_type}/{device_id}/frame"


def subscriber_main(service_name: str) -> None:
    """Subscriber entry point for child processes."""
    camera = SharedCamera(service_name)
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
    parser.add_argument("--device", type=int, default=None, help="UVC device index")
    parser.add_argument("--camera-type", default=None, help="Camera backend (uvc, realsense)")
    parser.add_argument("--serial-number", help="RealSense serial number")
    parser.add_argument("--service-name", help="Override iceoryx2 service name")
    parser.add_argument("--subscribers", type=int, default=2, help="Number of subscriber processes")
    args = parser.parse_args()

    if args.camera_type is not None:
        camera_type = args.camera_type
        if camera_type == "realsense":
            if not args.serial_number:
                raise SystemExit("--serial-number is required when using --camera-type realsense")
            camera_kwargs = {"serial_number": args.serial_number}
        else:
            camera_kwargs = {"device": args.device if args.device is not None else 0}
    else:
        camera_type, camera_kwargs = _select_camera()

    service = args.service_name or _service_name(camera_type, camera_kwargs)

    spec = CameraSpec(camera_type=camera_type, camera_kwargs=camera_kwargs)
    publisher = CameraPublisher(spec, service)
    publisher.start()
    print(f"Publisher started on {service}")

    subscribers = [mp.Process(target=subscriber_main, args=(service,)) for _ in range(args.subscribers)]

    try:
        for process in subscribers:
            process.start()

        for process in subscribers:
            process.join()
    finally:
        publisher.stop()

    print("Done")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user, exiting.")
