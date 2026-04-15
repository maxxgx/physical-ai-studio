# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Discover a RealSense camera and print a few live RGBD frame summaries."""

# ruff: noqa: D103, INP001

from physicalai.capture.cameras.realsense import RealSenseCamera


def main() -> None:
    devices = RealSenseCamera.discover()
    for i, dev in enumerate(devices):
        print(f"[{i}] {dev.name} (serial={dev.hardware_id})")
    if not devices:
        print("No RealSense devices found.")
        return

    choice = input(f"Select device [0-{len(devices) - 1}]: ").strip()
    serial = devices[int(choice)].device_id

    with RealSenseCamera(serial_number=serial, fps=30, width=640, height=480) as cam:
        print(f"\nStreaming from {cam.device_id} — 640x480 @ 30 fps\n")
        for _ in range(10):
            color, depth = cam.read_rgbd()
            h, w = depth.data.shape
            print(
                f"color={color.data.shape} "
                f"depth={depth.data.shape} [{depth.data.dtype}] "
                f"center={depth.data[h // 2, w // 2]}mm"
            )


if __name__ == "__main__":
    main()
