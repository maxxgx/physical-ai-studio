# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Demo: N live shared-memory camera subscribers in a single grid window.

First instance (discovers cameras, starts publisher):

    uv run python examples/capture/shared_camera_live_demo.py

Additional instances (skip discovery, reuse existing publisher):

    uv run python examples/capture/shared_camera_live_demo.py \
        --camera-type realsense --serial-number 353322271391 --subscribers 4

Or specify the iceoryx2 service name directly:

    uv run python examples/capture/shared_camera_live_demo.py \
        --service-name physicalai/camera/realsense/353322271391/frame \
        --subscribers 4

Uses SharedCamera to auto-spawn a publisher if none exists, or reuse
an existing one.  Multiple instances of this demo can run concurrently
against the same camera — they all share the same publisher.
Press 'q' to quit.
"""

from __future__ import annotations

import argparse
import math
import time
from typing import Any, cast

import cv2
import numpy as np

from physicalai.capture.discovery import DeviceInfo, discover_all
from physicalai.capture.transport import SharedCamera

DEFAULT_SUBSCRIBERS = 16
TILE_WIDTH = 320
TILE_HEIGHT = 240


# ---------------------------------------------------------------------------
# Camera selection
# ---------------------------------------------------------------------------


def _select_camera() -> tuple[str, dict]:
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
    if driver == "realsense":
        if not dev.hardware_id:
            raise SystemExit(f"RealSense device {dev.device_id} has no serial number.")
        return "realsense", {"serial_number": dev.hardware_id}
    return driver, {"device": dev.index}


# ---------------------------------------------------------------------------
# Main — single-process display loop with N subscribers
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Live shared-memory camera demo (NxN grid)")
    parser.add_argument("--device", type=int, default=None, help="UVC device index")
    parser.add_argument("--camera-type", default=None, help="Camera backend (uvc, realsense)")
    parser.add_argument("--serial-number", help="RealSense serial number")
    parser.add_argument("--service-name", help="Override iceoryx2 service name")
    parser.add_argument("--subscribers", type=int, default=DEFAULT_SUBSCRIBERS, help="Number of subscriber tiles")
    args = parser.parse_args()

    num_subs = max(1, args.subscribers)
    grid_cols = math.ceil(math.sqrt(num_subs))
    shared_camera_ctor = cast(Any, SharedCamera)

    if args.service_name is not None:
        # Explicit service name — subscribe directly, no discovery needed.
        cameras = [shared_camera_ctor.from_publisher(args.service_name, zero_copy=True) for _ in range(num_subs)]
    elif args.camera_type is not None:
        camera_type = args.camera_type
        if camera_type == "realsense":
            if not args.serial_number:
                raise SystemExit("--serial-number is required when using --camera-type realsense")
            camera_kwargs = {"serial_number": args.serial_number}
        else:
            camera_kwargs = {"device": args.device if args.device is not None else 0}
        cameras = [shared_camera_ctor(camera_type, zero_copy=True, **camera_kwargs) for _ in range(num_subs)]
    else:
        # Interactive discovery — may crash if another process holds the
        # camera.  When a publisher is already running, pass --camera-type
        # or --service-name to skip discovery.
        camera_type, camera_kwargs = _select_camera()
        cameras = [shared_camera_ctor(camera_type, zero_copy=True, **camera_kwargs) for _ in range(num_subs)]

    for cam in cameras:
        cam.connect()

    grid_rows = math.ceil(num_subs / grid_cols)
    print(f"Connected {num_subs} subscribers in a {grid_cols}x{grid_rows} grid. Press 'q' to quit.\n")

    # Per-subscriber FPS tracking.
    frame_counts = [0] * num_subs
    fps_values = [0.0] * num_subs
    t_starts = [time.monotonic()] * num_subs
    canvas_h = grid_rows * TILE_HEIGHT
    canvas_w = grid_cols * TILE_WIDTH

    try:
        while True:
            canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

            for i, cam in enumerate(cameras):
                frame = cam.read_latest()

                # Update FPS counter.
                frame_counts[i] += 1
                elapsed = time.monotonic() - t_starts[i]
                if elapsed >= 1.0:
                    fps_values[i] = frame_counts[i] / elapsed
                    frame_counts[i] = 0
                    t_starts[i] = time.monotonic()

                bgr = cv2.cvtColor(frame.data, cv2.COLOR_RGB2BGR)
                tile = cv2.resize(bgr, (TILE_WIDTH, TILE_HEIGHT))

                text = f"Sub {i}  seq={frame.sequence}  fps={fps_values[i]:.1f}"
                cv2.putText(tile, text, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                row, col = divmod(i, grid_cols)
                y0 = row * TILE_HEIGHT
                x0 = col * TILE_WIDTH
                canvas[y0 : y0 + TILE_HEIGHT, x0 : x0 + TILE_WIDTH] = tile

            cv2.imshow(f"SHM Camera x{num_subs}", canvas)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        cv2.destroyAllWindows()
        for cam in cameras:
            cam.disconnect()
        print("Done.")


if __name__ == "__main__":
    main()
