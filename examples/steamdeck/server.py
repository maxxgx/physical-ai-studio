# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Steam Deck controller → WebSocket bridge.

Reads the Steam Deck's integrated controller via evdev and broadcasts
state to browser clients over WebSocket. Serves a built-in test UI.

Usage::

    python examples/steamdeck/server.py
    python examples/steamdeck/server.py --device /dev/input/event6
    python examples/steamdeck/server.py --hz 60 --port 8080

Requires: ``pip install evdev aiohttp`` (or ``pip install physicalai[steamdeck]``)

Must run on the Steam Deck itself (or any Linux host with evdev controllers).
Stop Steam first so it doesn't grab exclusive access to the controller.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

import evdev
from evdev import ecodes
from aiohttp import web

# ---------------------------------------------------------------------------
# Axis / button friendly-name maps
# ---------------------------------------------------------------------------

AXIS_NAMES: dict[int, str] = {
    ecodes.ABS_X: "left_stick_x",
    ecodes.ABS_Y: "left_stick_y",
    ecodes.ABS_RX: "right_stick_x",
    ecodes.ABS_RY: "right_stick_y",
    ecodes.ABS_HAT0X: "left_pad_x",
    ecodes.ABS_HAT0Y: "left_pad_y",
    ecodes.ABS_HAT1X: "right_pad_x",
    ecodes.ABS_HAT1Y: "right_pad_y",
    ecodes.ABS_HAT2X: "left_trigger",
    ecodes.ABS_HAT2Y: "right_trigger",
}

BUTTON_NAMES: dict[int, str] = {
    ecodes.BTN_SOUTH: "a",
    ecodes.BTN_EAST: "b",
    ecodes.BTN_NORTH: "x",
    ecodes.BTN_WEST: "y",
    ecodes.BTN_TL: "l1",
    ecodes.BTN_TR: "r1",
    ecodes.BTN_TL2: "l2",
    ecodes.BTN_TR2: "r2",
    ecodes.BTN_SELECT: "select",
    ecodes.BTN_START: "start",
    ecodes.BTN_THUMBL: "l3",
    ecodes.BTN_THUMBR: "r3",
    ecodes.BTN_MODE: "steam",
    544: "dpad_up",      # BTN_DPAD_UP
    545: "dpad_down",    # BTN_DPAD_DOWN
    546: "dpad_left",    # BTN_DPAD_LEFT
    547: "dpad_right",   # BTN_DPAD_RIGHT
    ecodes.BTN_THUMB: "l4",
    ecodes.BTN_THUMB2: "r4",
    ecodes.BTN_BASE: "quick_access",
    704: "l5",           # BTN_TRIGGER_HAPPY1
    705: "r5",           # BTN_TRIGGER_HAPPY2
    706: "left_pad_press",   # BTN_TRIGGER_HAPPY3
    707: "right_pad_press",  # BTN_TRIGGER_HAPPY4
}



# ---------------------------------------------------------------------------
# Controller state tracker
# ---------------------------------------------------------------------------


class ControllerState:
    """Accumulates evdev events into a normalized state snapshot."""

    def __init__(self, device: evdev.InputDevice) -> None:
        self.axes: dict[str, float] = {}
        self.buttons: dict[str, int] = {}
        self._axis_info: dict[int, evdev.AbsInfo] = {}
        self._init_from_device(device)

    def _init_from_device(self, device: evdev.InputDevice) -> None:
        caps = device.capabilities(absinfo=True)
        for code, absinfo in caps.get(ecodes.EV_ABS, []):
            name = AXIS_NAMES.get(code, f"axis_{code}")
            self._axis_info[code] = absinfo
            self.axes[name] = 0.0
        for code in caps.get(ecodes.EV_KEY, []):
            if code in BUTTON_NAMES:
                self.buttons[BUTTON_NAMES[code]] = 0

    def update(self, event: evdev.InputEvent) -> None:
        """Process a single evdev event."""
        if event.type == ecodes.EV_ABS:
            name = AXIS_NAMES.get(event.code, f"axis_{event.code}")
            info = self._axis_info.get(event.code)
            if info and (info.max - info.min) > 0:
                mid = (info.max + info.min) / 2
                half_range = (info.max - info.min) / 2
                self.axes[name] = round((event.value - mid) / half_range, 4)
            else:
                self.axes[name] = float(event.value)
        elif event.type == ecodes.EV_KEY and event.code in BUTTON_NAMES:
            self.buttons[BUTTON_NAMES[event.code]] = min(event.value, 1)

    def to_dict(self) -> dict[str, Any]:
        return {"axes": self.axes, "buttons": self.buttons}


# ---------------------------------------------------------------------------
# Device discovery
# ---------------------------------------------------------------------------


def _is_gamepad(dev: evdev.InputDevice) -> bool:
    """Return True if the device has absolute axes typical of a gamepad."""
    caps = dev.capabilities()
    abs_codes = {code for code, _ in caps.get(ecodes.EV_ABS, [])}
    # A real gamepad has at least left-stick X/Y axes
    return {ecodes.ABS_X, ecodes.ABS_Y}.issubset(abs_codes)


def find_controller(device_path: str | None = None) -> evdev.InputDevice:
    """Auto-detect the Steam Deck gamepad device, or use an explicit path.

    The Deck exposes multiple input nodes (trackpad-as-mouse, keyboard,
    gamepad, motion sensors).  We want the one named "Steam Deck" that
    provides absolute axes (sticks/triggers) — typically /dev/input/event8.
    """
    if device_path:
        return evdev.InputDevice(device_path)

    candidates: list[evdev.InputDevice] = []
    for path in evdev.list_devices():
        dev = evdev.InputDevice(path)
        name_lower = dev.name.lower()
        if "steam" in name_lower and _is_gamepad(dev):
            candidates.append(dev)

    # Prefer the device named exactly "Steam Deck" (the gamepad node)
    # over "Valve Software Steam Controller" (trackpad/mouse node) or
    # "Steam Deck Motion Sensors" (gyro).
    for dev in candidates:
        if dev.name == "Steam Deck":
            return dev
    if candidates:
        return candidates[0]

    print("Steam Deck controller not found automatically.")
    print("Available input devices:")
    for path in evdev.list_devices():
        dev = evdev.InputDevice(path)
        print(f"  {dev.path}: {dev.name}")
    raise SystemExit("Use --device /dev/input/eventN to specify the controller.")


# ---------------------------------------------------------------------------
# WebSocket bridge
# ---------------------------------------------------------------------------


class WebSocketBridge:
    """Reads evdev events and broadcasts JSON state over WebSocket."""

    def __init__(self, device: evdev.InputDevice, hz: int = 60) -> None:
        self.device = device
        self.state = ControllerState(device)
        self.hz = hz
        self.clients: set[web.WebSocketResponse] = set()

    async def read_events(self) -> None:
        event_count = 0
        async for event in self.device.async_read_loop():
            self.state.update(event)
            event_count += 1
            if event_count % 100 == 1:
                print(f"[evdev] {event_count} events received")

    async def broadcast_loop(self) -> None:
        interval = 1.0 / self.hz
        while True:
            if self.clients:
                msg = json.dumps(self.state.to_dict())
                stale: list[web.WebSocketResponse] = []
                for ws in self.clients:
                    try:
                        await ws.send_str(msg)
                    except (ConnectionResetError, ConnectionError):
                        stale.append(ws)
                for ws in stale:
                    self.clients.discard(ws)
            await asyncio.sleep(interval)

    async def ws_handler(self, request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        self.clients.add(ws)
        print(f"[ws] client connected  ({len(self.clients)} total)")
        try:
            async for _ in ws:
                pass
        finally:
            self.clients.discard(ws)
            print(f"[ws] client disconnected ({len(self.clients)} total)")
        return ws


# ---------------------------------------------------------------------------
# HTTP handlers
# ---------------------------------------------------------------------------


async def index_handler(_request: web.Request) -> web.FileResponse:
    return web.FileResponse(Path(__file__).parent / "index.html")


# ---------------------------------------------------------------------------
# App setup & entry point
# ---------------------------------------------------------------------------


def build_app(bridge: WebSocketBridge) -> web.Application:
    app = web.Application()
    app.router.add_get("/", index_handler)
    app.router.add_get("/ws", bridge.ws_handler)
    return app


async def run(args: argparse.Namespace) -> None:
    device = find_controller(args.device)
    print(f"Controller : {device.name}  ({device.path})")

    bridge = WebSocketBridge(device, hz=args.hz)
    app = build_app(bridge)

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", args.port)
    await site.start()

    print(f"UI         : http://localhost:{args.port}")
    print(f"WebSocket  : ws://localhost:{args.port}/ws")
    print(f"Rate       : {args.hz} Hz")
    print()

    await asyncio.gather(bridge.read_events(), bridge.broadcast_loop())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Steam Deck controller → WebSocket bridge",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="evdev device path, e.g. /dev/input/event6 (auto-detect if omitted)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="HTTP / WebSocket port (default: 8080)",
    )
    parser.add_argument(
        "--hz",
        type=int,
        default=60,
        help="broadcast rate in Hz (default: 60)",
    )
    args = parser.parse_args()
    try:
        asyncio.run(run(args))
    except KeyboardInterrupt:
        print("\nShutdown.")


if __name__ == "__main__":
    main()
