#!/usr/bin/env python3
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Test script: connect a Steam Deck controller to a robot via the backend.

Prerequisites:
    1. Backend is running (default: http://localhost:7860)
    2. Steam Deck server is running (default: ws://localhost:8080/ws)
    3. A project with an environment (robot + cameras) is configured in the app

Usage::

    # Auto-detect first project & environment:
    python examples/steamdeck/test_steamdeck_control.py

    # Specify explicitly:
    python examples/steamdeck/test_steamdeck_control.py \\
        --backend http://localhost:7860 \\
        --steamdeck ws://192.168.1.50:8080/ws \\
        --project-id <uuid> \\
        --environment-id <uuid>

Requires: ``pip install websockets httpx``
"""

from __future__ import annotations

import argparse
import asyncio
import json
import signal
import sys

import httpx
from websockets.asyncio.client import connect


async def fetch_ids(
    base_url: str,
    project_id: str | None,
    environment_id: str | None,
) -> tuple[str, str]:
    """Resolve project and environment IDs, auto-detecting if not provided."""
    async with httpx.AsyncClient(base_url=base_url, timeout=10) as client:
        if not project_id:
            resp = await client.get("/api/projects")
            resp.raise_for_status()
            projects = resp.json()
            if not projects:
                print("No projects found. Create one in the UI first.")
                sys.exit(1)
            project_id = projects[0]["id"]
            print(f"Using project: {projects[0].get('name', project_id)}")

        if not environment_id:
            resp = await client.get(f"/api/projects/{project_id}/environments")
            resp.raise_for_status()
            envs = resp.json()
            if not envs:
                print("No environments found. Create one in the UI first (add a robot + camera).")
                sys.exit(1)
            environment_id = envs[0]["id"]
            print(f"Using environment: {envs[0].get('name', environment_id)}")

        return project_id, environment_id


async def fetch_environment(base_url: str, project_id: str, environment_id: str) -> dict:
    """Fetch the full environment with relations from the backend REST API."""
    async with httpx.AsyncClient(base_url=base_url, timeout=10) as client:
        resp = await client.get(f"/api/projects/{project_id}/environments/{environment_id}")
        resp.raise_for_status()
        return resp.json()


async def run(args: argparse.Namespace) -> None:
    project_id, environment_id = await fetch_ids(args.backend, args.project_id, args.environment_id)
    environment = await fetch_environment(args.backend, project_id, environment_id)

    ws_url = args.backend.replace("http://", "ws://").replace("https://", "wss://")
    ws_url = f"{ws_url}/api/record/robot_control/ws"

    print(f"\nConnecting to backend WS: {ws_url}")
    print(f"Steam Deck server:       {args.steamdeck}")
    print()

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)

    async with connect(ws_url) as ws:
        # 1. Load environment
        await ws.send(json.dumps({
            "event": "load_environment",
            "data": {"environment": environment},
        }))
        print("[1/3] Sent load_environment")

        # Wait for environment_loaded state
        while True:
            msg = json.loads(await ws.recv())
            if msg.get("event") == "error":
                print(f"\n[ERROR] {msg.get('data')}")
                print("Fix the error above and try again.")
                return
            if msg.get("event") == "state" and msg.get("data", {}).get("environment_loaded"):
                print("      Environment loaded!")
                break

        # 2. Connect Steam Deck
        await ws.send(json.dumps({
            "event": "connect_steamdeck",
            "data": {"url": args.steamdeck},
        }))
        print("[2/3] Sent connect_steamdeck")

        # 3. Set follower source
        await ws.send(json.dumps({
            "event": "set_follower_source",
            "data": {"follower_source": "steamdeck"},
        }))
        print("[3/3] Sent set_follower_source: steamdeck")
        print("\nRobot is now controlled by Steam Deck. Press Ctrl+C to stop.\n")

        # Print incoming observations
        while not stop.is_set():
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=0.5)
            except TimeoutError:
                continue
            msg = json.loads(raw)
            match msg.get("event"):
                case "observations":
                    state = msg.get("data", {}).get("state", {})
                    actions = msg.get("data", {}).get("actions")
                    parts = [f"{k}: {v:7.1f}" for k, v in state.items()]
                    line = " | ".join(parts)
                    if actions:
                        act_parts = [f"{k}: {v:7.1f}" for k, v in actions.items()]
                        line += "  ->  " + " | ".join(act_parts)
                    print(f"\r{line}", end="", flush=True)
                case "error":
                    print(f"\n[ERROR] {msg.get('data')}")
                case "state":
                    data = msg.get("data", {})
                    print(f"\n[STATE] follower_source={data.get('follower_source')}")

        # Cleanup
        await ws.send(json.dumps({"event": "disconnect_steamdeck", "data": {}}))
        await ws.send(json.dumps({"event": "disconnect", "data": {}}))
        print("\n\nDisconnected.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Test Steam Deck -> robot control via backend")
    parser.add_argument("--backend", default="http://localhost:7860", help="Backend base URL")
    parser.add_argument("--steamdeck", default="ws://localhost:8080/ws", help="Steam Deck WS URL")
    parser.add_argument("--project-id", default=None, help="Project UUID (auto-detect if omitted)")
    parser.add_argument("--environment-id", default=None, help="Environment UUID (auto-detect if omitted)")
    args = parser.parse_args()

    try:
        asyncio.run(run(args))
    except KeyboardInterrupt:
        print("\nDone.")


if __name__ == "__main__":
    main()
