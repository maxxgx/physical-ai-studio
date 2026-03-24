# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Async WebSocket client that connects to the Steam Deck controller server."""

from __future__ import annotations

import asyncio

from loguru import logger
from websockets.asyncio.client import connect
from websockets.exceptions import ConnectionClosed, InvalidURI, WebSocketException

from schemas.steamdeck import SteamDeckState


class SteamDeckClient:
    """Connects to the Steam Deck WebSocket server and caches the latest gamepad state.

    The client runs a background task that continuously reads messages and stores
    the most recent state.  Callers retrieve it via ``get_state()`` (non-blocking).
    """

    def __init__(self, url: str, reconnect_delay: float = 2.0) -> None:
        self._url = url
        self._reconnect_delay = reconnect_delay
        self._state: SteamDeckState | None = None
        self._task: asyncio.Task[None] | None = None
        self._stop = asyncio.Event()

    @property
    def url(self) -> str:
        return self._url

    @property
    def connected(self) -> bool:
        return self._state is not None

    async def start(self) -> None:
        """Start the background reader task."""
        self._stop.clear()
        self._task = asyncio.create_task(self._read_loop())
        logger.info(f"Steam Deck client connecting to {self._url}")

    async def stop(self) -> None:
        """Stop the background reader and close the connection."""
        self._stop.set()
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        self._state = None
        logger.info("Steam Deck client stopped")

    def get_state(self) -> SteamDeckState | None:
        """Return the most recently received gamepad state, or ``None`` if not connected."""
        return self._state

    async def _read_loop(self) -> None:
        """Connect (with auto-reconnect) and read messages into ``_state``."""
        while not self._stop.is_set():
            try:
                async with connect(self._url) as ws:
                    logger.info(f"Steam Deck WS connected: {self._url}")
                    async for message in ws:
                        if self._stop.is_set():
                            break
                        if isinstance(message, str):
                            self._state = SteamDeckState.model_validate_json(message)
            except (ConnectionClosed, ConnectionRefusedError, OSError) as exc:
                logger.warning(f"Steam Deck WS connection lost ({exc}), reconnecting in {self._reconnect_delay}s")
            except InvalidURI:
                logger.error(f"Invalid Steam Deck WS URL: {self._url}")
                break
            except WebSocketException as exc:
                logger.warning(f"Steam Deck WS error ({exc}), reconnecting in {self._reconnect_delay}s")

            self._state = None
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=self._reconnect_delay)
                break  # stop was requested during the delay
            except TimeoutError:
                pass  # reconnect
