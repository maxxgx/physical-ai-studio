# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pydantic import BaseModel, Field


class SteamDeckState(BaseModel):
    """Latest gamepad state received from the Steam Deck WebSocket server."""

    axes: dict[str, float] = Field(default_factory=dict)
    buttons: dict[str, int] = Field(default_factory=dict)


class SteamDeckAxisMapping(BaseModel):
    """Maps a gamepad axis to a robot joint."""

    joint: str
    scale: float = 1.0


class SteamDeckButtonMapping(BaseModel):
    """Maps a gamepad button to a robot joint direction."""

    joint: str
    direction: float = 1.0


class SteamDeckMapping(BaseModel):
    """Full gamepad-to-joint mapping configuration."""

    axes: dict[str, SteamDeckAxisMapping] = Field(default_factory=dict)
    buttons: dict[str, SteamDeckButtonMapping] = Field(default_factory=dict)
    max_speed: float = Field(default=90.0, description="Max joint speed in degrees/second per axis")
    dead_zone: float = Field(default=0.05, description="Ignore axis values below this threshold")
