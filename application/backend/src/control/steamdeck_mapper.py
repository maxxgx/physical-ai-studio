# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Maps Steam Deck gamepad state to robot joint position deltas (velocity mode)."""

from __future__ import annotations

from schemas.steamdeck import SteamDeckAxisMapping, SteamDeckButtonMapping, SteamDeckMapping, SteamDeckState

# Default mapping for SO101 6-DOF arm.
# Axis names match the Steam Deck server's output (examples/steamdeck/server.py).
DEFAULT_SO101_MAPPING = SteamDeckMapping(
    axes={
        "right_stick_x": SteamDeckAxisMapping(joint="shoulder_pan.pos", scale=1.0),
        "right_stick_y": SteamDeckAxisMapping(joint="shoulder_lift.pos", scale=-1.0),
        "left_stick_x": SteamDeckAxisMapping(joint="wrist_roll.pos", scale=1.0),
        "left_stick_y": SteamDeckAxisMapping(joint="wrist_flex.pos", scale=-1.0),
    },
    buttons={
        "l1": SteamDeckButtonMapping(joint="gripper.pos", direction=-1.0),
        "r1": SteamDeckButtonMapping(joint="gripper.pos", direction=1.0),
    },
    max_speed=90.0,
    dead_zone=0.05,
)


class SteamDeckMapper:
    """Converts ``SteamDeckState`` into joint position deltas (velocity mode).

    Each frame, the stick deflection is converted to a position change::

        delta = axis_value * scale * max_speed * goal_time

    Triggers are handled as a differential pair for elbow_flex
    (left_trigger pushes positive, right_trigger pushes negative).
    """

    def __init__(self, mapping: SteamDeckMapping | None = None) -> None:
        self._mapping = mapping or DEFAULT_SO101_MAPPING

    @property
    def mapping(self) -> SteamDeckMapping:
        return self._mapping

    def compute_deltas(self, state: SteamDeckState, goal_time: float) -> dict[str, float]:
        """Return ``{joint_name: delta_degrees}`` for one control frame.

        Args:
            state: Current gamepad state from the Steam Deck server.
            goal_time: Time budget for this frame in seconds (1/fps).

        Returns:
            Dict mapping joint names to position changes in degrees.
        """
        deltas: dict[str, float] = {}
        dead_zone = self._mapping.dead_zone
        max_speed = self._mapping.max_speed

        # Triggers go from -1 (released) or 0 (never touched) to +1 (fully pressed).
        # Use max(0, v) to treat both released and untouched as zero.
        right_trigger = max(0.0, state.axes.get("right_trigger", 0.0))
        left_trigger = max(0.0, state.axes.get("left_trigger", 0.0))
        trigger_diff = right_trigger - left_trigger
        if abs(trigger_diff) > dead_zone:
            deltas["elbow_flex.pos"] = trigger_diff * max_speed * goal_time

        # Axis mappings
        for axis_name, axis_map in self._mapping.axes.items():
            value = state.axes.get(axis_name, 0.0)
            if abs(value) <= dead_zone:
                continue
            delta = value * axis_map.scale * max_speed * goal_time
            deltas[axis_map.joint] = deltas.get(axis_map.joint, 0.0) + delta

        # Button mappings (held = constant velocity)
        for button_name, btn_map in self._mapping.buttons.items():
            if state.buttons.get(button_name, 0):
                delta = btn_map.direction * max_speed * goal_time
                deltas[btn_map.joint] = deltas.get(btn_map.joint, 0.0) + delta

        return deltas
