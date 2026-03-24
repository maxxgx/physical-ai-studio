# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Maps Steam Deck gamepad state to joint deltas via FPS-style inverse kinematics.

The control point is the **wrist link** (where the camera is mounted).
Movement is relative to the camera's facing direction, like an FPS game.

Controls:
    Left stick Y  -> move forward / backward (camera facing direction)
    Left stick X  -> strafe left / right
    Right stick X -> look left / right (yaw)
    Right stick Y -> look up / down (pitch)
    Left trigger  -> move up
    Right trigger -> move down
    L1 / R1       -> gripper close / open (direct, not through IK)
"""

from __future__ import annotations

import numpy as np

from control.so101_kinematics import (
    NUM_WRIST_JOINTS,
    WRIST_JOINT_NAMES,
    check_workspace_bounds_wrist,
    clamp_to_limits,
    forward_kinematics_wrist,
    solve_cartesian_velocity_wrist,
)
from schemas.steamdeck import SteamDeckState


class SteamDeckIKMapper:
    """FPS-style gamepad control using the wrist as the camera point.

    Movement is in the wrist's local frame (forward = camera facing direction).
    Looking rotates the wrist orientation via IK.

    Args:
        max_linear_speed: Maximum linear velocity in m/s.
        max_angular_speed: Maximum angular velocity in rad/s.
        dead_zone: Stick dead-zone threshold (0-1).
        damping: DLS damping factor for the Jacobian pseudo-inverse.
        max_joint_vel: Hard cap on joint velocity in rad/s.
        max_delta_deg: Hard cap on joint position change per frame in degrees.
        gripper_speed: Gripper speed in degrees/s when a bumper is held.
    """

    def __init__(
        self,
        max_linear_speed: float = 0.08,
        max_angular_speed: float = 0.8,
        dead_zone: float = 0.15,
        damping: float = 0.05,
        max_joint_vel: float = 0.8,
        max_delta_deg: float = 1.5,
        gripper_speed: float = 90.0,
    ) -> None:
        self._max_linear_speed = max_linear_speed
        self._max_angular_speed = max_angular_speed
        self._dead_zone = dead_zone
        self._damping = damping
        self._max_joint_vel = max_joint_vel
        self._max_delta_deg = max_delta_deg
        self._gripper_speed = gripper_speed

    def _apply_deadzone(self, value: float) -> float:
        if abs(value) <= self._dead_zone:
            return 0.0
        return value

    def compute_deltas(
        self,
        state: SteamDeckState,
        goal_time: float,
        current_positions_deg: dict[str, float],
    ) -> dict[str, float]:
        """Return ``{joint_name: delta_degrees}`` for one control frame.

        Args:
            state: Current gamepad state.
            goal_time: Frame period in seconds.
            current_positions_deg: Current joint positions in degrees,
                keyed by ``"<joint>.pos"`` (e.g. ``"shoulder_pan.pos"``).

        Returns:
            Dict mapping joint names (with ``.pos`` suffix) to position
            changes in degrees.
        """
        # --- Read stick inputs ---
        lx = self._apply_deadzone(state.axes.get("left_stick_x", 0.0))
        ly = self._apply_deadzone(state.axes.get("left_stick_y", 0.0))
        rx = self._apply_deadzone(state.axes.get("right_stick_x", 0.0))
        ry = self._apply_deadzone(state.axes.get("right_stick_y", 0.0))
        lt = self._apply_deadzone(state.axes.get("left_trigger", 0.0))
        rt = self._apply_deadzone(state.axes.get("right_trigger", 0.0))

        deltas: dict[str, float] = {}

        has_movement = abs(lx) + abs(ly) + abs(lt) + abs(rt) > 0
        has_look = abs(rx) + abs(ry) > 0

        if has_movement or has_look:
            # Current wrist joint angles
            q_rad = np.array(
                [np.deg2rad(current_positions_deg.get(f"{jn}.pos", 0.0)) for jn in WRIST_JOINT_NAMES]
            )

            # Get wrist frame to determine local axes.
            # Wrist frame convention (from URDF):
            #   forward (camera look dir) = -col1 (negative local Y)
            #   right                     =  col2 (local Z)
            #   up                        = -col0 (negative local X)
            T_wrist = forward_kinematics_wrist(q_rad)
            cam_forward = -T_wrist[:3, 1]
            cam_right = T_wrist[:3, 2]

            # --- Movement: left stick + triggers -> base-frame velocity ---
            # Project forward/right onto the horizontal plane (XY) for ground movement.
            fwd_xy = cam_forward[:2]
            fwd_norm = np.linalg.norm(fwd_xy)
            if fwd_norm > 1e-6:
                fwd_xy = fwd_xy / fwd_norm
            else:
                fwd_xy = np.array([1.0, 0.0])

            right_xy = np.array([fwd_xy[1], -fwd_xy[0]])  # perp to forward in XY

            move_fwd = -ly * self._max_linear_speed  # stick up = forward
            move_strafe = lx * self._max_linear_speed  # stick right = strafe right
            move_up = (lt - rt) * self._max_linear_speed  # LT = up, RT = down

            vx = move_fwd * fwd_xy[0] + move_strafe * right_xy[0]
            vy = move_fwd * fwd_xy[1] + move_strafe * right_xy[1]
            vz = move_up

            # --- Look: right stick -> angular velocity ---
            wz = rx * self._max_angular_speed  # right stick X = yaw (around base Z)
            # Pitch: rotate around the wrist's "right" axis (cam_right).
            # Express as angular velocity in base frame.
            pitch_amount = -ry * self._max_angular_speed  # stick up = look up
            wx = pitch_amount * cam_right[0]
            w_y = pitch_amount * cam_right[1]
            w_z_pitch = pitch_amount * cam_right[2]

            cart_vel = np.array([vx, vy, vz, wx, w_y, wz + w_z_pitch])

            if np.linalg.norm(cart_vel) > 1e-8:
                dq_rad = solve_cartesian_velocity_wrist(
                    q_rad, cart_vel, damping=self._damping, max_joint_vel=self._max_joint_vel
                )

                # Candidate new positions (clamped to joint limits)
                q_new = clamp_to_limits(q_rad + dq_rad * goal_time)

                # Workspace guard: reject moves that push the wrist below the table
                if not check_workspace_bounds_wrist(q_new):
                    return deltas

                dq_clamped = q_new - q_rad

                for i in range(NUM_WRIST_JOINTS):
                    delta_deg = float(np.rad2deg(dq_clamped[i]))
                    delta_deg = max(-self._max_delta_deg, min(self._max_delta_deg, delta_deg))
                    if abs(delta_deg) > 1e-4:
                        deltas[f"{WRIST_JOINT_NAMES[i]}.pos"] = delta_deg

        # Gripper: direct control via bumpers (not through IK)
        if state.buttons.get("l1", 0):
            deltas["gripper.pos"] = deltas.get("gripper.pos", 0.0) - self._gripper_speed * goal_time
        if state.buttons.get("r1", 0):
            deltas["gripper.pos"] = deltas.get("gripper.pos", 0.0) + self._gripper_speed * goal_time

        return deltas
