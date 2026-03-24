# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Pure-numpy forward kinematics and Jacobian for the SO101 6-DOF arm.

The kinematic parameters are extracted from the SO101 URDF
(``application/ui/public/SO101/so101_new_calib.urdf``).

Only the 5 positioning joints (shoulder_pan … wrist_roll) are modelled;
the gripper joint does not affect end-effector pose.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Joint names (must match the observation keys minus the ".pos" suffix)
# ---------------------------------------------------------------------------
JOINT_NAMES: list[str] = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
]

NUM_JOINTS: int = len(JOINT_NAMES)

# The first 4 joints position the wrist link (where the camera is mounted).
# wrist_roll (joint 4) rotates the gripper but does not move the wrist point.
NUM_WRIST_JOINTS: int = 4
WRIST_JOINT_NAMES: list[str] = JOINT_NAMES[:NUM_WRIST_JOINTS]

# ---------------------------------------------------------------------------
# Fixed transforms between consecutive links (URDF ``<origin xyz rpy>``).
# Order: base_link → shoulder_link → upper_arm_link → lower_arm_link
#        → wrist_link → gripper_link.
# ---------------------------------------------------------------------------
_JOINT_ORIGINS: list[dict[str, list[float]]] = [
    # shoulder_pan: base_link → shoulder_link
    {"xyz": [0.0388353, -8.97657e-09, 0.0624], "rpy": [3.14159, 4.18253e-17, -3.14159]},
    # shoulder_lift: shoulder_link → upper_arm_link
    {"xyz": [-0.0303992, -0.0182778, -0.0542], "rpy": [-1.5708, -1.5708, 0.0]},
    # elbow_flex: upper_arm_link → lower_arm_link
    {"xyz": [-0.11257, -0.028, 0.0], "rpy": [0.0, 0.0, 1.5708]},
    # wrist_flex: lower_arm_link → wrist_link
    {"xyz": [-0.1349, 0.0052, 0.0], "rpy": [0.0, 0.0, -1.5708]},
    # wrist_roll: wrist_link → gripper_link
    {"xyz": [0.0, -0.0611, 0.0181], "rpy": [1.5708, 0.0486795, 3.14159]},
]

# Fixed transform from gripper_link to the gripper_frame (tool tip).
_EE_ORIGIN: dict[str, list[float]] = {
    "xyz": [-0.0079, -0.000218121, -0.0981274],
    "rpy": [0.0, 3.14159, 0.0],
}

# Joint limits in radians [lower, upper].
JOINT_LIMITS: list[tuple[float, float]] = [
    (-1.91986, 1.91986),  # shoulder_pan
    (-1.74533, 1.74533),  # shoulder_lift
    (-1.69, 1.69),  # elbow_flex
    (-1.65806, 1.65806),  # wrist_flex
    (-2.74385, 2.84121),  # wrist_roll
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _rpy_to_rotation(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """URDF convention: R = Rz(yaw) Ry(pitch) Rx(roll)."""
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )


def _make_transform(xyz: list[float], rpy: list[float]) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = _rpy_to_rotation(*rpy)
    T[:3, 3] = xyz
    return T


def _rot_z(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    T = np.eye(4)
    T[0, 0] = c
    T[0, 1] = -s
    T[1, 0] = s
    T[1, 1] = c
    return T


# Pre-compute the fixed transforms (they never change).
_FIXED_TRANSFORMS: list[np.ndarray] = [_make_transform(o["xyz"], o["rpy"]) for o in _JOINT_ORIGINS]
_EE_TRANSFORM: np.ndarray = _make_transform(_EE_ORIGIN["xyz"], _EE_ORIGIN["rpy"])


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def forward_kinematics(joint_angles_rad: np.ndarray) -> np.ndarray:
    """Compute the 4x4 end-effector pose in the base frame.

    Args:
        joint_angles_rad: Array of 5 joint angles in radians.

    Returns:
        4x4 homogeneous transformation matrix.
    """
    T = np.eye(4)
    for i in range(NUM_JOINTS):
        T = T @ _FIXED_TRANSFORMS[i] @ _rot_z(joint_angles_rad[i])
    return T @ _EE_TRANSFORM


def compute_jacobian(joint_angles_rad: np.ndarray) -> np.ndarray:
    """Compute the 6x5 geometric Jacobian at the given configuration.

    Rows 0-2: linear velocity component.
    Rows 3-5: angular velocity component.

    Args:
        joint_angles_rad: Array of 5 joint angles in radians.

    Returns:
        6x5 Jacobian matrix.
    """
    T = np.eye(4)
    z_axes: list[np.ndarray] = []
    origins: list[np.ndarray] = []

    for i in range(NUM_JOINTS):
        T = T @ _FIXED_TRANSFORMS[i]
        # After the fixed transform the joint rotates about local z.
        z_axes.append(T[:3, 2].copy())
        origins.append(T[:3, 3].copy())
        T = T @ _rot_z(joint_angles_rad[i])

    # End-effector position
    T_ee = T @ _EE_TRANSFORM
    p_ee = T_ee[:3, 3]

    J = np.zeros((6, NUM_JOINTS))
    for i in range(NUM_JOINTS):
        J[:3, i] = np.cross(z_axes[i], p_ee - origins[i])  # linear
        J[3:, i] = z_axes[i]  # angular
    return J


def solve_cartesian_velocity(
    joint_angles_rad: np.ndarray,
    cartesian_vel: np.ndarray,
    damping: float = 0.05,
    max_joint_vel: float = 1.0,
) -> np.ndarray:
    """Resolve a desired Cartesian velocity into joint velocities.

    Uses the damped-least-squares (DLS) pseudo-inverse of the Jacobian
    which stays well-conditioned near singularities.

    Args:
        joint_angles_rad: Current joint angles (5,) in radians.
        cartesian_vel: Desired [vx, vy, vz, wx, wy, wz] (6,).
        damping: Damping factor for DLS.
        max_joint_vel: Maximum allowed joint velocity in rad/s per joint.

    Returns:
        Joint velocities (5,) in rad/s, clamped to ``max_joint_vel``.
    """
    J = compute_jacobian(joint_angles_rad)
    JJT = J @ J.T
    raw: np.ndarray = J.T @ np.linalg.solve(JJT + damping**2 * np.eye(6), cartesian_vel)

    # Clamp individual joint velocities for safety
    return np.clip(raw, -max_joint_vel, max_joint_vel)


# Minimum end-effector Z height (metres) above the base frame origin.
# Prevents the arm from commanding poses below the table surface.
MIN_EE_HEIGHT: float = 0.02


def check_workspace_bounds(joint_angles_rad: np.ndarray) -> bool:
    """Return True if the configuration keeps the EE above the floor."""
    T = forward_kinematics(joint_angles_rad)
    return float(T[2, 3]) >= MIN_EE_HEIGHT


def clamp_to_limits(joint_angles_rad: np.ndarray) -> np.ndarray:
    """Clamp joint angles to the URDF-defined limits."""
    clamped = joint_angles_rad.copy()
    for i in range(min(len(clamped), len(JOINT_LIMITS))):
        lo, hi = JOINT_LIMITS[i]
        clamped[i] = np.clip(clamped[i], lo, hi)
    return clamped


# ---------------------------------------------------------------------------
# Wrist-frame kinematics (camera control point)
# ---------------------------------------------------------------------------
def forward_kinematics_wrist(joint_angles_rad: np.ndarray) -> np.ndarray:
    """Compute the 4x4 wrist-link pose in the base frame.

    Uses only the first 4 joints (shoulder_pan through wrist_flex).
    The wrist link is where the camera is mounted.

    Args:
        joint_angles_rad: Array of at least 4 joint angles in radians.

    Returns:
        4x4 homogeneous transformation matrix.
    """
    T = np.eye(4)
    for i in range(NUM_WRIST_JOINTS):
        T = T @ _FIXED_TRANSFORMS[i] @ _rot_z(joint_angles_rad[i])
    return T


def compute_jacobian_wrist(joint_angles_rad: np.ndarray) -> np.ndarray:
    """Compute the 6x4 geometric Jacobian for the wrist link.

    Only joints 0-3 (shoulder_pan through wrist_flex) affect the
    wrist position and orientation.

    Args:
        joint_angles_rad: Array of at least 4 joint angles in radians.

    Returns:
        6x4 Jacobian matrix.
    """
    T = np.eye(4)
    z_axes: list[np.ndarray] = []
    origins: list[np.ndarray] = []

    for i in range(NUM_WRIST_JOINTS):
        T = T @ _FIXED_TRANSFORMS[i]
        z_axes.append(T[:3, 2].copy())
        origins.append(T[:3, 3].copy())
        T = T @ _rot_z(joint_angles_rad[i])

    p_wrist = T[:3, 3]

    J = np.zeros((6, NUM_WRIST_JOINTS))
    for i in range(NUM_WRIST_JOINTS):
        J[:3, i] = np.cross(z_axes[i], p_wrist - origins[i])
        J[3:, i] = z_axes[i]
    return J


def solve_cartesian_velocity_wrist(
    joint_angles_rad: np.ndarray,
    cartesian_vel: np.ndarray,
    damping: float = 0.05,
    max_joint_vel: float = 0.8,
) -> np.ndarray:
    """Resolve a Cartesian velocity into wrist joint velocities (4 joints).

    Args:
        joint_angles_rad: Current joint angles (at least 4) in radians.
        cartesian_vel: Desired [vx, vy, vz, wx, wy, wz] (6,).
        damping: DLS damping factor.
        max_joint_vel: Hard cap on joint velocity in rad/s.

    Returns:
        Joint velocities (4,) in rad/s, clamped to ``max_joint_vel``.
    """
    J = compute_jacobian_wrist(joint_angles_rad)
    JJT = J @ J.T
    raw: np.ndarray = J.T @ np.linalg.solve(JJT + damping**2 * np.eye(6), cartesian_vel)
    return np.clip(raw, -max_joint_vel, max_joint_vel)


def check_workspace_bounds_wrist(joint_angles_rad: np.ndarray) -> bool:
    """Return True if the wrist is above the floor."""
    T = forward_kinematics_wrist(joint_angles_rad)
    return float(T[2, 3]) >= MIN_EE_HEIGHT
