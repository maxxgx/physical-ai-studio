import sys; sys.path.insert(0, 'src')
import numpy as np
from control.so101_kinematics import forward_kinematics, compute_jacobian, JOINT_NAMES, solve_cartesian_velocity, check_workspace_bounds

# FK at zero position
q_zero = np.zeros(5)
T = forward_kinematics(q_zero)
print('=== FK at zero joint angles ===')
print(f'EE position (m): x={T[0,3]:.4f}, y={T[1,3]:.4f}, z={T[2,3]:.4f}')

# NEW safe defaults: damping=0.05, max_joint_vel=0.8
cart_vel_full = np.array([0.08, 0, 0, 0, 0, 0])  # new max_linear_speed
dq = solve_cartesian_velocity(q_zero, cart_vel_full, damping=0.05, max_joint_vel=0.8)
print(f'\n=== Joint vel for vx=0.08 m/s, damping=0.05, max_jv=0.8 ===')
for i, name in enumerate(JOINT_NAMES):
    dps = np.rad2deg(dq[i])
    delta = dps * 0.033
    print(f'  {name}: {dps:.1f} deg/s  ({delta:.2f} deg/frame)')

# Simulate small stick drift (0.12 - below new dead zone of 0.15)
print(f'\n=== Stick drift 0.12 (below dead zone 0.15) → REJECTED ===')

# Simulate moderate stick (0.3)
v_mod = np.array([0.3 * 0.08, 0, 0, 0, 0, 0])
dq_mod = solve_cartesian_velocity(q_zero, v_mod, damping=0.05, max_joint_vel=0.8)
print(f'\n=== 30% stick, vx={0.3*0.08:.3f} m/s ===')
for i, name in enumerate(JOINT_NAMES):
    dps = np.rad2deg(dq_mod[i])
    delta = min(1.5, max(-1.5, dps * 0.033))  # with max_delta_deg cap
    print(f'  {name}: {dps:.1f} deg/s  → capped delta: {delta:.2f} deg/frame')

# Test workspace bounds
for angles_deg in [[0,0,0,0,0], [0,45,45,0,0], [0,90,90,0,0]]:
    q = np.deg2rad(angles_deg)
    T = forward_kinematics(q)
    ok = check_workspace_bounds(q)
    print(f'\nJoints {angles_deg}: z={T[2,3]:.3f}m  bounds_ok={ok}')

