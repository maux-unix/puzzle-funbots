"""
pybullet_ilc_kuka.py
Simple Iterative Learning Control (position-level) on KUKA iiwa in PyBullet.

Usage:
    python pybullet_ilc_kuka.py
"""
import time
import numpy as np
import pybullet as p
import pybullet_data
import matplotlib.pyplot as plt

# ---------- Simulation parameters ----------
GUI = True           # set False to run headless
DT = 1.0/240.0       # physics timestep
T_total = 4.0        # seconds per trial
steps_per_trial = int(T_total / DT)
n_iters = 30         # number of ILC iterations

# ILC gain (scalar or per-joint). Start small (e.g., 0.2).
L_gain = 0.25

# PD low-level gains for position control
Kp = 200.0
Kd = 2.0

# ---------- Start pybullet ----------
if GUI:
    physics_client = p.connect(p.GUI)
else:
    physics_client = p.connect(p.DIRECT)

p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.resetSimulation()
p.setGravity(0, 0, -9.81)
p.setTimeStep(DT)
p.setRealTimeSimulation(0)

plane = p.loadURDF("plane.urdf")
start_pos = [0,0,0]
start_orientation = p.getQuaternionFromEuler([0,0,0])

# Load KUKA iiwa model included in pybullet_data
robot_urdf = "kuka_iiwa/model.urdf"
robot = p.loadURDF(robot_urdf, [0,0,0], start_orientation, useFixedBase=True)

# get joint indices and names for revolute joints
joint_indices = []
for j in range(p.getNumJoints(robot)):
    info = p.getJointInfo(robot, j)
    jtype = info[2]
    if jtype == p.JOINT_REVOLUTE or jtype == p.JOINT_PRISMATIC:
        joint_indices.append(j)

n_joints = len(joint_indices)
print("Controlled joints:", n_joints, joint_indices)

# Disable default motors so we can use position control via API in each step
# (we will still call setJointMotorControlArray with POSITION_CONTROL each step)
# Prepare initial joint positions (home)
init_q = [0.0]*n_joints
for i, j in enumerate(joint_indices):
    p.resetJointState(robot, j, init_q[i])

# ---------- Create desired trajectory (joint-space) ----------
# Example: smooth sinusoidal trajectories with small amplitude
t = np.linspace(0, T_total, steps_per_trial)
# desired shape (steps, joints)
desired = np.zeros((steps_per_trial, n_joints))
amps = 0.4 * np.linspace(1.0, 0.6, n_joints)  # different amplitude per joint
freqs = 0.5 + 0.1 * np.arange(n_joints)       # different frequencies
for i in range(n_joints):
    desired[:, i] = amps[i] * np.sin(2*np.pi*freqs[i]*t)  # rad

# desired velocities (finite difference)
desired_vel = np.vstack([np.zeros((1,n_joints)), np.diff(desired, axis=0)/DT])

# ---------- ILC internal arrays ----------
# feedforward correction: per-iteration, per-time-step, per-joint
ff = np.zeros_like(desired)   # start with zero correction

# storage for plotting
rms_errors = []

# ---------- Simulation loop over iterations ----------
for itr in range(n_iters):
    print(f"=== Iteration {itr+1}/{n_iters} ===")
    # storage for actual trajectory this trial
    actual = np.zeros_like(desired)
    actual_vel = np.zeros_like(desired_vel)
    # Reset robot to initial (home) pose each trial
    for i, j in enumerate(joint_indices):
        p.resetJointState(robot, j, init_q[i])

    # run trial
    for step in range(steps_per_trial):
        q_targets = desired[step, :] + ff[step, :]   # add feedforward correction
        # Use POSITION_CONTROL to follow q_targets with PD gains
        p.setJointMotorControlArray(
            bodyUniqueId=robot,
            jointIndices=joint_indices,
            controlMode=p.POSITION_CONTROL,
            targetPositions=q_targets.tolist(),
            positionGains=[Kp/(Kp+1e-9)]*n_joints,  # pybullet expects gains ~0..1; this mapping is crude
            velocityGains=[Kd]*n_joints
        )
        p.stepSimulation()

        # read actual joint positions & velocities
        qs = []
        qds = []
        for i, j in enumerate(joint_indices):
            st = p.getJointState(robot, j)
            qs.append(st[0])
            qds.append(st[1])
        actual[step, :] = np.array(qs)
        actual_vel[step, :] = np.array(qds)

        if GUI and (step % 200 == 0):
            # slow-down for visualization a bit
            time.sleep(0.0)

    # compute iteration error: desired - actual
    err = desired - actual
    # rms over time & joints
    rms = np.sqrt(np.mean(err**2))
    rms_errors.append(rms)
    print(f"Iteration {itr+1} RMS error: {rms:.6f} rad")

    # ---------- ILC update ----------
    # Simple update: ff_{k+1}(t) = ff_k(t) + L * e_k(t)
    # Optionally clamp ff to reasonable range to avoid explosion.
    ff = ff + L_gain * err

    # optionally apply low-pass filter to ff to add robustness (not implemented here)

# ---------- Plot results ----------
plt.figure(figsize=(8,5))
plt.plot(np.arange(1, n_iters+1), rms_errors, '-o')
plt.xlabel("ILC iteration")
plt.ylabel("RMS joint position error (rad)")
plt.title("ILC learning curve")
plt.grid(True)
plt.show()

# Save the final ff for reuse
np.save("ff_correction.npy", ff)
print("Done. Feedforward correction saved to ff_correction.npy")

# disconnect
p.disconnect()
