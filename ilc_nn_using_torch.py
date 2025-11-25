import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import pybullet as p
import pybullet_data
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import math

# --- 1. Definisi Neural Network ---
class ILCNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ILCNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        output = self.fc3(x) 
        return output

# --- 2. Konfigurasi ---
USE_GUI = True
TIMESTEPS = 350       
ITERATIONS = 20       
LEARNING_RATE = 0.005 

# Posisi & Dimensi Tray
TRAY_Z_BASE = 0.02 
POS_LEFT = [0.5, 0.3, TRAY_Z_BASE + 0.025]   
POS_RIGHT = [0.5, -0.3, TRAY_Z_BASE + 0.025] 
OFFSET_TIP = 0.15 

# Setup PyBullet
if USE_GUI:
    p.connect(p.GUI)
    # --- PERBAIKAN DI SINI ---
    # Set ke 1 agar menu samping MUNCUL
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1) 
    # Hilangkan bayangan/rendering ekstra agar ringan
    p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)
else:
    p.connect(p.DIRECT)

p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.8)
p.loadURDF("plane.urdf")
p.loadURDF("table/table.urdf", basePosition=[0.5, 0, -0.65])

# --- FUNGSI MEMBUAT LOYANG ---
def create_tray(center_pos, size=[0.25, 0.35, 0.05]):
    w, l, h = size
    thickness = 0.005
    color = [0.2, 0.2, 0.2, 1] 
    
    def make_part(pos, dim):
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=dim, rgbaColor=color)
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=dim)
        p.createMultiBody(baseMass=0, baseVisualShapeIndex=vis, baseCollisionShapeIndex=col, basePosition=pos)

    make_part([center_pos[0], center_pos[1], 0.01], [w/2, l/2, 0.01]) 
    z_wall = 0.02 + h/2
    make_part([center_pos[0], center_pos[1]+l/2, z_wall], [w/2, thickness, h/2])
    make_part([center_pos[0], center_pos[1]-l/2, z_wall], [w/2, thickness, h/2])
    make_part([center_pos[0]+w/2, center_pos[1], z_wall], [thickness, l/2, h/2])
    make_part([center_pos[0]-w/2, center_pos[1], z_wall], [thickness, l/2, h/2])

create_tray([POS_LEFT[0], POS_LEFT[1], 0])
create_tray([POS_RIGHT[0], POS_RIGHT[1], 0])

# Load Box & Robot
box_id = p.loadURDF("cube_small.urdf", basePosition=POS_LEFT, globalScaling=1.2)
p.changeVisualShape(box_id, -1, rgbaColor=[1, 0, 0, 1])

robot_id = p.loadURDF("franka_panda/panda.urdf", basePosition=[0, 0, 0], useFixedBase=True)
arm_joints = [0, 1, 2, 3, 4, 5, 6] 
gripper_joints = [9, 10]
dof_arm = len(arm_joints)

# Kamera
p.resetDebugVisualizerCamera(cameraDistance=1.2, cameraYaw=90, cameraPitch=-35, cameraTargetPosition=[0.5, 0, 0.2])

# Inisialisasi NN
ilc_net = ILCNetwork(input_dim=dof_arm*2, output_dim=dof_arm)
optimizer = optim.Adam(ilc_net.parameters(), lr=LEARNING_RATE)
criterion = nn.MSELoss()

# --- 3. Generator Trajectory ---
print("Generating Trajectory...")
waypoints = [
    ([0.3, 0.0, 0.6], 0),         
    ([POS_LEFT[0], POS_LEFT[1], 0.3], 0), 
    ([POS_LEFT[0], POS_LEFT[1], POS_LEFT[2] + OFFSET_TIP], 0), 
    ([POS_LEFT[0], POS_LEFT[1], POS_LEFT[2] + OFFSET_TIP], 1), # GRASP
    ([POS_LEFT[0], POS_LEFT[1], 0.4], 1), # LIFT
    ([POS_RIGHT[0], POS_RIGHT[1], 0.4], 1), # MOVE
    ([POS_RIGHT[0], POS_RIGHT[1], POS_RIGHT[2] + OFFSET_TIP], 1), # LOWER
    ([POS_RIGHT[0], POS_RIGHT[1], POS_RIGHT[2] + OFFSET_TIP], 0), # RELEASE
    ([POS_RIGHT[0], POS_RIGHT[1], 0.4], 0)  
]

target_joint_traj = []
gripper_schedule = [] 
steps_per_segment = TIMESTEPS // (len(waypoints) - 1)
orientation = p.getQuaternionFromEuler([math.pi, 0, 0]) 

for i in range(len(waypoints) - 1):
    start_pos, start_grip = waypoints[i]
    end_pos, end_grip = waypoints[i+1]
    for step in range(steps_per_segment):
        alpha = step / steps_per_segment
        curr_xyz = np.array(start_pos) * (1 - alpha) + np.array(end_pos) * alpha
        joint_poses = p.calculateInverseKinematics(robot_id, 11, curr_xyz, orientation)
        target_joint_traj.append(joint_poses[:dof_arm])
        gripper_schedule.append(start_grip if alpha < 0.5 else end_grip)

while len(target_joint_traj) < TIMESTEPS:
    target_joint_traj.append(target_joint_traj[-1])
    gripper_schedule.append(gripper_schedule[-1])
target_joint_traj = np.array(target_joint_traj)

# Reset Robot Awal
for j in range(p.getNumJoints(robot_id)): p.resetJointState(robot_id, j, 0)

# --- 4. Loop Utama ILC ---
text_id = p.addUserDebugText("Ready", [0,0,0.8], textColorRGB=[0,0,0], textSize=1.5)

print("Mulai Training...")
for k in range(ITERATIONS):
    # Reset Environment
    for j in range(p.getNumJoints(robot_id)): p.resetJointState(robot_id, j, 0)
    p.resetJointState(robot_id, 9, 0.04) 
    p.resetJointState(robot_id, 10, 0.04)
    p.resetBasePositionAndOrientation(box_id, POS_LEFT, [0,0,0,1])
    grasp_constraint = None
    
    trial_states = []
    trial_targets = []
    trial_errors = []
    
    p.removeUserDebugItem(text_id)
    text_id = p.addUserDebugText(f"Iterasi: {k+1}/{ITERATIONS}", [0.5, 0, 0.8], textColorRGB=[0,0,0], textSize=1.5)
    
    # TRIAL
    for t in range(TIMESTEPS):
        # A. Control Arm
        states = p.getJointStates(robot_id, arm_joints)
        q_curr = np.array([s[0] for s in states], dtype=np.float32)
        q_des = target_joint_traj[t].astype(np.float32)
        
        inp = torch.tensor(np.concatenate([q_curr, q_des]), dtype=torch.float32)
        with torch.no_grad():
            correction = ilc_net(inp).numpy()
            
        p.setJointMotorControlArray(robot_id, arm_joints, p.POSITION_CONTROL,
                                    targetPositions = q_des + correction,
                                    forces=[100]*dof_arm, positionGains=[0.03]*dof_arm)
        
        # B. Control Gripper
        should_grasp = gripper_schedule[t]
        gripper_target = 0.0 if should_grasp == 1 else 0.04
        p.setJointMotorControlArray(robot_id, gripper_joints, p.POSITION_CONTROL,
                                    targetPositions=[gripper_target, gripper_target],
                                    forces=[50, 50])
        
        # C. Constraint
        ee_pos = p.getLinkState(robot_id, 11)[0]
        box_pos, _ = p.getBasePositionAndOrientation(box_id)
        
        if should_grasp == 1 and grasp_constraint is None:
            if np.linalg.norm(np.array(ee_pos) - np.array(box_pos)) < 0.15:
                grasp_constraint = p.createConstraint(robot_id, 11, box_id, -1, 
                                                      p.JOINT_FIXED, [0,0,0], [0,0,0], [0,0,0])
        elif should_grasp == 0 and grasp_constraint is not None:
            p.removeConstraint(grasp_constraint)
            grasp_constraint = None
            
        p.stepSimulation()
        if USE_GUI: time.sleep(1./240.) 
        
        trial_states.append(q_curr)
        trial_targets.append(q_des)
        trial_errors.append(q_des - q_curr)

    # TRAINING
    mean_err = np.mean(np.linalg.norm(trial_errors, axis=1))
    print(f"Iterasi {k+1} | Mean Error: {mean_err:.4f}")
    
    X_train = []
    Y_train = []
    for i in range(TIMESTEPS):
        X_train.append(np.concatenate([trial_states[i], trial_targets[i]]))
        Y_train.append(trial_errors[i])
        
    X_tensor = torch.tensor(np.array(X_train), dtype=torch.float32)
    Y_tensor = torch.tensor(np.array(Y_train), dtype=torch.float32)
    
    ilc_net.train()
    for _ in range(50):
        optimizer.zero_grad()
        pred = ilc_net(X_tensor)
        loss = criterion(pred, Y_tensor)
        loss.backward()
        optimizer.step()

# --- 5. UI Button ---
p.removeUserDebugItem(text_id)
p.addUserDebugText("TRAINING SELESAI", [0.5, 0, 0.8], textColorRGB=[0,0,0], textSize=2)

# Buat Tombol di Sidebar
# Parameter: Nama Tombol, Min, Max, Start
btn_save_exit = p.addUserDebugParameter("SIMPAN & KELUAR", 1, 0, 0)
btn_prev_val = p.readUserDebugParameter(btn_save_exit)

print("\n=== Selesai ===")
print("Lihat sidebar kanan PyBullet, klik tombol 'SIMPAN & KELUAR'.")

while True:
    p.stepSimulation()
    time.sleep(0.1)
    
    # Baca status tombol
    btn_curr_val = p.readUserDebugParameter(btn_save_exit)
    if btn_curr_val > btn_prev_val:
        print("Tombol ditekan! Menyimpan model...")
        torch.save(ilc_net.state_dict(), "ilc_pickplace_model.pth")
        print("Model tersimpan: ilc_pickplace_model.pth")
        break # Keluar dari loop

p.disconnect()