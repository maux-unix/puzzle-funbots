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
import random
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# --- 1. Definisi Neural Network ---
class ILCNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ILCNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, output_dim)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        output = self.fc3(x) 
        return output

# --- 2. Konfigurasi ---
USE_GUI = True
ITERATIONS = 10         
LEARNING_RATE = 0.002

# --- TUNING KECEPATAN & JEDA ---
STEPS_TRAVEL = 20       
STEPS_APPROACH = 60     
STEPS_WAIT = 60         

GRASP_THRESHOLD = 0.15  

POS_TRAY_SOURCE = [0.5, 0.45, 0.02]   
POS_CONTAINER   = [0.5, -0.45, 0.02]  
CELL_SIZE = 0.13        
SAFE_HEIGHT = 0.35

RED = [0.9, 0.1, 0.1, 1]
BLUE = [0.1, 0.1, 0.9, 1]
YELLOW = [0.9, 0.9, 0.1, 1]
COLORS_RGB = [RED, BLUE, YELLOW]
COLOR_NAMES = ["Merah", "Biru", "Kuning"]

if USE_GUI:
    p.connect(p.GUI)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
    p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)
    p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0) 
    p.resetDebugVisualizerCamera(cameraDistance=1.8, cameraYaw=90, cameraPitch=-60, cameraTargetPosition=[0.5, 0, 0.0])
else:
    p.connect(p.DIRECT)

p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.8)
p.loadURDF("plane.urdf") 

robot_id = p.loadURDF("franka_panda/panda.urdf", basePosition=[0, 0, 0], useFixedBase=True)
arm_joints = [0, 1, 2, 3, 4, 5, 6] 
gripper_joints = [9, 10]
dof_arm = len(arm_joints)

ilc_net = ILCNetwork(input_dim=dof_arm*2, output_dim=dof_arm)
optimizer = optim.Adam(ilc_net.parameters(), lr=LEARNING_RATE)
criterion = nn.MSELoss()

# --- 3. Dashboard Matplotlib ---
plt.ion() 
fig = plt.figure(figsize=(10, 8))
gs = GridSpec(2, 2, figure=fig)
ax_loss = fig.add_subplot(gs[0, 0])
ax_error = fig.add_subplot(gs[0, 1])
ax_conf = fig.add_subplot(gs[1, :])
loss_history = []
error_history = []
conf_matrix = np.zeros((3, 3), dtype=int) 

def update_plots(curr_iter):
    ax_loss.clear(); ax_error.clear(); ax_conf.clear()
    ax_loss.plot(loss_history, 'r-', linewidth=1); ax_loss.set_title("Training Loss"); ax_loss.grid(True, alpha=0.3)
    ax_error.plot(error_history, 'b-o', linewidth=1); ax_error.set_title("Tracking Error (ILC)"); ax_error.set_xlabel("Iterasi"); ax_error.grid(True, alpha=0.3)
    im = ax_conf.imshow(conf_matrix, cmap='Greens')
    ax_conf.set_title(f"Sorting Accuracy (Iterasi {curr_iter})")
    ax_conf.set_xticks(np.arange(3)); ax_conf.set_yticks(np.arange(3))
    ax_conf.set_xticklabels(COLOR_NAMES); ax_conf.set_yticklabels(COLOR_NAMES)
    ax_conf.set_xlabel("Lokasi Jatuh (Prediksi)"); ax_conf.set_ylabel("Warna Kubus (Aktual)")
    for i in range(3):
        for j in range(3): ax_conf.text(j, i, conf_matrix[i, j], ha="center", va="center", color="black", fontsize=14, fontweight='bold')
    plt.tight_layout(); plt.pause(0.01)

# --- 4. Helper Functions ---
def create_tray(center_pos, size=[0.5, 0.5, 0.06]):
    w, l, h = size
    col = [0.2, 0.2, 0.2, 1]
    vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[w/2, l/2, 0.01], rgbaColor=col)
    col_s = p.createCollisionShape(p.GEOM_BOX, halfExtents=[w/2, l/2, 0.01])
    p.createMultiBody(0, vis, col_s, basePosition=[center_pos[0], center_pos[1], 0.01])
    def wall(pos, dim):
        v = p.createVisualShape(p.GEOM_BOX, halfExtents=dim, rgbaColor=col)
        c = p.createCollisionShape(p.GEOM_BOX, halfExtents=[dim[0], dim[1], dim[2]*0.5]) 
        p.createMultiBody(0, v, c, basePosition=pos)
    wall([center_pos[0], center_pos[1]+l/2, 0.035], [w/2, 0.005, h/2])
    wall([center_pos[0], center_pos[1]-l/2, 0.035], [w/2, 0.005, h/2])
    wall([center_pos[0]+w/2, center_pos[1], 0.035], [0.005, l/2, h/2])
    wall([center_pos[0]-w/2, center_pos[1], 0.035], [0.005, l/2, h/2])

def create_smart_container(center_pos, cell_size=0.12):
    wall_h = 0.05 
    half_s = (cell_size*3)/2
    col_transparent = [0.5, 0.4, 0.3, 0.3] 
    def wall(pos, dim):
        v = p.createVisualShape(p.GEOM_BOX, halfExtents=dim, rgbaColor=col_transparent)
        c = p.createCollisionShape(p.GEOM_BOX, halfExtents=dim)
        p.createMultiBody(0, v, c, basePosition=pos)
    z_wall = wall_h / 2
    for i in range(4):
        off = -half_s + i*cell_size
        wall([center_pos[0]+off, center_pos[1], z_wall], [0.003, half_s, wall_h/2]) 
        wall([center_pos[0], center_pos[1]+off, z_wall], [half_s, 0.003, wall_h/2]) 
    floor_ids = []
    targets = []
    for r in range(3):
        for c in range(3):
            x = center_pos[0] - cell_size + r*cell_size
            y = center_pos[1] - cell_size + c*cell_size
            z = 0.01
            vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[cell_size/2-0.005, cell_size/2-0.005, 0.005], rgbaColor=[1,1,1,1])
            col_s = p.createCollisionShape(p.GEOM_BOX, halfExtents=[cell_size/2, cell_size/2, 0.005])
            fid = p.createMultiBody(0, vis, col_s, basePosition=[x, y, z])
            floor_ids.append(fid)
            targets.append([x, y, z + 0.05])
    return floor_ids, targets

create_tray(POS_TRAY_SOURCE)
floor_ids, target_coords = create_smart_container(POS_CONTAINER, cell_size=CELL_SIZE)
cubes = []

def spawn_cubes():
    for c in cubes: p.removeBody(c['id']) 
    cubes.clear()
    colors = [0, 0, 0, 1, 1, 1, 2, 2, 2]
    random.shuffle(colors)
    spacing = 0.12 
    start_x = POS_TRAY_SOURCE[0] - spacing
    start_y = POS_TRAY_SOURCE[1] - spacing
    for i in range(9):
        row = i // 3; col = i % 3
        rx = start_x + (row * spacing) + random.uniform(-0.005, 0.005)
        ry = start_y + (col * spacing) + random.uniform(-0.005, 0.005)
        rz = 0.03 
        col_idx = colors[i]
        cid = p.loadURDF("cube_small.urdf", [rx, ry, rz], globalScaling=0.8)
        p.changeVisualShape(cid, -1, rgbaColor=COLORS_RGB[col_idx])
        p.changeDynamics(cid, -1, lateralFriction=1.0, mass=0.5, rollingFriction=0.01, spinningFriction=0.01, restitution=0.0)  
        cubes.append({'id': cid, 'color_idx': col_idx})
    for _ in range(20): p.stepSimulation()

def generate_dynamic_trajectory(current_hand_pos, pickup_pos, drop_pos):
    traj = []
    grip = []
    
    HOVER_Z = SAFE_HEIGHT
    PICK_Z = pickup_pos[2] + 0.05 
    LIFT_Z = pickup_pos[2] + 0.60  
    
    # --- PERBAIKAN: DROP SANGAT RENDAH ---
    # Turun sampai 3cm dari lantai (sangat dekat)
    DROP_Z = drop_pos[2] + 0.03
    
    points_config = [
        ([pickup_pos[0], pickup_pos[1], HOVER_Z], 0, STEPS_TRAVEL),
        ([pickup_pos[0], pickup_pos[1], PICK_Z], 0, STEPS_APPROACH),
        ([pickup_pos[0], pickup_pos[1], PICK_Z], 1, STEPS_WAIT), 
        ([pickup_pos[0], pickup_pos[1], PICK_Z + 0.1], 1, STEPS_APPROACH),
        ([pickup_pos[0], pickup_pos[1], LIFT_Z], 1, STEPS_TRAVEL),
        ([drop_pos[0], drop_pos[1], LIFT_Z], 1, STEPS_TRAVEL),
        ([drop_pos[0], drop_pos[1], DROP_Z], 1, STEPS_APPROACH),
        ([drop_pos[0], drop_pos[1], DROP_Z], 0, STEPS_WAIT), 
        ([drop_pos[0], drop_pos[1], HOVER_Z], 0, STEPS_TRAVEL)
    ]
    orn = p.getQuaternionFromEuler([math.pi, 0, 0])
    current_pos = current_hand_pos
    current_grip = 0
    for pt in points_config:
        target_pos = pt[0]
        target_grip = pt[1]
        steps = pt[2]
        for s in range(steps):
            alpha = s/steps
            pos = np.array(current_pos)*(1-alpha) + np.array(target_pos)*alpha
            ik = p.calculateInverseKinematics(robot_id, 11, pos, orn)
            traj.append(ik[:dof_arm])
            if steps == STEPS_WAIT: grip.append(target_grip)
            else: grip.append(current_grip if alpha < 0.5 else target_grip)
        current_pos = target_pos
        current_grip = target_grip
    return np.array(traj), grip

# --- 5. Main Loop ---
text_main = p.addUserDebugText("START", [0,0,1], textSize=1.5)
btn_exit = p.addUserDebugParameter("SIMPAN & KELUAR", 1, 0, 0)
prev_btn = p.readUserDebugParameter(btn_exit)

print("Simulasi Dimulai (Weak Magnet & Low Drop)...")

for k in range(ITERATIONS):
    for j in range(p.getNumJoints(robot_id)): p.resetJointState(robot_id, j, 0)
    p.resetJointState(robot_id, 9, 0.04); p.resetJointState(robot_id, 10, 0.04)
    spawn_cubes()
    slot_assignments = [0]*3 + [1]*3 + [2]*3 
    random.shuffle(slot_assignments)
    
    target_map = {0: [], 1: [], 2: []} 
    floor_color_map = {} 
    for idx, color_code in enumerate(slot_assignments):
        rgb = COLORS_RGB[color_code]
        floor_rgb = [rgb[0]*0.6, rgb[1]*0.6, rgb[2]*0.6, 1]
        p.changeVisualShape(floor_ids[idx], -1, rgbaColor=floor_rgb)
        target_map[color_code].append(target_coords[idx])
        floor_color_map[idx] = color_code
    
    p.removeUserDebugItem(text_main)
    text_main = p.addUserDebugText(f"Iterasi: {k+1}/{ITERATIONS}", [0, 0, 0], textSize=1.5)
    
    iter_X, iter_Y = [], []
    iter_errors = []
    available_targets = {0: target_map[0].copy(), 1: target_map[1].copy(), 2: target_map[2].copy()}
    
    for cube_obj in cubes:
        c_pos, _ = p.getBasePositionAndOrientation(cube_obj['id'])
        c_col = cube_obj['color_idx']
        
        if c_pos[0] < 0.1 or c_pos[0] > 0.8: continue 
        if len(available_targets[c_col]) == 0: continue 
        target_pos = available_targets[c_col].pop(0)
        
        ee_state = p.getLinkState(robot_id, 11)
        current_hand_pos = ee_state[0]
        
        joint_traj, grip_sched = generate_dynamic_trajectory(current_hand_pos, c_pos, target_pos)
        grasp_constraint = None 
        
        for t in range(len(joint_traj)):
            q_curr = np.array([s[0] for s in p.getJointStates(robot_id, arm_joints)], dtype=np.float32)
            q_des = joint_traj[t].astype(np.float32)
            inp = torch.tensor(np.concatenate([q_curr, q_des]), dtype=torch.float32)
            with torch.no_grad(): correction = ilc_net(inp).numpy()
            p.setJointMotorControlArray(robot_id, arm_joints, p.POSITION_CONTROL, targetPositions=q_des + correction, forces=[200]*dof_arm, positionGains=[0.05]*dof_arm)
            
            should_grasp = grip_sched[t]
            g_val = 0.0 if should_grasp else 0.04
            p.setJointMotorControlArray(robot_id, gripper_joints, p.POSITION_CONTROL, targetPositions=[g_val, g_val], forces=[50,50])
            
            ee_pos = p.getLinkState(robot_id, 11)[0]
            box_pos, _ = p.getBasePositionAndOrientation(cube_obj['id'])
            dist = np.linalg.norm(np.array(ee_pos)-np.array(box_pos))
            if should_grasp: 
                debug_col = [1,0,0] if dist > GRASP_THRESHOLD else [0,1,0]
                p.addUserDebugLine(ee_pos, box_pos, debug_col, lifeTime=0.1)

            if should_grasp and grasp_constraint is None:
                if dist < GRASP_THRESHOLD:
                    grasp_constraint = p.createConstraint(robot_id, 11, cube_obj['id'], -1, p.JOINT_FIXED, [0,0,0], [0,0,0], [0,0,0])
                    
                    # --- PERBAIKAN: MAGNET LEMAH ---
                    # Force hanya 40N. Cukup untuk angkat pelan, tapi jatuh kalau kasar.
                    p.changeConstraint(grasp_constraint, maxForce=100) 
                    
            elif not should_grasp and grasp_constraint is not None:
                p.removeConstraint(grasp_constraint)
                grasp_constraint = None

            p.stepSimulation()
            if USE_GUI: time.sleep(0.001) 
            iter_X.append(np.concatenate([q_curr, q_des]))
            iter_Y.append(q_des - q_curr)
            iter_errors.append(np.linalg.norm(q_des - q_curr))
            if p.readUserDebugParameter(btn_exit) > prev_btn: break
        
        final_cube_pos, _ = p.getBasePositionAndOrientation(cube_obj['id'])
        predicted_class = -1; min_dist = 0.1 
        for idx_floor, target_c in enumerate(target_coords):
            d = np.linalg.norm(np.array(final_cube_pos[:2]) - np.array(target_c[:2]))
            if d < min_dist: predicted_class = floor_color_map[idx_floor]; break
        if predicted_class != -1: conf_matrix[c_col, predicted_class] += 1
        if p.readUserDebugParameter(btn_exit) > prev_btn: break
    
    mean_err = np.mean(iter_errors) if iter_errors else 0
    error_history.append(mean_err)
    print(f"Iterasi {k+1} Selesai | Mean Error: {mean_err:.5f}")
    
    if iter_X:
        X_tensor = torch.tensor(np.array(iter_X), dtype=torch.float32)
        Y_tensor = torch.tensor(np.array(iter_Y), dtype=torch.float32)
        ilc_net.train()
        for _ in range(30): 
            optimizer.zero_grad(); pred = ilc_net(X_tensor); loss = criterion(pred, Y_tensor)
            loss.backward(); optimizer.step(); loss_history.append(loss.item())
    update_plots(k+1)
        
    if p.readUserDebugParameter(btn_exit) > prev_btn:
        print("Menyimpan model..."); torch.save(ilc_net.state_dict(), "ilc_fast_sorter.pth"); break

p.disconnect()