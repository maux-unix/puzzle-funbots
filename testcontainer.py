import pybullet as p
import pybullet_data
import numpy as np
import time
import imageio

# --- 1. Konfigurasi Simulasi & Recording ---
USE_GUI = False # Harus False untuk recording HD yang efisien
TIMESTEPS = 700 # Diperpanjang sedikit agar gerakan lebih mulus
ITERATIONS = 15
LEARNING_RATE = 0.15
GIF_NAME = "ilc_hd_isometric.gif"

# --- KONFIGURASI HD ---
HD_WIDTH = 1280
HD_HEIGHT = 720
# Rekam lebih jarang agar file tidak meledak ukurannya dan proses tidak terlalu lama
RECORD_EVERY_N_STEPS = 12 

# Offset "Pucuk Tangan" (Virtual Gripper)
TIP_OFFSET = 0.2 

# --- 2. Setup Lingkungan & Fungsi Container ---
# (Bagian ini tidak berubah dari sebelumnya, hanya perbaikan variabel 'col' yang kemarin)

def create_source_tray(center_pos, size_x=0.25, size_y=0.35, height=0.04):
    wall_thickness = 0.005
    col = [0.2, 0.2, 0.2, 1] # Dark Grey
    
    def make_part(pos, dim):
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=dim, rgbaColor=col)
        col_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=dim)
        p.createMultiBody(baseMass=0, baseVisualShapeIndex=vis, baseCollisionShapeIndex=col_shape, basePosition=pos)

    # Alas & Dinding
    make_part([center_pos[0], center_pos[1], center_pos[2]], [size_x/2, size_y/2, wall_thickness/2])
    z_wall = center_pos[2] + height/2
    make_part([center_pos[0], center_pos[1] + size_y/2, z_wall], [size_x/2, wall_thickness, height/2])
    make_part([center_pos[0], center_pos[1] - size_y/2, z_wall], [size_x/2, wall_thickness, height/2])
    make_part([center_pos[0] + size_x/2, center_pos[1], z_wall], [wall_thickness, size_y/2, height/2])
    make_part([center_pos[0] - size_x/2, center_pos[1], z_wall], [wall_thickness, size_y/2, height/2])

    return [center_pos[0], center_pos[1], center_pos[2] + 0.05]

def create_3x3_container(center_pos, cell_size=0.08, wall_height=0.05):
    wall_thickness = 0.005
    half_size = (cell_size * 3) / 2
    col = [0.6, 0.4, 0.2, 1] # Coklat Kayu
    
    base_id = p.loadURDF("cube.urdf", basePosition=[center_pos[0] + half_size, center_pos[1], center_pos[2]], globalScaling=1, useFixedBase=True)
    p.changeVisualShape(base_id, -1, rgbaColor=col)
    
    def make_wall(pos, dim):
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=dim, rgbaColor=col)
        col_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=dim)
        p.createMultiBody(baseMass=0, baseVisualShapeIndex=vis, baseCollisionShapeIndex=col_shape, basePosition=pos)

    for i in range(4):
        offset = -half_size + (i * cell_size)
        make_wall([center_pos[0] + offset, center_pos[1], center_pos[2] + wall_height/2], [wall_thickness, half_size, wall_height/2])
    for i in range(4):
        offset = -half_size + (i * cell_size)
        make_wall([center_pos[0], center_pos[1] + offset, center_pos[2] + wall_height/2], [half_size, wall_thickness, wall_height/2])

    targets = []
    for r in range(3):
        for c in range(3):
            x = center_pos[0] - cell_size + (r * cell_size)
            y = center_pos[1] - cell_size + (c * cell_size)
            targets.append([x, y, center_pos[2] + 0.05])
    return targets

# Setup PyBullet
if USE_GUI:
    p.connect(p.GUI)
else:
    p.connect(p.DIRECT) # Gunakan DIRECT untuk rendering HD di background

p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.8)
p.loadURDF("plane.urdf")
p.loadURDF("table/table.urdf", basePosition=[0.5, 0, -0.65])

# --- 3. Placing Objects ---
# Tray Sumber (Kanan)
TRAY_POS = [0.6, -0.25, 0.0] 
POS_PICK = create_source_tray(TRAY_POS)

# Container Tujuan (Kiri)
CONTAINER_POS = [0.6, 0.25, 0.0]
container_targets = create_3x3_container(CONTAINER_POS)
POS_PLACE = container_targets[4] # Slot tengah

# Load Robot
kuka_id = p.loadURDF("kuka_iiwa/model.urdf", basePosition=[0, 0, 0], useFixedBase=True)
num_joints = p.getNumJoints(kuka_id)
kuka_joints = [i for i in range(num_joints) if p.getJointInfo(kuka_id, i)[2] != p.JOINT_FIXED]

# --- 4. Kinematics & Helper ---
def get_joint_positions():
    states = p.getJointStates(kuka_id, kuka_joints)
    return np.array([s[0] for s in states])

def inverse_kinematics_tip(target_pos_tip):
    target_pos_wrist = [target_pos_tip[0], target_pos_tip[1], target_pos_tip[2] + TIP_OFFSET]
    orn = p.getQuaternionFromEuler([0, -np.pi, 0])
    joint_poses = p.calculateInverseKinematics(kuka_id, 6, target_pos_wrist, orn)
    return np.array(joint_poses[:len(kuka_joints)])

def reset_simulation():
    for j in kuka_joints:
        p.resetJointState(kuka_id, j, targetValue=0, targetVelocity=0)
    p.removeAllUserDebugItems()
    cube_id = p.loadURDF("cube_small.urdf", basePosition=POS_PICK, globalScaling=1.2)
    p.changeVisualShape(cube_id, -1, rgbaColor=[1, 0, 0, 1])
    return cube_id

# --- 5. Trajectory Generation ---
print("Generating Trajectory...")
target_joint_trajectory = []
gripper_schedule = []

# Waypoints: Home -> Hover Tray -> Pick -> GRIP -> Lift -> Move -> Place -> RELEASE -> Lift
waypoints = [
    ([0, 0, 0.7], 0),                   
    ([POS_PICK[0], POS_PICK[1], 0.3], 0), 
    (POS_PICK, 0),                      
    (POS_PICK, 1),                      
    ([POS_PICK[0], POS_PICK[1], 0.3], 1), 
    ([POS_PLACE[0], POS_PLACE[1], 0.3], 1), 
    (POS_PLACE, 1),                     
    (POS_PLACE, 0),                     
    ([POS_PLACE[0], POS_PLACE[1], 0.3], 0)  
]

steps_per_segment = TIMESTEPS // (len(waypoints) - 1)

for i in range(len(waypoints) - 1):
    start_pos, start_grip = waypoints[i]
    end_pos, end_grip = waypoints[i+1]
    for step in range(steps_per_segment):
        alpha = step / steps_per_segment
        curr_pos = np.array(start_pos) * (1 - alpha) + np.array(end_pos) * alpha
        target_joint_trajectory.append(inverse_kinematics_tip(curr_pos))
        gripper_schedule.append(start_grip if alpha < 0.5 else end_grip)

while len(target_joint_trajectory) < TIMESTEPS:
    target_joint_trajectory.append(target_joint_trajectory[-1])
    gripper_schedule.append(gripper_schedule[-1])
target_joint_trajectory = np.array(target_joint_trajectory)

# --- 6. Main ILC Loop & HD Camera Setup ---
U_ilc = np.zeros((TIMESTEPS, len(kuka_joints))) 
frame_buffer = []

# --- SETUP KAMERA HD BARU ---
# Targetkan kamera di tengah-tengah antara tray dan container
camera_target_pos = [0.6, 0.0, 0.0]

# View Matrix: Sudut isometrik (dari atas-samping) agar semua terlihat
view_matrix = p.computeViewMatrixFromYawPitchRoll(
    cameraTargetPosition=camera_target_pos,
    distance=1.5,   # Jarak pass untuk melihat kedua objek
    yaw=50,         # Sudut dari samping
    pitch=-35,      # Sudut menghadap ke bawah
    roll=0,
    upAxisIndex=2
)

# Projection Matrix: Sesuaikan aspect ratio dengan resolusi HD
aspect_ratio = HD_WIDTH / HD_HEIGHT
proj_matrix = p.computeProjectionMatrixFOV(
    fov=60, aspect=aspect_ratio, nearVal=0.1, farVal=100.0
)

print(f"Starting HD ILC Simulation ({ITERATIONS} iterations). This will be slow...")

for k in range(ITERATIONS):
    cube_id = reset_simulation()
    actual_trajectory = []
    grasp_constraint = None
    
    print(f"Processing Iteration {k+1}/{ITERATIONS}...")
    
    for t in range(TIMESTEPS):
        # ILC Control
        q_des = target_joint_trajectory[t]
        q_command = q_des + U_ilc[t]
        p.setJointMotorControlArray(kuka_id, kuka_joints, p.POSITION_CONTROL, 
                                    targetPositions=q_command, forces=[200]*7, positionGains=[0.05]*7)
        
        # Tip Calculation
        link_state = p.getLinkState(kuka_id, 6, computeForwardKinematics=True)
        wrist_pos, wrist_orn = np.array(link_state[0]), np.array(link_state[1])
        rot_matrix = np.array(p.getMatrixFromQuaternion(wrist_orn)).reshape(3, 3)
        current_tip_pos = wrist_pos + np.dot(rot_matrix, [0, 0, TIP_OFFSET])
        
        # Grasping Logic
        if gripper_schedule[t] == 1 and grasp_constraint is None:
            cube_pos, _ = p.getBasePositionAndOrientation(cube_id)
            if np.linalg.norm(np.array(cube_pos) - current_tip_pos) < 0.1:
                grasp_constraint = p.createConstraint(
                    kuka_id, 6, cube_id, -1, p.JOINT_FIXED, 
                    [0, 0, 0], [0, 0, TIP_OFFSET], [0, 0, 0]
                )
        elif gripper_schedule[t] == 0 and grasp_constraint is not None:
            p.removeConstraint(grasp_constraint)
            grasp_constraint = None
            
        p.stepSimulation()
        if USE_GUI: time.sleep(1./240.)
        
        # --- HD RECORDING LOGIC ---
        # Rekam iterasi Awal, Tengah, dan Akhir saja untuk menghemat waktu render
        if (k == 0 or k == ITERATIONS//2 or k == ITERATIONS-1) and t % RECORD_EVERY_N_STEPS == 0:
            # Ambil gambar HD
            w, h, rgb, _, _ = p.getCameraImage(
                width=HD_WIDTH, 
                height=HD_HEIGHT, 
                viewMatrix=view_matrix, 
                projectionMatrix=proj_matrix, 
                renderer=p.ER_BULLET_HARDWARE_OPENGL # Penting untuk kecepatan render HD
            )
            
            # Konversi ke numpy array dan buang channel Alpha (RGBA -> RGB)
            # Ini penting agar ukuran file GIF tidak terlalu besar
            rgb_array = np.array(rgb, dtype=np.uint8)
            rgb_array = rgb_array.reshape((HD_HEIGHT, HD_WIDTH, 4))
            rgb_array = rgb_array[:, :, :3] # Ambil hanya R, G, B
            
            frame_buffer.append(rgb_array)
            
        actual_trajectory.append(get_joint_positions())

    # Update ILC
    error = target_joint_trajectory - np.array(actual_trajectory)
    U_ilc += LEARNING_RATE * error
    print(f"   -> Mean Error: {np.mean(np.linalg.norm(error, axis=1)):.4f}")

p.disconnect()

if len(frame_buffer) > 0:
    print(f"Saving HD GIF to {GIF_NAME} with {len(frame_buffer)} frames...")
    # Gunakan fps lebih rendah (misal 15) karena kita melompati banyak frame saat merekam
    imageio.mimsave(GIF_NAME, frame_buffer, fps=15) 
    print("Done.")
else:
    print("No frames recorded.")