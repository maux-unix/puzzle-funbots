import pybullet as p
import pybullet_data
import numpy as np
import time
import imageio

# --- 1. Konfigurasi Simulasi ---
USE_GUI = False # Set True untuk melihat langsung, False untuk recording lebih cepat
TIMESTEPS = 500
ITERATIONS = 15
LEARNING_RATE = 0.15
GIF_NAME = "ilc_kuka_container_3x3.gif"
RECORD_EVERY_N_STEPS = 5

# Offset "Pucuk Tangan" (Virtual Gripper)
# Kita anggap ada gripper sepanjang 20cm (0.2m) dari poros link terakhir
TIP_OFFSET = 0.2 

# Posisi Awal (Fix)
POS_PICK = [0.4, -0.2, 0.05] 

# --- 2. Setup Lingkungan & Container 3x3 ---

def create_3x3_container(center_pos, cell_size=0.08, wall_height=0.05):
    """
    Membuat visualisasi container 3x3 menggunakan dinding tipis.
    center_pos: [x, y, z] pusat container
    """
    wall_thickness = 0.005
    half_size = (cell_size * 3) / 2
    
    # Warna container (Abu-abu gelap)
    col = [0.3, 0.3, 0.3, 1]
    
    # --- PERBAIKAN DI SINI ---
    # Simpan ID objek ke variabel 'wall_id'
    wall_id = p.loadURDF("cube.urdf", 
                         basePosition=[center_pos[0] + half_size, center_pos[1], center_pos[2]], 
                         globalScaling=1, 
                         useFixedBase=True)
    
    # Gunakan 'wall_id' tersebut untuk mengubah warna
    p.changeVisualShape(wall_id, -1, rgbaColor=col)
    
    # Kita gunakan p.createMultiBody untuk dinding custom agar lebih rapi
    def make_wall(pos, dim):
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=dim, rgbaColor=col)
        col_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=dim)
        p.createMultiBody(baseMass=0, baseVisualShapeIndex=vis, baseCollisionShapeIndex=col_shape, basePosition=pos)

    # 4 Dinding Garis X (Horizontal)
    for i in range(4):
        offset = -half_size + (i * cell_size)
        make_wall([center_pos[0] + offset, center_pos[1], center_pos[2] + wall_height/2], 
                  [wall_thickness, half_size, wall_height/2])
        
    # 4 Dinding Garis Y (Vertikal)
    for i in range(4):
        offset = -half_size + (i * cell_size)
        make_wall([center_pos[0], center_pos[1] + offset, center_pos[2] + wall_height/2], 
                  [half_size, wall_thickness, wall_height/2])

    # Hitung koordinat pusat setiap sel untuk target
    targets = []
    for r in range(3): # row
        for c in range(3): # col
            x = center_pos[0] - cell_size + (r * cell_size)
            y = center_pos[1] - cell_size + (c * cell_size)
            targets.append([x, y, center_pos[2] + 0.05])
    
    return targets

# Setup PyBullet
if USE_GUI:
    p.connect(p.GUI)
else:
    p.connect(p.DIRECT)

p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.8)

# Load Plane & Table
p.loadURDF("plane.urdf")
table_id = p.loadURDF("table/table.urdf", basePosition=[0.5, 0, -0.65])

# Buat Container 3x3 di atas meja
CONTAINER_CENTER = [0.6, 0.2, 0.0]
container_targets = create_3x3_container(CONTAINER_CENTER)

# Kita pilih SLOT TENGAH (index 4) sebagai target ILC kali ini
# (ILC belajar gerakan berulang ke satu titik)
POS_PLACE = container_targets[4] 

# Load Robot
kuka_id = p.loadURDF("kuka_iiwa/model.urdf", basePosition=[0, 0, 0], useFixedBase=True)
num_joints = p.getNumJoints(kuka_id)
kuka_joints = [i for i in range(num_joints) if p.getJointInfo(kuka_id, i)[2] != p.JOINT_FIXED]

# --- 3. Kinematika & Helper ---

def get_joint_positions():
    states = p.getJointStates(kuka_id, kuka_joints)
    return np.array([s[0] for s in states])

def inverse_kinematics_tip(target_pos_tip):
    """
    IK Custom: Menghitung sudut joint agar TIP (bukan wrist) sampai ke target.
    Karena Kuka model ini tidak punya link tip, kita hitung target wrist.
    Target Wrist = Target Tip + Offset (ke atas Z)
    """
    # Kita ingin 'Ujung Jari' di target_pos_tip.
    # Maka 'Pergelangan' harus berada di atasnya setinggi TIP_OFFSET.
    # Asumsi: Gripper selalu menghadap ke bawah vertikal saat pick/place.
    
    target_pos_wrist = [target_pos_tip[0], target_pos_tip[1], target_pos_tip[2] + TIP_OFFSET]
    
    # Orientasi Gripper (Menghadap Bawah)
    orn = p.getQuaternionFromEuler([0, -np.pi, 0])
    
    joint_poses = p.calculateInverseKinematics(kuka_id, 6, target_pos_wrist, orn)
    return np.array(joint_poses[:len(kuka_joints)])

def reset_simulation():
    for j in kuka_joints:
        p.resetJointState(kuka_id, j, targetValue=0, targetVelocity=0)
    p.removeAllUserDebugItems()
    
    # Re-draw container (karena removeAll menghapusnya)
    # Note: Untuk efisiensi di simulasi berat, sebaiknya container dimuat sekali saja dengan ID tetap,
    # tapi disini kita re-draw agar simple dengan removeAllUserDebugItems
    _ = create_3x3_container(CONTAINER_CENTER)
    
    # Load Cube di POS_PICK
    cube_id = p.loadURDF("cube_small.urdf", basePosition=POS_PICK, globalScaling=1.2)
    p.changeVisualShape(cube_id, -1, rgbaColor=[1, 0, 0, 1])
    return cube_id

# --- 4. Trajectory Planning (Path Generation) ---
print("Generating Trajectory...")
target_joint_trajectory = []
gripper_schedule = []

# Waypoints: Start -> Atas Pick -> Pick (Turun) -> Angkat -> Geser -> Place (Turun) -> Lepas -> Angkat
waypoints = [
    ([0, 0, 0.7], 0),              # Home
    ([POS_PICK[0], POS_PICK[1], 0.4], 0), # Hover above Pick
    (POS_PICK, 0),                 # Turun ke Pick
    (POS_PICK, 1),                 # GRASP
    ([POS_PICK[0], POS_PICK[1], 0.4], 1), # Angkat lurus
    ([POS_PLACE[0], POS_PLACE[1], 0.4], 1), # Geser ke atas Container
    (POS_PLACE, 1),                # Turun ke dalam Container
    (POS_PLACE, 0),                # LEPAS (Place)
    ([POS_PLACE[0], POS_PLACE[1], 0.4], 0)  # Angkat kembali
]

steps_per_segment = TIMESTEPS // (len(waypoints) - 1)

for i in range(len(waypoints) - 1):
    start_pos, start_grip = waypoints[i]
    end_pos, end_grip = waypoints[i+1]
    
    for step in range(steps_per_segment):
        alpha = step / steps_per_segment
        # Interpolasi Linear Posisi
        curr_pos = np.array(start_pos) * (1 - alpha) + np.array(end_pos) * alpha
        
        # HITUNG IK UNTUK TIP
        q_target = inverse_kinematics_tip(curr_pos)
        
        target_joint_trajectory.append(q_target)
        gripper_schedule.append(start_grip if alpha < 0.5 else end_grip)

# Padding sisa langkah
while len(target_joint_trajectory) < TIMESTEPS:
    target_joint_trajectory.append(target_joint_trajectory[-1])
    gripper_schedule.append(gripper_schedule[-1])
target_joint_trajectory = np.array(target_joint_trajectory)

# --- 5. Main Loop ILC ---
U_ilc = np.zeros((TIMESTEPS, len(kuka_joints))) 
frame_buffer = []

# Kamera Setup
view_matrix = p.computeViewMatrixFromYawPitchRoll(
    cameraTargetPosition=[0.5, 0.0, 0.2], distance=1.6, yaw=45, pitch=-35, roll=0, upAxisIndex=2)
proj_matrix = p.computeProjectionMatrixFOV(fov=60, aspect=1.0, nearVal=0.1, farVal=100.0)

print(f"Mulai Simulasi {ITERATIONS} Iterasi...")

for k in range(ITERATIONS):
    cube_id = reset_simulation()
    actual_trajectory = []
    grasp_constraint = None
    
    # Indikator Visual di pucuk tangan (Bola hijau kecil)
    # Agar kita bisa lihat di mana "ujung" tangan robot berada
    debug_tip_visual = p.addUserDebugText(".", [0,0,0]) 
    
    print(f"Iterasi {k+1}/{ITERATIONS}")
    
    for t in range(TIMESTEPS):
        # ILC Control Update
        q_des = target_joint_trajectory[t]
        q_command = q_des + U_ilc[t]
        
        # Kirim perintah ke motor
        p.setJointMotorControlArray(kuka_id, kuka_joints, p.POSITION_CONTROL, 
                                    targetPositions=q_command, forces=[200]*7, positionGains=[0.05]*7)
        
        # --- Logika Grasping di Pucuk Tangan ---
        # Hitung posisi pucuk tangan (Wrist + Offset Z local)
        link_state = p.getLinkState(kuka_id, 6, computeForwardKinematics=True)
        wrist_pos = np.array(link_state[0])
        wrist_orn = np.array(link_state[1])
        
        # Hitung posisi TIP sebenarnya berdasarkan orientasi wrist saat ini
        # Rotasi vektor offset [0, 0, TIP_OFFSET] dengan quaternion wrist
        rot_matrix = p.getMatrixFromQuaternion(wrist_orn)
        rot_matrix = np.array(rot_matrix).reshape(3, 3)
        tip_offset_vec = np.dot(rot_matrix, [0, 0, TIP_OFFSET])
        current_tip_pos = wrist_pos + tip_offset_vec
        
        # Visualisasi titik pucuk tangan (Titik Hijau)
        p.addUserDebugLine(wrist_pos, current_tip_pos, [0,1,0], 2, lifeTime=0.1)

        cube_pos, _ = p.getBasePositionAndOrientation(cube_id)
        
        # Logic Grasp
        if gripper_schedule[t] == 1 and grasp_constraint is None:
            # Cek jarak antara CUBE dan TIP (Bukan Wrist)
            dist = np.linalg.norm(np.array(cube_pos) - current_tip_pos)
            
            if dist < 0.1: # Toleransi jarak
                # Buat constraint di ujung offset (Parent Frame: [0, 0, TIP_OFFSET])
                grasp_constraint = p.createConstraint(
                    kuka_id, 6, cube_id, -1, p.JOINT_FIXED, 
                    [0, 0, 0], 
                    [0, 0, TIP_OFFSET], # <-- POSISI PICK DI PUCUK TANGAN
                    [0, 0, 0]
                )
        elif gripper_schedule[t] == 0 and grasp_constraint is not None:
            p.removeConstraint(grasp_constraint)
            grasp_constraint = None
        
        p.stepSimulation()
        
        # Recording (Hanya iterasi Awal, Tengah, Akhir untuk hemat waktu/size)
        if (k == 0 or k == ITERATIONS//2 or k == ITERATIONS-1) and t % RECORD_EVERY_N_STEPS == 0:
            w, h, rgb, _, _ = p.getCameraImage(320, 240, view_matrix, proj_matrix, renderer=p.ER_BULLET_HARDWARE_OPENGL)
            frame_buffer.append(rgb)
            
        actual_trajectory.append(get_joint_positions())

    # Update ILC
    error = target_joint_trajectory - np.array(actual_trajectory)
    U_ilc = U_ilc + LEARNING_RATE * error
    print(f"   Mean Error: {np.mean(np.linalg.norm(error, axis=1)):.4f}")

p.disconnect()

# Simpan GIF
print("Menyimpan GIF...")
if len(frame_buffer) > 0:
    imageio.mimsave(GIF_NAME, frame_buffer, fps=25)
    print(f"Saved: {GIF_NAME}")
else:
    print("Tidak ada frame yang direkam.")