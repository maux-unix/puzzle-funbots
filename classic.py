import pybullet as p
import pybullet_data
import time
import numpy as np

# --- 1. Konfigurasi Simulasi PyBullet ---
def setup_pybullet():
    """Mengatur koneksi ke PyBullet dan memuat model dasar."""
    # Menghubungkan ke server simulasi (GUI untuk visualisasi)
    physicsClient = p.connect(p.GUI) 
    
    # Menambahkan direktori data tambahan (misalnya untuk model lantai, kubus)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    
    # Mengatur gravitasi
    p.setGravity(0, 0, -9.81)
    
    # Memuat bidang datar (lantai)
    p.loadURDF("plane.urdf")
    
    # Mengatur kecepatan langkah simulasi
    p.setTimeStep(1./240.) 
    
    print("PyBullet diinisialisasi.")
    return physicsClient

# --- 2. Memuat Robot dan Objek ---
def load_models():
    """Memuat model robot KUKA iiwa dan objek kubus."""
    
    # Memuat KUKA iiwa (konfigurasi sederhana tanpa gripper)
    # Gunakan KUKA iiwa yang sudah ada di pybullet_data
    robotId = p.loadURDF("kuka_lbr_iiwa/model.urdf", [0, 0, 0], useFixedBase=True)
    
    # Menemukan End-Effector (EE) Link Index.
    # KUKA iiwa standar memiliki 7 sendi (joint), dan link ke-7 (index 6) adalah end-effector.
    # Kita perlu tahu ini untuk Inverse Kinematics (IK).
    EE_LINK_INDEX = 6 
    
    # Membuat objek yang akan dipindahkan (misalnya, kubus merah)
    obj_mass = 0.1
    obj_visual_shape = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.02, 0.02, 0.02], rgbaColor=[1, 0, 0, 1])
    obj_collision_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.02, 0.02, 0.02])
    
    # Posisi awal objek (di depan robot)
    obj_start_pos = [0.4, 0.2, 0.02] 
    
    objectId = p.createMultiBody(obj_mass, obj_collision_shape, obj_visual_shape, obj_start_pos)
    
    # Mengatur konfigurasi awal sendi robot (posisi istirahat/ready)
    initial_joint_positions = [0, 0, 0, -1.57, 0, 1.57, 0]
    for jointIndex in range(p.getNumJoints(robotId)):
        p.resetJointState(robotId, jointIndex, initial_joint_positions[jointIndex])
        
    return robotId, EE_LINK_INDEX, objectId

# --- 3. Fungsi Kontrol Gerak ---
def move_to_target(robotId, ee_link_index, target_pos, target_orn=None):
    """
    Menghitung Inverse Kinematics (IK) dan menggerakkan robot ke posisi target.
    """
    
    # Orientasi default (lengan ke bawah) jika tidak ditentukan
    if target_orn is None:
        # Orientasi End-Effector yang diinginkan (Quaternions). 
        # Di sini, EE menghadap ke bawah, sejajar sumbu Z global.
        target_orn = p.getQuaternionFromEuler([np.pi/2, 0, 0]) 
    
    # Menghitung posisi sendi yang diperlukan menggunakan Inverse Kinematics (IK)
    # PyBullet menyelesaikan IK untuk menemukan posisi sendi
    joint_poses = p.calculateInverseKinematics(
        bodyUniqueId=robotId,
        endEffectorLinkIndex=ee_link_index,
        targetPosition=target_pos,
        targetOrientation=target_orn
    )
    
    # Mengatur posisi sendi (Joint Control)
    # Menggerakkan sendi robot ke posisi yang dihitung oleh IK
    p.setJointMotorControlArray(
        bodyUniqueId=robotId,
        jointIndices=range(p.getNumJoints(robotId)),
        controlMode=p.POSITION_CONTROL,
        targetPositions=joint_poses,
        # Kecepatan dan gaya maksimum (optional, untuk kontrol yang lebih baik)
        forces=[500]*p.getNumJoints(robotId) 
    )

# --- 4. Fungsi Utama Simulasi Pick-and-Place ---
def run_simulation():
    # Posisi target (X, Y, Z)
    # 1. Posisi Pick (Tepat di atas objek)
    PICK_POS = [0.4, 0.2, 0.2] 
    
    # 2. Posisi Place (Lokasi lain)
    PLACE_POS = [0.4, -0.4, 0.2]
    
    # 3. Posisi Aman (Posisi di atas Pick/Place untuk menghindari rintangan)
    SAFE_Z = 0.4
    
    # Inisialisasi
    physicsClient = setup_pybullet()
    robotId, ee_link_index, objectId = load_models()
    
    # Memberi waktu bagi simulasi untuk stabil di posisi awal
    time.sleep(1) 

    # --- SIKLUS PICK-AND-PLACE ---

    # --- A. Pergi ke posisi aman di atas Pick ---
    print("\n[STEP 1] Bergerak ke posisi aman di atas objek...")
    move_to_target(robotId, ee_link_index, [PICK_POS[0], PICK_POS[1], SAFE_Z])
    
    # Loop untuk menunggu robot mencapai posisi IK yang dihitung
    for _ in range(240): # Berjalan selama 1 detik (240 steps)
        p.stepSimulation()
        time.sleep(1./240.)
        
    # --- B. Turun ke posisi Pick ---
    print("[STEP 2] Menurunkan end-effector ke objek...")
    move_to_target(robotId, ee_link_index, PICK_POS) 
    
    # Loop dan tunggu
    for _ in range(240):
        p.stepSimulation()
        time.sleep(1./240.)
    
    # --- C. 'Grasp' (Mengambil) Objek ---
    # Di PyBullet, "mengambil" dilakukan dengan menambahkan kendala/constraint (fixed joint)
    print("[STEP 3] Mengambil objek (menambahkan constraint)...")
    # Constraint antara link EE robot dan objek
    cid = p.createConstraint(
        parentBodyUniqueId=robotId, 
        parentLinkIndex=ee_link_index, 
        childBodyUniqueId=objectId, 
        childLinkIndex=-1, # Objek adalah base body (-1)
        jointType=p.JOINT_FIXED,
        jointAxis=[0, 0, 0],
        parentFramePosition=[0, 0, 0],
        childFramePosition=[0, 0, 0]
    )
    
    time.sleep(0.5) # Jeda visual

    # --- D. Kembali ke posisi aman (Lift) ---
    print("[STEP 4] Mengangkat objek ke posisi aman...")
    move_to_target(robotId, ee_link_index, [PICK_POS[0], PICK_POS[1], SAFE_Z])
    
    # Loop dan tunggu
    for _ in range(240):
        p.stepSimulation()
        time.sleep(1./240.)

    # --- E. Pindah ke posisi aman di atas Place ---
    print("[STEP 5] Memindahkan objek ke atas posisi Place...")
    move_to_target(robotId, ee_link_index, [PLACE_POS[0], PLACE_POS[1], SAFE_Z])
    
    # Loop dan tunggu
    for _ in range(480): # Waktu tunggu lebih lama untuk pergerakan lateral
        p.stepSimulation()
        time.sleep(1./240.)

    # --- F. Turun ke posisi Place ---
    print("[STEP 6] Menurunkan objek ke posisi Place...")
    move_to_target(robotId, ee_link_index, PLACE_POS) 
    
    # Loop dan tunggu
    for _ in range(240):
        p.stepSimulation()
        time.sleep(1./240.)

    # --- G. 'Release' (Melepas) Objek ---
    print("[STEP 7] Melepas objek (menghapus constraint)...")
    p.removeConstraint(cid)
    
    time.sleep(0.5) # Jeda visual

    # --- H. Kembali ke posisi aman dan selesai ---
    print("[STEP 8] Kembali ke posisi aman dan simulasi selesai.")
    move_to_target(robotId, ee_link_index, [PLACE_POS[0], PLACE_POS[1], SAFE_Z]) 
    
    # Loop dan tunggu
    for _ in range(240): 
        p.stepSimulation()
        time.sleep(1./240.)
        
    print("\nSimulasi Selesai. Tekan Ctrl+C di terminal untuk keluar.")
    
    # Agar simulasi tetap terbuka sampai user menutupnya
    while p.isConnected():
        p.stepSimulation()
        time.sleep(1./240.)

if __name__ == "__main__":
    try:
        run_simulation()
    except p.error as e:
        print(f"Error PyBullet: {e}")
    finally:
        # Memutuskan koneksi server saat selesai atau terjadi error
        p.disconnect()