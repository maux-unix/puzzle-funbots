import pybullet as p
import pybullet_data
import time

# --- Konfigurasi Simulasi ---
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.8)
plane_id = p.loadURDF("plane.urdf")

# --- Fungsi untuk Membuat Container Kayu 3x3 ---
def create_wooden_container_3x3(center_pos, size_x=0.6, size_y=0.6, height=0.1, thickness=0.02):
    """
    Membuat container 3x3 dari balok-balok kayu.
    
    Args:
        center_pos (list): Posisi pusat container [x, y, z] (posisi alasnya).
        size_x (float): Total panjang container (sumbu x).
        size_y (float): Total lebar container (sumbu y).
        height (float): Tinggi dinding container.
        thickness (float): Ketebalan dinding dan penyekat.
    """
    
    # Warna kayu (coklat)
    wood_color = [0.6, 0.4, 0.2, 1.0]

    # Fungsi bantuan untuk membuat satu dinding/balok
    def make_box(pos, half_extents):
        vis_shape = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents, rgbaColor=wood_color)
        col_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents)
        body_id = p.createMultiBody(baseMass=0, # Massa 0 agar statis
                                    baseVisualShapeIndex=vis_shape,
                                    baseCollisionShapeIndex=col_shape,
                                    basePosition=pos)
        return body_id

    # --- 1. Membuat Alas ---
    # Alas diletakkan sedikit di bawah center_pos agar center_pos adalah permukaan atas alas
    base_z = center_pos[2] - thickness / 2
    make_box([center_pos[0], center_pos[1], base_z], 
             [size_x / 2, size_y / 2, thickness / 2])

    # --- 2. Membuat Dinding Luar ---
    wall_z = center_pos[2] + height / 2
    
    # Dinding Depan & Belakang (sepanjang sumbu X)
    make_box([center_pos[0], center_pos[1] + size_y/2 - thickness/2, wall_z], 
             [size_x / 2, thickness / 2, height / 2]) # Belakang
    make_box([center_pos[0], center_pos[1] - size_y/2 + thickness/2, wall_z], 
             [size_x / 2, thickness / 2, height / 2]) # Depan
             
    # Dinding Kiri & Kanan (sepanjang sumbu Y)
    # Panjangnya dikurangi ketebalan dinding depan/belakang agar pas
    side_wall_len = size_y / 2 - thickness
    make_box([center_pos[0] - size_x/2 + thickness/2, center_pos[1], wall_z], 
             [thickness / 2, side_wall_len, height / 2]) # Kiri
    make_box([center_pos[0] + size_x/2 - thickness/2, center_pos[1], wall_z], 
             [thickness / 2, side_wall_len, height / 2]) # Kanan

    # --- 3. Membuat Penyekat Dalam (Dividers) ---
    # Kita butuh 2 penyekat horizontal dan 2 vertikal untuk membuat grid 3x3
    
    # Ukuran sel (ruang kosong di dalam)
    cell_size_x = (size_x - 4 * thickness) / 3
    cell_size_y = (size_y - 4 * thickness) / 3
    
    # Penyekat sepanjang sumbu X (Horizontal)
    divider_len_x = size_x / 2 - thickness # Agar pas di dalam dinding samping
    y_offset_1 = -size_y/2 + thickness + cell_size_y + thickness/2
    y_offset_2 = y_offset_1 + cell_size_y + thickness

    make_box([center_pos[0], center_pos[1] + y_offset_1, wall_z], 
             [divider_len_x, thickness / 2, height / 2])
    make_box([center_pos[0], center_pos[1] + y_offset_2, wall_z], 
             [divider_len_x, thickness / 2, height / 2])
             
    # Penyekat sepanjang sumbu Y (Vertikal)
    divider_len_y = size_y / 2 - thickness # Agar pas di dalam dinding depan/belakang
    x_offset_1 = -size_x/2 + thickness + cell_size_x + thickness/2
    x_offset_2 = x_offset_1 + cell_size_x + thickness

    make_box([center_pos[0] + x_offset_1, center_pos[1], wall_z], 
             [thickness / 2, divider_len_y, height / 2])
    make_box([center_pos[0] + x_offset_2, center_pos[1], wall_z], 
             [thickness / 2, divider_len_y, height / 2])

# --- Jalankan Simulasi ---
# Posisi container di dunia
container_position = [0.5, 0.0, 0.02] # Sedikit di atas tanah

# Buat container
create_wooden_container_3x3(container_position)

print("Container 3x3 berhasil dibuat.")

# Loop agar jendela simulasi tidak langsung tertutup
while True:
    p.stepSimulation()
    time.sleep(1./240.)