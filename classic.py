import pyray as rl
import math
import random
import numpy as np

# --- KONFIGURASI LAYAR & FISIK ---
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
BASE_POS = (400, 550) # Posisi robot disesuaikan ke tengah bawah window baru
PIXELS_PER_METER = 200.0

# Dimensi Robot
LINK_1 = 1.2 # Panjang lengan atas
LINK_2 = 1.0 # Panjang lengan bawah

# Warna
COLOR_BG = (30, 30, 30, 255)
COLOR_ROBOT = (220, 220, 220, 255)
COLOR_JOINT = (50, 50, 50, 255)
COLOR_RED = (230, 41, 55, 255)
COLOR_BLUE = (0, 121, 241, 255)
COLOR_YELLOW = (253, 249, 0, 255)
# Warna baru untuk visualisasi box
COLOR_SLOT_BG = (60, 60, 60, 255) # Latar belakang slot
COLOR_GRIP = (50, 255, 50, 255)   # Warna gripper saat menjepit

# --- MATEMATIKA ROBOT (Kinematika Murni) ---

def forward_kinematics(theta1, theta2):
    """Menghitung posisi (x, y) dari sudut sendi"""
    x1 = LINK_1 * math.cos(theta1)
    y1 = LINK_1 * math.sin(theta1)
    x2 = x1 + LINK_2 * math.cos(theta1 + theta2)
    y2 = y1 + LINK_2 * math.sin(theta1 + theta2)
    return (x1, y1), (x2, y2)

def inverse_kinematics(target_x, target_y):
    """Menghitung sudut (theta1, theta2) untuk mencapai (x, y)"""
    dist = math.sqrt(target_x**2 + target_y**2)
    max_reach = LINK_1 + LINK_2
    
    if dist > max_reach:
        scale = max_reach / dist
        target_x *= scale
        target_y *= scale
        dist = max_reach

    # Law of Cosines
    cos_angle2 = (target_x**2 + target_y**2 - LINK_1**2 - LINK_2**2) / (2 * LINK_1 * LINK_2)
    cos_angle2 = max(-1.0, min(1.0, cos_angle2))
    theta2 = math.acos(cos_angle2)
    
    k1 = LINK_1 + LINK_2 * math.cos(theta2)
    k2 = LINK_2 * math.sin(theta2)
    theta1 = math.atan2(target_y, target_x) - math.atan2(k2, k1)
    return theta1, theta2

def world_to_screen(x, y):
    """Mengubah koordinat meter ke pixel"""
    px = BASE_POS[0] + (x * PIXELS_PER_METER)
    py = BASE_POS[1] - (y * PIXELS_PER_METER)
    return int(px), int(py)

# --- LOGIKA GAME ---

def main():
    rl.init_window(SCREEN_WIDTH, SCREEN_HEIGHT, "2D Robot Sorter (Random Targets)")
    rl.set_target_fps(60)
    
    curr_theta1 = math.pi / 2
    curr_theta2 = 0.0
    
    # 1. Setup Kubus (Kiri) - Warna Acak
    cubes = []
    cube_colors = [COLOR_RED]*3 + [COLOR_BLUE]*3 + [COLOR_YELLOW]*3
    random.shuffle(cube_colors)
    
    for i in range(9):
        row, col = i // 3, i % 3
        cx = -1.8 + (col * 0.25)
        cy = 0.5 + (row * 0.25)
        cubes.append({"pos": [cx, cy], "color": cube_colors[i], "state": "IDLE"})
        
    # 2. Setup Slots (Kanan) - WARNA ACAK TOTAL
    slots = []
    # Buat pool warna target (3 Merah, 3 Biru, 3 Kuning)
    slot_palette = [COLOR_RED]*3 + [COLOR_BLUE]*3 + [COLOR_YELLOW]*3
    # Acak posisinya!
    random.shuffle(slot_palette)
    
    slot_index = 0
    for r in range(3):
        for c in range(3):
            sx = 1.0 + (c * 0.25)
            sy = 1.0 - (r * 0.25)
            
            # Ambil warna dari palette yang sudah diacak
            c_target = slot_palette[slot_index]
            slot_index += 1
            
            slots.append({"pos": [sx, sy], "color": c_target, "filled": False})

    # State Variables
    target_cube_idx = -1
    target_slot_idx = -1
    path = []
    path_idx = 0
    grip_active = False
    
    def generate_smooth_path(start_xy, end_xy, steps=60):
        """Membuat lintasan melengkung"""
        pts = []
        for i in range(steps):
            t = i / steps
            lx = start_xy[0] * (1-t) + end_xy[0] * t
            ly = start_xy[1] * (1-t) + end_xy[1] * t
            arc = 0.5 * math.sin(t * math.pi) # Lengkungan ke atas
            pts.append([lx, ly + arc])
        pts.append(end_xy)
        return pts

    # --- LOOP UTAMA ---
    while not rl.window_should_close():
        # 1. LOGIKA PILIH TUGAS
        if target_cube_idx == -1:
            for i, c in enumerate(cubes):
                if c["state"] == "IDLE":
                    # Cari slot kosong yang warnanya cocok
                    found_slot = False
                    for j, s in enumerate(slots):
                        if not s["filled"] and s["color"] == c["color"]:
                            target_cube_idx = i
                            target_slot_idx = j
                            rob_pos = forward_kinematics(curr_theta1, curr_theta2)[1]
                            path = generate_smooth_path(rob_pos, c["pos"], steps=40)
                            path_idx = 0
                            found_slot = True
                            break
                    if found_slot: break
        
        # 2. EKSEKUSI PERGERAKAN
        if target_cube_idx != -1 and path_idx < len(path):
            target_xy = path[path_idx]
            t1_target, t2_target = inverse_kinematics(target_xy[0], target_xy[1])
            # Gerakan smooth (lerp)
            curr_theta1 = curr_theta1 * 0.85 + t1_target * 0.15
            curr_theta2 = curr_theta2 * 0.85 + t2_target * 0.15
            
            if cubes[target_cube_idx]["state"] == "GRIPPED":
                hand_pos = forward_kinematics(curr_theta1, curr_theta2)[1]
                cubes[target_cube_idx]["pos"] = list(hand_pos)
            
            # Cek jarak untuk lanjut ke waypoint berikutnya
            dist_sq = (target_xy[0] - forward_kinematics(curr_theta1, curr_theta2)[1][0])**2 + \
                      (target_xy[1] - forward_kinematics(curr_theta1, curr_theta2)[1][1])**2
            if dist_sq < 0.005: path_idx += 1
                
        # 3. TRANSISI STATE (Grip/Lepas)
        elif target_cube_idx != -1 and path_idx >= len(path):
            cube = cubes[target_cube_idx]
            if cube["state"] == "IDLE": # Sampai di kubus -> GRIP
                cube["state"] = "GRIPPED"
                grip_active = True
                curr_pos = forward_kinematics(curr_theta1, curr_theta2)[1]
                dest_pos = slots[target_slot_idx]["pos"]
                path = generate_smooth_path(curr_pos, dest_pos, steps=60)
                path_idx = 0
            elif cube["state"] == "GRIPPED": # Sampai di slot -> LEPAS
                cube["state"] = "DONE"
                slots[target_slot_idx]["filled"] = True
                # Snap posisi kubus ke tengah slot agar rapi
                cube["pos"] = slots[target_slot_idx]["pos"]
                grip_active = False
                target_cube_idx = -1; target_slot_idx = -1

        # --- RENDER (GAMBAR) ---
        rl.begin_drawing()
        rl.clear_background(COLOR_BG)
        
        # 1. Gambar Slot Container (Kanan) - DIPERBAIKI
        for s in slots:
            sx, sy = world_to_screen(s["pos"][0], s["pos"][1])
            # Ukuran kotak slot (sedikit lebih besar dari kubus)
            slot_size = 22 
            
            # Gambar Latar Belakang Solid (Abu-abu gelap)
            rl.draw_rectangle(sx - slot_size, sy - slot_size, slot_size*2, slot_size*2, COLOR_SLOT_BG)
            
            # Gambar Border Tebal Sesuai Warna Target
            # Trik: Gambar 2 kotak garis dengan offset 1 pixel untuk efek tebal
            rl.draw_rectangle_lines(sx - slot_size, sy - slot_size, slot_size*2, slot_size*2, s["color"])
            rl.draw_rectangle_lines(sx - slot_size + 1, sy - slot_size + 1, slot_size*2 - 2, slot_size*2 - 2, s["color"])

            
        # 2. Gambar Kubus
        for c in cubes:
            cx, cy = world_to_screen(c["pos"][0], c["pos"][1])
            # Ukuran kubus (lebih kecil dari slot)
            cube_size = 15 
            rl.draw_rectangle(cx - cube_size, cy - cube_size, cube_size*2, cube_size*2, c["color"])
            # Garis pinggir hitam tipis untuk kubus
            rl.draw_rectangle_lines(cx - cube_size, cy - cube_size, cube_size*2, cube_size*2, rl.BLACK)
            
        # 3. Gambar Robot
        j1_xy = BASE_POS
        j2_m, j3_m = forward_kinematics(curr_theta1, curr_theta2)
        j2_xy = world_to_screen(j2_m[0], j2_m[1])
        j3_xy = world_to_screen(j3_m[0], j3_m[1])
        
        rl.draw_line_ex(j1_xy, j2_xy, 12, COLOR_ROBOT) # Lengan 1
        rl.draw_circle_v(j1_xy, 12, COLOR_JOINT)
        rl.draw_circle_v(j2_xy, 10, COLOR_JOINT)
        rl.draw_line_ex(j2_xy, j3_xy, 8, COLOR_ROBOT)  # Lengan 2
        
        # Gripper (Berubah warna saat menjepit)
        grip_col = COLOR_GRIP if grip_active else rl.WHITE
        rl.draw_circle_v(j3_xy, 10, grip_col)
        rl.draw_circle_lines(j3_xy[0], j3_xy[1], 12, rl.BLACK)
        
        # UI Text
        rl.draw_text("2D Robot Sorter (Random Targets)", 20, 20, 20, rl.WHITE)
        if all(s["filled"] for s in slots):
             rl.draw_text("SEMUA SELESAI!", SCREEN_WIDTH//2 - 100, SCREEN_HEIGHT//2, 30, rl.GREEN)
        
        rl.end_drawing()

    rl.close_window()

if __name__ == "__main__":
    main()