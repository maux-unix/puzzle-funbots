import pyray as rl
import math
import random
import numpy as np
import torch
import torch.nn as nn
import os

# --- KONFIGURASI ---
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
BASE_POS = (400, 550) 
PIXELS_PER_METER = 200.0

# Nama file model yang akan dites (Pastikan file ini ada!)
MODEL_PATH = "ilc_online_learner.pth" 

# Fisika Robot
LINK_1 = 1.2 
LINK_2 = 1.0 
GRAVITY_VAL = 0.20 # Gravitasi harus SAMA dengan saat training untuk melihat hasilnya

# Network Param
HIDDEN_SIZE = 64

# Warna
COLOR_BG = (30, 30, 30, 255)
COLOR_ROBOT = (220, 220, 220, 255)
COLOR_GHOST = (0, 255, 0, 100)
COLOR_JOINT = (50, 50, 50, 255)
COLOR_RED = (230, 41, 55, 255)
COLOR_BLUE = (0, 121, 241, 255)
COLOR_YELLOW = (253, 249, 0, 255)
COLOR_SLOT_BG = (60, 60, 60, 255)
COLOR_GRIP = (50, 255, 50, 255)

# --- 1. NEURAL NETWORK (Arsitektur HARUS SAMA) ---
class ILCNetwork(nn.Module):
    def __init__(self):
        super(ILCNetwork, self).__init__()
        self.fc1 = nn.Linear(4, HIDDEN_SIZE)
        self.fc2 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.fc3 = nn.Linear(HIDDEN_SIZE, 2)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

# --- 2. FISIKA & KINEMATIKA ---
def forward_kinematics(theta1, theta2):
    x1 = LINK_1 * math.cos(theta1)
    y1 = LINK_1 * math.sin(theta1)
    x2 = x1 + LINK_2 * math.cos(theta1 + theta2)
    y2 = y1 + LINK_2 * math.sin(theta1 + theta2)
    return (float(x1), float(y1)), (float(x2), float(y2))

def inverse_kinematics(target_x, target_y):
    dist = math.sqrt(target_x**2 + target_y**2)
    max_reach = LINK_1 + LINK_2
    if dist > max_reach:
        scale = max_reach / dist
        target_x *= scale
        target_y *= scale

    try:
        cos_angle2 = (target_x**2 + target_y**2 - LINK_1**2 - LINK_2**2) / (2 * LINK_1 * LINK_2)
        cos_angle2 = max(-1.0, min(1.0, cos_angle2))
        theta2 = math.acos(cos_angle2)
        k1 = LINK_1 + LINK_2 * math.cos(theta2)
        k2 = LINK_2 * math.sin(theta2)
        theta1 = math.atan2(target_y, target_x) - math.atan2(k2, k1)
        return np.array([float(theta1), float(theta2)], dtype=np.float32)
    except:
        return np.array([math.pi/2, 0], dtype=np.float32)

def world_to_screen(x, y):
    px = BASE_POS[0] + (x * PIXELS_PER_METER)
    py = BASE_POS[1] - (y * PIXELS_PER_METER)
    return int(px), int(py)

def apply_gravity_disturbance(theta1, theta2):
    # Simulasi gangguan yang sama persis
    act_t1 = float(theta1) - (GRAVITY_VAL * math.cos(theta1)) 
    act_t2 = float(theta2) - (GRAVITY_VAL * 0.5 * math.cos(theta1 + theta2))
    return np.array([act_t1, act_t2], dtype=np.float32)

# --- 3. LOGIKA UTAMA ---
def main():
    rl.init_window(SCREEN_WIDTH, SCREEN_HEIGHT, "TEST MODE: Neural ILC")
    rl.set_target_fps(60)
    
    # Init & Load Model
    net = ILCNetwork()
    model_loaded = False
    
    if os.path.exists(MODEL_PATH):
        try:
            checkpoint = torch.load(MODEL_PATH)
            # Handle format save yang berbeda (dict vs state_dict langsung)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                net.load_state_dict(checkpoint['model_state_dict'])
                print(f"Model Loaded. Best Loss saat training: {checkpoint.get('best_loss', 'N/A')}")
            else:
                net.load_state_dict(checkpoint)
            
            net.eval() # PENTING: Set ke mode evaluasi (matikan dropout/batchnorm jika ada)
            model_loaded = True
        except Exception as e:
            print(f"Gagal load model: {e}")
    else:
        print(f"File {MODEL_PATH} tidak ditemukan! Pastikan sudah training dulu.")

    curr_theta = np.array([math.pi / 2, 0.0])
    
    # Statistik Test
    total_cubes = 0
    success_cubes = 0
    
    def reset_level():
        c_list = []
        cols = [COLOR_RED]*3 + [COLOR_BLUE]*3 + [COLOR_YELLOW]*3
        random.shuffle(cols)
        for i in range(9):
            r, c = i // 3, i % 3
            cx = -1.8 + (c * 0.25)
            cy = 0.5 + (r * 0.25)
            c_list.append({"pos": [cx, cy], "color": cols[i], "state": "IDLE"})
            
        s_list = []
        s_cols = [COLOR_RED]*3 + [COLOR_BLUE]*3 + [COLOR_YELLOW]*3
        random.shuffle(s_cols) 
        idx = 0
        for r in range(3):
            for c in range(3):
                sx = 1.0 + (c * 0.25)
                sy = 1.0 - (r * 0.25)
                s_list.append({"pos": [sx, sy], "color": s_cols[idx], "filled": False})
                idx += 1
        return c_list, s_list

    cubes, slots = reset_level()
    
    target_cube_idx = -1
    target_slot_idx = -1
    path = []
    path_idx = 0
    grip_active = False
    
    def generate_smooth_path(start_xy, end_xy, steps=50):
        pts = []
        for i in range(steps):
            t = i / steps
            lx = start_xy[0] * (1-t) + end_xy[0] * t
            ly = start_xy[1] * (1-t) + end_xy[1] * t
            arc = 0.6 * math.sin(t * math.pi)
            pts.append([lx, ly + arc])
        pts.append(end_xy)
        return pts

    while not rl.window_should_close():
        
        # Speed Control
        speed_multiplier = 5 if rl.is_key_down(rl.KEY_SPACE) else 1
        
        for _ in range(speed_multiplier):
            
            # 1. Cari Tugas
            if target_cube_idx == -1:
                found_task = False
                for i, c in enumerate(cubes):
                    if c["state"] == "IDLE":
                        for j, s in enumerate(slots):
                            if not s["filled"] and s["color"] == c["color"]:
                                target_cube_idx = i
                                target_slot_idx = j
                                rob_pos = forward_kinematics(curr_theta[0], curr_theta[1])[1]
                                path = generate_smooth_path(rob_pos, c["pos"], steps=40)
                                path_idx = 0
                                found_task = True
                                break
                        if found_task: break
                
                if not found_task:
                    cubes, slots = reset_level() # Reset level jika habis

            # 2. Eksekusi Gerakan (INFERENCE)
            if target_cube_idx != -1 and path_idx < len(path):
                target_xy = path[path_idx]
                
                # A. Hitung Target Ideal
                ideal_theta = inverse_kinematics(target_xy[0], target_xy[1])
                
                # B. MINTA KOREKSI DARI MODEL (Jika ada)
                correction = np.array([0.0, 0.0])
                if model_loaded:
                    nn_input = torch.tensor(np.concatenate([curr_theta, ideal_theta]), dtype=torch.float32)
                    with torch.no_grad(): # MATIKAN GRADIENT (Hemat memori, mode test)
                        correction = net(nn_input).numpy()
                
                # C. Command + Disturbance
                command_theta = ideal_theta + correction
                
                # D. Simulasi Gangguan (Gravitasi tetap ada!)
                actual_theta = apply_gravity_disturbance(command_theta[0], command_theta[1])
                
                curr_theta = actual_theta
                
                if cubes[target_cube_idx]["state"] == "GRIPPED":
                    hand_pos = forward_kinematics(curr_theta[0], curr_theta[1])[1]
                    cubes[target_cube_idx]["pos"] = list(hand_pos)
                
                path_idx += 1
                    
            # 3. Transisi
            elif target_cube_idx != -1 and path_idx >= len(path):
                cube = cubes[target_cube_idx]
                if cube["state"] == "IDLE": 
                    cube["state"] = "GRIPPED"
                    grip_active = True
                    curr_pos = forward_kinematics(curr_theta[0], curr_theta[1])[1]
                    dest_pos = slots[target_slot_idx]["pos"]
                    path = generate_smooth_path(curr_pos, dest_pos, steps=60)
                    path_idx = 0
                elif cube["state"] == "GRIPPED":
                    cube["state"] = "DONE"
                    grip_active = False
                    slots[target_slot_idx]["filled"] = True 
                    
                    # Hitung Statistik Sukses
                    final_x, final_y = cube["pos"]
                    target_x, target_y = slots[target_slot_idx]["pos"]
                    dist_error = math.sqrt((final_x - target_x)**2 + (final_y - target_y)**2)
                    
                    total_cubes += 1
                    # Jika error < 5cm, anggap sukses masuk kotak
                    if dist_error < 0.05:
                        success_cubes += 1
                    
                    target_cube_idx = -1; target_slot_idx = -1

        # --- RENDER ---
        rl.begin_drawing()
        rl.clear_background(COLOR_BG)
        
        # Slots
        for s in slots:
            sx, sy = world_to_screen(s["pos"][0], s["pos"][1])
            rl.draw_rectangle(sx-20, sy-20, 40, 40, COLOR_SLOT_BG)
            rl.draw_rectangle_lines_ex(rl.Rectangle(sx-20, sy-20, 40, 40), 2, s["color"])
        # Cubes
        for c in cubes:
            cx, cy = world_to_screen(c["pos"][0], c["pos"][1])
            rl.draw_rectangle(cx-14, cy-14, 28, 28, c["color"])
            rl.draw_rectangle_lines(cx-14, cy-14, 28, 28, rl.BLACK)
            
        # Robot
        j1_xy = BASE_POS
        j2_m, j3_m = forward_kinematics(curr_theta[0], curr_theta[1])
        j2_xy = world_to_screen(j2_m[0], j2_m[1])
        j3_xy = world_to_screen(j3_m[0], j3_m[1])
        
        rl.draw_line_ex(j1_xy, j2_xy, 10, COLOR_ROBOT)
        rl.draw_circle_v(j1_xy, 8, COLOR_JOINT)
        rl.draw_circle_v(j2_xy, 7, COLOR_JOINT)
        rl.draw_line_ex(j2_xy, j3_xy, 6, COLOR_ROBOT)
        rl.draw_circle_v(j3_xy, 8, COLOR_GRIP if grip_active else rl.WHITE)
        
        # --- UI TEST ---
        if model_loaded:
            rl.draw_text("MODE: TESTING (INFERENCE)", 10, 10, 20, rl.GREEN)
        else:
            rl.draw_text("MODE: NO MODEL (GRAVITY ERROR)", 10, 10, 20, rl.RED)
            
        rl.draw_text(f"Processed: {total_cubes}", 10, 40, 20, rl.WHITE)
        
        acc = 0.0
        if total_cubes > 0:
            acc = (success_cubes / total_cubes) * 100.0
            
        col_acc = rl.GREEN if acc > 90 else (rl.YELLOW if acc > 70 else rl.RED)
        rl.draw_text(f"Accuracy: {acc:.1f}%", 10, 65, 20, col_acc)
        
        rl.draw_text("Space: Turbo", SCREEN_WIDTH - 150, 10, 20, rl.GRAY)
        
        rl.end_drawing()

    rl.close_window()

if __name__ == "__main__":
    main()