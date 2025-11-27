import pyray as rl
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# --- KONFIGURASI ---
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
BASE_POS = (400, 550) 
PIXELS_PER_METER = 200.0
MODEL_PATH = "ilc_smart_stop.pth" 

# --- GRAVITASI ---
GRAVITY_VAL = 0.20 

# --- KECEPATAN ---
SPEED_LEVELS = [1, 2, 5, 10, 20, 50, 100]

# --- PARAMETER BARU ---
EARLY_STOP_THRESHOLD = 0.00005  # Jika error di bawah ini, stop training (sudah pintar)
ACCEPTABLE_LOSS = 0.002         # Batas toleransi untuk menyimpan model "Cukup Bagus"
SLOT_RADIUS = 0.06              # Setengah lebar slot (batas aman border)

# Fisika Robot
LINK_1 = 1.2 
LINK_2 = 1.0 

# ILC Parameter
LEARNING_RATE = 0.005
HIDDEN_SIZE = 64

# Warna
COLOR_BG = (30, 30, 30, 255)
COLOR_ROBOT = (220, 220, 220, 255)
COLOR_GHOST = (0, 255, 0, 100)
COLOR_JOINT = (50, 50, 50, 255)
RED_RGB = (230, 41, 55, 255)
BLUE_RGB = (0, 121, 241, 255)
YELLOW_RGB = (253, 249, 0, 255)
COLOR_SLOT_BG = (60, 60, 60, 255)
COLOR_GRIP = (50, 255, 50, 255)

COLOR_MAP = {RED_RGB: 0, BLUE_RGB: 1, YELLOW_RGB: 2}
COLOR_NAMES = ["Merah", "Biru", "Kuning"]

# --- 1. NEURAL NETWORK ---
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
    act_t1 = float(theta1) - (GRAVITY_VAL * math.cos(theta1)) 
    act_t2 = float(theta2) - (GRAVITY_VAL * 0.5 * math.cos(theta1 + theta2))
    return np.array([act_t1, act_t2], dtype=np.float32)

# --- 3. DASHBOARD VISUALIZATION ---
plt.ion()
fig = plt.figure(figsize=(10, 8))
gs = GridSpec(2, 2, figure=fig)

ax_loss = fig.add_subplot(gs[0, 0])
ax_score = fig.add_subplot(gs[0, 1])
ax_cm = fig.add_subplot(gs[1, :])

history_loss = []
history_score = []
# Matrix Terbaik Global
best_display_matrix = np.zeros((3, 3), dtype=int) 

def update_dashboard(curr_iter, best_loss_val):
    ax_loss.clear(); ax_score.clear(); ax_cm.clear()
    
    if history_loss:
        ax_loss.plot(history_loss, color='red', label='Per-Cube Loss')
        ax_loss.axhline(y=EARLY_STOP_THRESHOLD, color='blue', linestyle=':', label='Early Stop')
        ax_loss.axhline(y=best_loss_val, color='green', linestyle='--', label='Best Record')
        ax_loss.set_title("Training Loss & Thresholds")
        ax_loss.legend(loc='upper right')
        ax_loss.set_yscale('log') # Pakai skala log biar kelihatan detail kecil
        ax_loss.grid(True, alpha=0.3)

    if history_score:
        ax_score.plot(history_score, color='blue', marker='o', linestyle='none', markersize=2)
        ax_score.set_title("Score Per Cube")
        ax_score.set_ylim(0, 105)
        ax_score.grid(True, alpha=0.3)

    im = ax_cm.imshow(best_display_matrix, cmap='Greens', vmin=0)
    total_samples = np.sum(best_display_matrix)
    accuracy = 0
    if total_samples > 0:
        accuracy = np.trace(best_display_matrix) / total_samples * 100.0
        
    ax_cm.set_title(f"Best Confusion Matrix (Acc: {accuracy:.1f}%)")
    ax_cm.set_xticks(np.arange(3)); ax_cm.set_yticks(np.arange(3))
    ax_cm.set_xticklabels(COLOR_NAMES); ax_cm.set_yticklabels(COLOR_NAMES)
    ax_cm.set_xlabel("Prediksi"); ax_cm.set_ylabel("Aktual")
    
    for i in range(3):
        for j in range(3):
            count = best_display_matrix[i, j]
            pct = 0.0
            row_sum = np.sum(best_display_matrix[i, :])
            if row_sum > 0: pct = (count / row_sum) * 100.0
            text_label = f"{count}\n({pct:.0f}%)"
            ax_cm.text(j, i, text_label, ha="center", va="center", color="black", fontsize=10)
            
    plt.tight_layout()
    plt.pause(0.001)

# --- 4. LOGIKA GAME ---
def main():
    rl.init_window(SCREEN_WIDTH, SCREEN_HEIGHT, "2D ILC (Early Stop + Smart Save)")
    rl.set_target_fps(60)
    
    net = ILCNetwork()
    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()
    
    best_loss = float('inf')
    
    # Load Model
    global best_display_matrix
    if os.path.exists(MODEL_PATH):
        try:
            checkpoint = torch.load(MODEL_PATH)
            net.load_state_dict(checkpoint['model_state_dict'])
            best_loss = checkpoint.get('best_loss', float('inf'))
            best_display_matrix = np.array(checkpoint.get('best_cm', np.zeros((3,3))))
            net.eval()
            print(f"Model Loaded. Best Loss: {best_loss:.5f}")
        except Exception as e:
            print(f"Error loading: {e}")
    
    curr_theta = np.array([math.pi / 2, 0.0])
    
    # Data Buffer Per Kubus
    cube_data_X = []
    cube_data_Y = []
    
    current_iter_matrix = np.zeros((3, 3), dtype=int)
    
    iteration = 1
    speed_level_index = 0
    speed_multiplier = SPEED_LEVELS[speed_level_index]
    
    save_msg = "Ready"
    save_col = rl.GRAY
    
    # Variabel untuk validasi posisi akhir kubus
    last_final_pos = [0,0]
    last_target_pos = [0,0]
    
    def reset_level():
        c_list = []
        cols = [RED_RGB]*3 + [BLUE_RGB]*3 + [YELLOW_RGB]*3
        random.shuffle(cols)
        for i in range(9):
            r, c = i // 3, i % 3
            cx = -1.8 + (c * 0.25)
            cy = 0.5 + (r * 0.25)
            c_list.append({"pos": [cx, cy], "color": cols[i], "state": "IDLE"})
            
        s_list = []
        s_cols = [RED_RGB]*3 + [BLUE_RGB]*3 + [YELLOW_RGB]*3
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
    
    is_training_moment = False
    train_moment_timer = 0
    
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

    update_dashboard(iteration, best_loss)

    while not rl.window_should_close():
        
        if rl.is_key_pressed(rl.KEY_RIGHT):
            speed_level_index = min(len(SPEED_LEVELS) - 1, speed_level_index + 1)
        if rl.is_key_pressed(rl.KEY_LEFT):
            speed_level_index = max(0, speed_level_index - 1)
        speed_multiplier = SPEED_LEVELS[speed_level_index]
        
        for _ in range(speed_multiplier):
            
            if is_training_moment:
                train_moment_timer += 1
                if train_moment_timer > 5:
                    if len(cube_data_X) > 0:
                        inp = torch.tensor(np.array(cube_data_X), dtype=torch.float32)
                        tgt = torch.tensor(np.array(cube_data_Y), dtype=torch.float32)
                        
                        net.train()
                        current_cube_loss = 0
                        
                        # --- EARLY STOPPING LOGIC ---
                        # Train max 30 epoch, tapi stop jika error sudah sangat kecil
                        for epoch in range(30):
                            optimizer.zero_grad()
                            out = net(inp)
                            loss = criterion(out, tgt)
                            loss.backward()
                            optimizer.step()
                            current_cube_loss = loss.item()
                            
                            if current_cube_loss < EARLY_STOP_THRESHOLD:
                                # Stop training loop ini
                                break
                        
                        history_loss.append(current_cube_loss)
                        
                        # --- SAVING LOGIC (RELAXED) ---
                        # 1. Hitung jarak fisik kubus ke target
                        dist_phys = math.sqrt((last_final_pos[0] - last_target_pos[0])**2 + 
                                              (last_final_pos[1] - last_target_pos[1])**2)
                        
                        # Apakah kubus masuk dalam radius toleransi (meski mepet pinggir)?
                        is_physically_good = dist_phys < SLOT_RADIUS
                        
                        # Logic Save:
                        # - Jika memecahkan rekor loss (BEST)
                        # - ATAU Jika secara fisik sukses DAN loss-nya wajar (ACCEPTABLE)
                        should_save = False
                        status_label = ""
                        
                        if current_cube_loss < best_loss:
                            best_loss = current_cube_loss
                            should_save = True
                            status_label = "NEW RECORD!"
                            save_col = rl.GREEN
                        elif is_physically_good and current_cube_loss < ACCEPTABLE_LOSS:
                            should_save = True
                            status_label = "GOOD ENOUGH (Saved)"
                            save_col = rl.BLUE
                        else:
                            save_msg = f"Skipped (L:{current_cube_loss:.4f} D:{dist_phys:.2f})"
                            save_col = rl.ORANGE

                        if should_save:
                            if np.sum(current_iter_matrix) > 0:
                                best_display_matrix = np.copy(current_iter_matrix)
                            
                            save_data = {
                                'model_state_dict': net.state_dict(),
                                'best_loss': best_loss,
                                'best_cm': best_display_matrix
                            }
                            torch.save(save_data, MODEL_PATH)
                            save_msg = f"{status_label} Loss: {current_cube_loss:.5f}"

                    cube_data_X = []
                    cube_data_Y = []
                    is_training_moment = False
                    update_dashboard(iteration, best_loss)
                
                continue 

            # --- LOGIKA FISIKA ---
            
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
                    cubes, slots = reset_level()
                    current_iter_matrix = np.zeros((3, 3), dtype=int) 
                    iteration += 1

            if target_cube_idx != -1 and path_idx < len(path):
                target_xy = path[path_idx]
                ideal_theta = inverse_kinematics(target_xy[0], target_xy[1])
                
                nn_input = torch.tensor(np.concatenate([curr_theta, ideal_theta]), dtype=torch.float32)
                with torch.no_grad():
                    correction = net(nn_input).numpy()
                
                command_theta = ideal_theta + correction
                actual_theta = apply_gravity_disturbance(command_theta[0], command_theta[1])
                
                error = ideal_theta - actual_theta
                cube_data_X.append(np.concatenate([curr_theta, ideal_theta]))
                cube_data_Y.append(error)
                
                curr_theta = actual_theta
                
                if cubes[target_cube_idx]["state"] == "GRIPPED":
                    hand_pos = forward_kinematics(curr_theta[0], curr_theta[1])[1]
                    cubes[target_cube_idx]["pos"] = list(hand_pos)
                
                path_idx += 1
                    
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
                    
                    last_final_pos = cube["pos"]
                    last_target_pos = slots[target_slot_idx]["pos"]
                    
                    dist_error = math.sqrt((last_final_pos[0] - last_target_pos[0])**2 + 
                                           (last_final_pos[1] - last_target_pos[1])**2)
                    reward = max(0, (0.15 - dist_error) * (100 / 0.15))
                    history_score.append(reward)
                    
                    act_idx = COLOR_MAP[cube["color"]]
                    pred_idx = -1
                    for s_check in slots:
                        sx, sy = s_check["pos"]
                        d = math.sqrt((last_final_pos[0] - sx)**2 + (last_final_pos[1] - sy)**2)
                        if d < 0.12: 
                            pred_idx = COLOR_MAP[s_check["color"]]
                            s_check["filled"] = True 
                            break
                    if pred_idx != -1:
                        current_iter_matrix[act_idx][pred_idx] += 1
                    slots[target_slot_idx]["filled"] = True 
                    
                    target_cube_idx = -1; target_slot_idx = -1
                    is_training_moment = True
                    train_moment_timer = 0

        # --- RENDER ---
        rl.begin_drawing()
        rl.clear_background(COLOR_BG)
        
        for s in slots:
            sx, sy = world_to_screen(s["pos"][0], s["pos"][1])
            rl.draw_rectangle(sx-20, sy-20, 40, 40, COLOR_SLOT_BG)
            rl.draw_rectangle_lines_ex(rl.Rectangle(sx-20, sy-20, 40, 40), 2, s["color"])
        for c in cubes:
            cx, cy = world_to_screen(c["pos"][0], c["pos"][1])
            rl.draw_rectangle(cx-14, cy-14, 28, 28, c["color"])
            rl.draw_rectangle_lines(cx-14, cy-14, 28, 28, rl.BLACK)
            
        j1_xy = BASE_POS
        j2_m, j3_m = forward_kinematics(curr_theta[0], curr_theta[1])
        j2_xy = world_to_screen(j2_m[0], j2_m[1])
        j3_xy = world_to_screen(j3_m[0], j3_m[1])
        
        rl.draw_line_ex(j1_xy, j2_xy, 10, COLOR_ROBOT)
        rl.draw_circle_v(j1_xy, 8, COLOR_JOINT)
        rl.draw_circle_v(j2_xy, 7, COLOR_JOINT)
        rl.draw_line_ex(j2_xy, j3_xy, 6, COLOR_ROBOT)
        rl.draw_circle_v(j3_xy, 8, COLOR_GRIP if grip_active else rl.WHITE)
        
        rl.draw_text(f"Iterasi: {iteration}", 10, 10, 20, rl.WHITE)
        rl.draw_text(f"Speed: {speed_multiplier}x", 10, 35, 20, rl.YELLOW)
        rl.draw_text(save_msg, 10, 60, 20, save_col)
        
        if is_training_moment:
            rl.draw_text("LEARNING...", j3_xy[0]+10, j3_xy[1]-10, 20, rl.GREEN)
        
        rl.end_drawing()

    rl.close_window()

if __name__ == "__main__":
    main()