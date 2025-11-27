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
MODEL_PATH = "ilc_online_learner.pth" 

# --- GRAVITASI ---
GRAVITY_VAL = 0.20 

# --- KECEPATAN ---
SPEED_LEVELS = [1, 2, 5, 10, 20, 50, 100]

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
fig = plt.figure(figsize=(12, 9))
gs = GridSpec(2, 2, figure=fig)

ax_loss = fig.add_subplot(gs[0, 0])
ax_score = fig.add_subplot(gs[0, 1])
ax_cm_curr = fig.add_subplot(gs[1, 0]) # Bawah Kiri (Current)
ax_cm_total = fig.add_subplot(gs[1, 1]) # Bawah Kanan (Total)

history_loss = []
history_score = []

def draw_confusion_matrix(ax, matrix, title, cmap='Blues'):
    ax.clear()
    im = ax.imshow(matrix, cmap=cmap, vmin=0)
    total = np.sum(matrix)
    acc = 0
    if total > 0: acc = np.trace(matrix) / total * 100.0
    
    ax.set_title(f"{title}\n(Acc: {acc:.1f}%, N={total})")
    ax.set_xticks(np.arange(3)); ax.set_yticks(np.arange(3))
    ax.set_xticklabels(COLOR_NAMES); ax.set_yticklabels(COLOR_NAMES)
    ax.set_xlabel("Prediksi"); ax.set_ylabel("Aktual")
    
    for i in range(3):
        for j in range(3):
            count = matrix[i, j]
            col = "white" if count > (np.max(matrix)/2) else "black"
            ax.text(j, i, str(count), ha="center", va="center", color=col, fontweight='bold')

def update_dashboard(curr_iter, best_score_val, current_matrix, total_matrix):
    # 1. Loss Plot
    ax_loss.clear()
    if history_loss:
        ax_loss.plot(history_loss, color='red')
        ax_loss.set_title("Training Loss")
        ax_loss.grid(True, alpha=0.3)

    # 2. Score Plot
    ax_score.clear()
    if history_score:
        ax_score.plot(history_score, color='blue', marker='o')
        ax_score.axhline(y=best_score_val, color='gold', linestyle='--')
        ax_score.set_title("Score Per Cube")
        ax_score.set_ylim(0, 105)
        ax_score.grid(True, alpha=0.3)

    # 3. Current Matrix
    draw_confusion_matrix(ax_cm_curr, current_matrix, f"Current Iteration ({curr_iter})", cmap='Blues')

    # 4. Total Matrix
    draw_confusion_matrix(ax_cm_total, total_matrix, "Total Accumulation", cmap='Greens')
            
    plt.tight_layout()
    plt.pause(0.001)
    
    # Save plot gambar
    if not os.path.exists("training_logs"): os.makedirs("training_logs")
    plt.savefig(f"training_logs/plot_iter_{curr_iter}.png")

# --- 4. LOGIKA GAME ---
def main():
    rl.init_window(SCREEN_WIDTH, SCREEN_HEIGHT, "2D ILC (Dual Matrix)")
    rl.set_target_fps(60)
    
    net = ILCNetwork()
    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()
    
    best_loss = float('inf')
    best_score = -1.0
    
    if os.path.exists(MODEL_PATH):
        try:
            checkpoint = torch.load(MODEL_PATH)
            net.load_state_dict(checkpoint['model_state_dict'])
            best_loss = checkpoint.get('best_loss', float('inf'))
            best_score = checkpoint.get('best_score', -1.0)
            net.eval()
            print(f"Model Loaded. Best Score: {best_score:.1f}")
        except: pass
    
    curr_theta = np.array([math.pi / 2, 0.0])
    
    cube_data_X = []
    cube_data_Y = []
    
    # DUA MATRIX BERBEDA
    current_iter_matrix = np.zeros((3, 3), dtype=int)
    total_confusion_matrix = np.zeros((3, 3), dtype=int)
    
    iteration = 1
    total_score_iter = 0
    speed_level_index = 0
    speed_multiplier = SPEED_LEVELS[speed_level_index]
    
    save_msg = "Ready"
    save_col = rl.GRAY
    
    current_cube_score = 0.0
    
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

    # Initial Plot
    update_dashboard(iteration, best_score, current_iter_matrix, total_confusion_matrix)

    while not rl.window_should_close():
        
        if rl.is_key_pressed(rl.KEY_RIGHT):
            speed_level_index = min(len(SPEED_LEVELS) - 1, speed_level_index + 1)
        if rl.is_key_pressed(rl.KEY_LEFT):
            speed_level_index = max(0, speed_level_index - 1)
        speed_multiplier = SPEED_LEVELS[speed_level_index]
        
        for _ in range(speed_multiplier):
            
            # --- TRAINING PHASE ---
            if is_training_moment:
                train_moment_timer += 1
                if train_moment_timer > 5:
                    if len(cube_data_X) > 0:
                        inp = torch.tensor(np.array(cube_data_X), dtype=torch.float32)
                        tgt = torch.tensor(np.array(cube_data_Y), dtype=torch.float32)
                        
                        net.train()
                        current_cube_loss = 0
                        for _ in range(20):
                            optimizer.zero_grad()
                            out = net(inp)
                            loss = criterion(out, tgt)
                            loss.backward()
                            optimizer.step()
                            current_cube_loss = loss.item()
                        
                        history_loss.append(current_cube_loss)
                        history_score.append(current_cube_score)
                        
                        # LOGIKA SAVE
                        is_new_best = False
                        reason = ""
                        if current_cube_score > best_score:
                            is_new_best = True
                            reason = f"New Best Score! ({current_cube_score:.1f})"
                        elif current_cube_score == best_score and current_cube_loss < best_loss:
                            is_new_best = True
                            reason = f"Lower Loss ({current_cube_loss:.5f})"
                        
                        if is_new_best:
                            best_score = current_cube_score
                            best_loss = current_cube_loss
                            save_data = {
                                'model_state_dict': net.state_dict(),
                                'best_loss': best_loss,
                                'best_score': best_score
                            }
                            torch.save(save_data, MODEL_PATH)
                            save_msg = f"SAVED! {reason}"
                            save_col = rl.GREEN
                        else:
                            save_msg = f"Learning... (Loss: {current_cube_loss:.4f})"
                            save_col = rl.ORANGE
                    
                    cube_data_X = []
                    cube_data_Y = []
                    is_training_moment = False
                    
                    # Update Grafik
                    update_dashboard(iteration, best_score, current_iter_matrix, total_confusion_matrix)
                
                continue 

            # --- PHYSICS PHASE ---
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
                    current_iter_matrix = np.zeros((3, 3), dtype=int) # Reset Current Matrix
                    # Total Matrix TIDAK DI-RESET (Terakumulasi terus)
                    iteration += 1

            if target_cube_idx != -1 and path_idx < len(path):
                target_xy = path[path_idx]
                ideal_theta = inverse_kinematics(target_xy[0], target_xy[1])
                
                nn_input = torch.tensor(np.concatenate([curr_theta, ideal_theta]), dtype=torch.float32)
                with torch.no_grad(): correction = net(nn_input).numpy()
                
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
                    
                    final_x, final_y = cube["pos"]
                    target_x, target_y = slots[target_slot_idx]["pos"]
                    dist_error = math.sqrt((final_x - target_x)**2 + (final_y - target_y)**2)
                    current_cube_score = max(0, (0.15 - dist_error) * (100 / 0.15))
                    total_score_iter += current_cube_score
                    
                    # --- UPDATE MATRIX ---
                    actual_idx = COLOR_MAP[cube["color"]]
                    pred_idx = -1
                    for s_check in slots:
                        sx, sy = s_check["pos"]
                        d = math.sqrt((final_x - sx)**2 + (final_y - sy)**2)
                        if d < 0.12: 
                            pred_idx = COLOR_MAP[s_check["color"]]
                            s_check["filled"] = True 
                            break
                    if pred_idx != -1:
                        # Update BOTH matrices
                        current_iter_matrix[actual_idx][pred_idx] += 1
                        total_confusion_matrix[actual_idx][pred_idx] += 1
                    
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