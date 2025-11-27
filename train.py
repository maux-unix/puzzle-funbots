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
MODEL_PATH = "blind_robot_memory.pth" 

# --- NO GRAVITY (Blind Task) ---
GRAVITY_VAL = 0.0 

# --- KECEPATAN ---
SPEED_LEVELS = [1, 2, 5, 10, 20, 50, 100]

# Fisika Robot
LINK_1 = 1.2 
LINK_2 = 1.0 

# ILC Parameter
LEARNING_RATE = 0.005
HIDDEN_SIZE = 128 

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

# --- 1. BLIND NEURAL NETWORK ---
class BlindNetwork(nn.Module):
    def __init__(self):
        super(BlindNetwork, self).__init__()
        self.fc1 = nn.Linear(2, HIDDEN_SIZE)
        self.fc2 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.fc3 = nn.Linear(HIDDEN_SIZE, 2)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

# --- 2. KINEMATIKA ---
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

def apply_physics_no_gravity(theta1, theta2):
    return np.array([theta1, theta2], dtype=np.float32)

# --- 3. DASHBOARD ---
plt.ion()
fig = plt.figure(figsize=(10, 8))
gs = GridSpec(2, 2, figure=fig)
ax_loss = fig.add_subplot(gs[0, 0])
ax_score = fig.add_subplot(gs[0, 1])
ax_cm = fig.add_subplot(gs[1, :])
history_loss, history_score = [], []
best_display_matrix = np.zeros((3, 3), dtype=int) 

def update_dashboard(curr_iter, best_score_val):
    ax_loss.clear(); ax_score.clear(); ax_cm.clear()
    
    if history_loss:
        ax_loss.plot(history_loss, color='red', label='Error')
        ax_loss.set_title("Training Loss")
        ax_loss.grid(True, alpha=0.3)

    if history_score:
        ax_score.plot(history_score, color='blue', marker='o')
        ax_score.axhline(y=best_score_val, color='gold', linestyle='--', label='Best Score')
        ax_score.set_title("Total Score")
        ax_score.legend()
        ax_score.grid(True, alpha=0.3)

    im = ax_cm.imshow(best_display_matrix, cmap='Purples', vmin=0)
    total = np.sum(best_display_matrix)
    acc = np.trace(best_display_matrix) / total * 100.0 if total > 0 else 0
    ax_cm.set_title(f"Best Matrix Accuracy: {acc:.1f}%")
    ax_cm.set_xticks(np.arange(3)); ax_cm.set_yticks(np.arange(3))
    ax_cm.set_xticklabels(COLOR_NAMES); ax_cm.set_yticklabels(COLOR_NAMES)
    ax_cm.set_xlabel("Predicted Slot"); ax_cm.set_ylabel("Actual Cube Color")
    
    for i in range(3):
        for j in range(3):
            count = best_display_matrix[i, j]
            col = "white" if count > 2 else "black"
            ax_cm.text(j, i, str(count), ha="center", va="center", color=col)
            
    plt.tight_layout(); plt.pause(0.001)

# --- 4. MAIN LOOP ---
def main():
    rl.init_window(SCREEN_WIDTH, SCREEN_HEIGHT, "Blind Robot Memorization")
    rl.set_target_fps(60)
    
    net = BlindNetwork()
    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()
    
    best_loss = float('inf')
    best_score = -1.0
    
    # Load Model
    global best_display_matrix
    if os.path.exists(MODEL_PATH):
        try:
            ckpt = torch.load(MODEL_PATH)
            net.load_state_dict(ckpt['model'])
            best_score = ckpt['score']
            best_display_matrix = ckpt['matrix']
            print(f"Loaded Brain. Best Score: {best_score:.1f}")
        except: pass
    
    curr_theta = np.array([math.pi / 2, 0.0])
    
    # Buffer Data
    train_X, train_Y = [], []
    # FIXED: Variable name updated here
    current_matrix = np.zeros((3, 3), dtype=int)
    
    iteration = 1
    total_score_iter = 0
    speed_idx = 0
    save_msg = "Ready"
    save_col = rl.GRAY
    
    # --- LEVEL SETUP (FIXED TARGETS) ---
    def reset_level():
        c_list = []
        cols = [RED_RGB]*3 + [BLUE_RGB]*3 + [YELLOW_RGB]*3
        random.shuffle(cols)
        
        for i in range(9):
            r, c = i // 3, i % 3
            cx = -1.8 + (c * 0.25); cy = 0.5 + (r * 0.25)
            
            target_id = -1
            if cols[i] == RED_RGB: target_id = random.randint(0, 2)
            elif cols[i] == BLUE_RGB: target_id = random.randint(3, 5)
            else: target_id = random.randint(6, 8)
            
            c_list.append({
                "id": i, "pos": [cx, cy], "color": cols[i], 
                "state": "IDLE", "target_slot_id": target_id
            })
            
        s_list = []
        fixed_colors = [RED_RGB, RED_RGB, RED_RGB, 
                        BLUE_RGB, BLUE_RGB, BLUE_RGB, 
                        YELLOW_RGB, YELLOW_RGB, YELLOW_RGB]
        
        for i in range(9):
            r, c = i // 3, i % 3
            sx = 1.0 + (c * 0.25); sy = 1.0 - (r * 0.25)
            s_list.append({"id": i, "pos": [sx, sy], "color": fixed_colors[i], "occupied": False})
        return c_list, s_list

    cubes, slots = reset_level()
    task_queue = list(range(9))
    random.shuffle(task_queue)
    curr_task = 0
    
    path, path_idx = [], 0
    grip_active = False
    active_idx = -1
    training_mode = False
    train_timer = 0
    last_ideal_pos = [0,0]
    current_goal = [0,0]
    is_pickup_phase = False
    
    def get_ideal_next_step(curr_pos_xy, target_pos_xy, progress_ratio):
        lx = curr_pos_xy[0] * (1-progress_ratio) + target_pos_xy[0] * progress_ratio
        ly = curr_pos_xy[1] * (1-progress_ratio) + target_pos_xy[1] * progress_ratio
        arc = 0.6 * math.sin(progress_ratio * math.pi)
        return [lx, ly + arc]

    update_dashboard(iteration, best_score)

    while not rl.window_should_close():
        
        if rl.is_key_pressed(rl.KEY_RIGHT): speed_idx = min(6, speed_idx+1)
        if rl.is_key_pressed(rl.KEY_LEFT): speed_idx = max(0, speed_idx-1)
        speed = SPEED_LEVELS[speed_idx]
        
        for _ in range(speed):
            
            if not training_mode:
                # 1. Pick Next Task
                if active_idx == -1:
                    if curr_task < 9:
                        active_idx = task_queue[curr_task]
                        start_p = forward_kinematics(curr_theta[0], curr_theta[1])[1]
                        end_p = cubes[active_idx]["pos"]
                        current_goal = end_p 
                        path_idx = 0
                        curr_task += 1
                        is_pickup_phase = True
                        last_ideal_pos = start_p # Reset tracking pos
                    else:
                        training_mode = True
                        train_timer = 0

                # 2. Robot Bergerak
                if active_idx != -1:
                    
                    # INPUT BUTA
                    nn_in = torch.tensor(curr_theta, dtype=torch.float32)
                    command_theta = net(nn_in).detach().numpy()
                    
                    # TRAINING LABEL GENERATION (CHEAT SHEET)
                    progress = (path_idx + 1) / 40 
                    if progress > 1.0: progress = 1.0
                    
                    ideal_xy = get_ideal_next_step(
                        forward_kinematics(curr_theta[0], curr_theta[1])[1] if path_idx==0 else last_ideal_pos, 
                        current_goal, 
                        1.0/ (40 - path_idx + 0.001) 
                    )
                    last_ideal_pos = ideal_xy 
                    ideal_theta_label = inverse_kinematics(ideal_xy[0], ideal_xy[1])
                    
                    train_X.append(curr_theta)
                    train_Y.append(ideal_theta_label)
                    
                    # Fisika
                    curr_theta = apply_physics_no_gravity(command_theta[0], command_theta[1])
                    
                    if cubes[active_idx]["state"] == "GRIPPED":
                        hand_pos = forward_kinematics(curr_theta[0], curr_theta[1])[1]
                        cubes[active_idx]["pos"] = list(hand_pos)

                    path_idx += 1
                    
                    if path_idx >= 40:
                        cube = cubes[active_idx]
                        
                        if is_pickup_phase: 
                            cube["state"] = "GRIPPED"
                            grip_active = True
                            tgt_slot = slots[cube["target_slot_id"]]
                            current_goal = tgt_slot["pos"]
                            path_idx = 0
                            is_pickup_phase = False
                            # Reset tracking for next leg
                            last_ideal_pos = forward_kinematics(curr_theta[0], curr_theta[1])[1]
                            
                        else: 
                            cube["state"] = "DONE"
                            grip_active = False
                            
                            # --- SCORE & MATRIX ---
                            fx, fy = cube["pos"]
                            tx, ty = slots[cube["target_slot_id"]]["pos"]
                            dist = math.sqrt((fx-tx)**2 + (fy-ty)**2)
                            
                            scr = max(0, (0.2 - dist) * (100/0.2))
                            total_score_iter += scr
                            
                            act = COLOR_MAP[cube["color"]]
                            pred = -1
                            for s in slots:
                                dx = cube["pos"][0] - s["pos"][0]
                                dy = cube["pos"][1] - s["pos"][1]
                                if math.sqrt(dx*dx + dy*dy) < 0.12:
                                    pred = COLOR_MAP[s["color"]]
                                    break
                            
                            # FIX: Gunakan current_matrix disini
                            if pred != -1: current_matrix[act][pred] += 1
                            
                            active_idx = -1
            
            # --- TRAINING PHASE ---
            else:
                train_timer += 1
                delay = 60 if speed == 1 else 1
                if train_timer > delay:
                    if len(train_X) > 0:
                        inp = torch.tensor(np.array(train_X), dtype=torch.float32)
                        lbl = torch.tensor(np.array(train_Y), dtype=torch.float32)
                        
                        net.train()
                        loss_val = 0
                        for _ in range(50):
                            optimizer.zero_grad()
                            out = net(inp)
                            loss = criterion(out, lbl)
                            loss.backward()
                            optimizer.step()
                            loss_val = loss.item()
                        
                        history_loss.append(loss_val)
                        history_score.append(total_score_iter)
                        
                        if total_score_iter > best_score:
                            best_score = total_score_iter
                            # FIX: Copy from current_matrix
                            best_display_matrix = np.copy(current_matrix)
                            torch.save({
                                'model': net.state_dict(),
                                'score': best_score,
                                'matrix': best_display_matrix
                            }, MODEL_PATH)
                            save_msg = f"NEW RECORD! Score: {best_score:.1f}"
                            save_col = rl.GREEN
                        else:
                            save_msg = f"Learning... (S:{total_score_iter:.1f})"
                            save_col = rl.ORANGE
                    
                    train_X, train_Y = [], []
                    # FIX: Reset current_matrix
                    current_matrix = np.zeros((3, 3), dtype=int)
                    total_score_iter = 0
                    cubes, slots = reset_level()
                    task_queue = list(range(9))
                    random.shuffle(task_queue)
                    curr_task = 0
                    curr_theta = np.array([math.pi/2, 0.0]) 
                    
                    iteration += 1
                    training_mode = False
                    update_dashboard(iteration, best_score)

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
            
        j1 = BASE_POS
        j2_m, j3_m = forward_kinematics(curr_theta[0], curr_theta[1])
        j2 = world_to_screen(j2_m[0], j2_m[1])
        j3 = world_to_screen(j3_m[0], j3_m[1])
        
        rl.draw_line_ex(j1, j2, 10, COLOR_ROBOT)
        rl.draw_circle_v(j1, 8, COLOR_JOINT)
        rl.draw_circle_v(j2, 7, COLOR_JOINT)
        rl.draw_line_ex(j2, j3, 6, COLOR_ROBOT)
        rl.draw_circle_v(j3, 8, COLOR_GRIP if grip_active else rl.WHITE)
        
        rl.draw_text(f"Blind ILC - Iter: {iteration}", 10, 10, 20, rl.WHITE)
        rl.draw_text(f"Score: {total_score_iter:.1f}", 10, 35, 20, rl.YELLOW)
        rl.draw_text(save_msg, 10, 60, 20, save_col)
        if training_mode: rl.draw_text("LEARNING...", 350, 300, 30, rl.GREEN)
        
        rl.end_drawing()

    rl.close_window()

if __name__ == "__main__":
    main()