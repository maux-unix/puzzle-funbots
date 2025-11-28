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

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
#device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
device = "cpu"
print("Using device:", device)

# --- KONFIGURASI DASAR ---
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
BASE_POS = (400, 300)            # Robot di tengah layar
PIXELS_PER_METER = 180.0        # Skala dunia -> layar

# Penyimpanan model
MODEL_BEST_PATH   = "ilc_3x3_best.pth"     # model dengan skor terbaik (global)
MODEL_LATEST_PATH = "ilc_3x3_latest.pth"   # model terakhir

# --- FISIKA DEADZONE (KOTAK PEMBATAS) ---
SAFE_MIN_X = -1.8  # Dinding Kiri
SAFE_MAX_X = 1.8   # Dinding Kanan
SAFE_MIN_Y = -0.8  # Lantai
SAFE_MAX_Y = 1.5   # Atap

# --- KECEPATAN SIMULASI ---
SPEED_LEVELS = [1, 2, 5, 10, 20, 50, 100, 1000, 5000, 10000]

# Fisika Robot
LINK_1 = 1.2
LINK_2 = 1.0

# TRAINING PARAM
LEARNING_RATE = 0.002
HIDDEN_SIZE = 128

# Warna
COLOR_BG = (30, 30, 30, 255)
COLOR_ROBOT = (220, 220, 220, 255)
COLOR_JOINT = (50, 50, 50, 255)
COLOR_WALL = (80, 80, 80, 255)
COLOR_SLOT_BG = (60, 60, 60, 255)
COLOR_GRIP = (50, 255, 50, 255)

COLORS_NUM = [
    (255, 100, 100, 255), (100, 255, 100, 255), (100, 100, 255, 255),
    (255, 255, 100, 255), (100, 255, 255, 255), (255, 100, 255, 255),
    (255, 150, 50, 255),  (50, 255, 150, 255),  (150, 50, 255, 255)
]

# skor teori: 9 cube * max 100
MAX_SCORE_THEORETICAL = 9 * 100.0
THRESHOLD_PUZZLE = 0.80 * MAX_SCORE_THEORETICAL  # 75% dari skor maksimal

# --- 1. NEURAL NETWORK ---
class PuzzleNet(nn.Module):
    def __init__(self):
        super(PuzzleNet, self).__init__()
        self.fc1 = nn.Linear(4, HIDDEN_SIZE)
        self.ln1 = nn.LayerNorm(HIDDEN_SIZE)
        self.fc2 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.ln2 = nn.LayerNorm(HIDDEN_SIZE)
        self.fc3 = nn.Linear(HIDDEN_SIZE, 2)
        self.relu = nn.LeakyReLU(0.1)

        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        torch.nn.init.xavier_uniform_(self.fc3.weight)

    def forward(self, x):
        # x shape: (..., 4)
        x = self.relu(self.ln1(self.fc1(x)))
        x = self.relu(self.ln2(self.fc2(x)))
        return self.fc3(x)  # output: (..., 2) -> sudut [theta1, theta2]


# --- 2. KINEMATIKA & KONSTRAIN ---

def safe_theta(theta: np.ndarray) -> np.ndarray:
    if np.isnan(theta).any() or np.isinf(theta).any():
        return np.array([math.pi / 2, 0.0], dtype=np.float32)
    return theta.astype(np.float32)

def forward_kinematics(theta1, theta2):
    theta1 = float(theta1)
    theta2 = float(theta2)
    x1 = LINK_1 * math.cos(theta1)
    y1 = LINK_1 * math.sin(theta1)
    x2 = x1 + LINK_2 * math.cos(theta1 + theta2)
    y2 = y1 + LINK_2 * math.sin(theta1 + theta2)
    return (x1, y1), (x2, y2)

def inverse_kinematics(target_x, target_y):
    dist = math.sqrt(target_x**2 + target_y**2)
    max_reach = LINK_1 + LINK_2 - 0.01
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
        return np.array([theta1, theta2], dtype=np.float32)
    except Exception:
        return None

def world_to_screen(x, y):
    px = BASE_POS[0] + (x * PIXELS_PER_METER)
    py = BASE_POS[1] - (y * PIXELS_PER_METER)
    return int(px), int(py)

def apply_physical_constraints(theta1, theta2):
    _, hand_pos = forward_kinematics(theta1, theta2)
    curr_x, curr_y = hand_pos

    clamped_x = curr_x
    clamped_y = curr_y
    hit_wall = False

    if curr_x > SAFE_MAX_X:
        clamped_x = SAFE_MAX_X
        hit_wall = True
    if curr_x < SAFE_MIN_X:
        clamped_x = SAFE_MIN_X
        hit_wall = True
    if curr_y < SAFE_MIN_Y:
        clamped_y = SAFE_MIN_Y
        hit_wall = True
    if curr_y > SAFE_MAX_Y:
        clamped_y = SAFE_MAX_Y
        hit_wall = True

    if hit_wall:
        corrected = inverse_kinematics(clamped_x, clamped_y)
        if corrected is not None:
            return safe_theta(corrected)

    return np.array([theta1, theta2], dtype=np.float32)


# --- 3. PLOT ERROR DI RAYLIB ---

# --- CONFUSION MATRIX & HISTORY ---
NUM_CLASSES = 9  # angka 1..9

# Matrix total (akumulasi semua episode)
total_conf_matrix = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)

hist_error = []  # average positional error per iteration
hist_accuracy = []   # akurasi (dari total_conf_matrix) per iterasi training

def draw_error_plot(hist_error):
    """Plot sederhana error gerakan per iterasi di pojok kiri bawah."""
    if not hist_error:
        return

    PLOT_W = 320
    PLOT_H = 140
    PLOT_X = 10
    PLOT_Y = SCREEN_HEIGHT - PLOT_H - 10

    # Background kotak plot
    rl.draw_rectangle(PLOT_X, PLOT_Y, PLOT_W, PLOT_H, (20, 20, 20, 255))
    rl.draw_rectangle_lines(PLOT_X, PLOT_Y, PLOT_W, PLOT_H, rl.GRAY)
    rl.draw_text("Avg Pos Error / Iter", PLOT_X + 5, PLOT_Y - 18, 16, rl.WHITE)

    n = len(hist_error)
    if n < 2:
        x = PLOT_X + 10
        y = PLOT_Y + PLOT_H - 10
        rl.draw_circle(x, y, 3, rl.GREEN)
        return

    max_err = max(max(hist_error), 1e-4)
    margin_top = 15
    margin_bottom = 15

    # Garis nol error (teoritis)
    y_zero = PLOT_Y + PLOT_H - margin_bottom
    rl.draw_line(PLOT_X, y_zero, PLOT_X + PLOT_W, y_zero, rl.DARKGRAY)

    for i in range(n - 1):
        t1 = i / max(1, n - 1)
        t2 = (i + 1) / max(1, n - 1)

        x1 = PLOT_X + int(t1 * (PLOT_W - 20)) + 10
        x2 = PLOT_X + int(t2 * (PLOT_W - 20)) + 10

        norm1 = min(hist_error[i] / max_err, 1.0)
        norm2 = min(hist_error[i + 1] / max_err, 1.0)

        y1 = PLOT_Y + PLOT_H - margin_bottom - int(norm1 * (PLOT_H - margin_top - margin_bottom))
        y2 = PLOT_Y + PLOT_H - margin_bottom - int(norm2 * (PLOT_H - margin_top - margin_bottom))

        rl.draw_line(x1, y1, x2, y2, rl.RED)

    # Titik terakhir ditandai
    last_x = PLOT_X + int(1.0 * (PLOT_W - 20)) + 10
    last_norm = min(hist_error[-1] / max_err, 1.0)
    last_y = PLOT_Y + PLOT_H - margin_bottom - int(last_norm * (PLOT_H - margin_top - margin_bottom))
    rl.draw_circle(last_x, last_y, 3, rl.YELLOW)

    # Info kecil di pojok
    txt = f"Last Err: {hist_error[-1]:.4f}"
    rl.draw_text(txt, PLOT_X + 5, PLOT_Y + 5, 14, rl.LIGHTGRAY)

def update_confusion_for_cube(cubes, slots, cube_index):
    """
    Update confusion matrix TOTAL saat satu cube selesai diletakkan.
    - True label  : angka cube (1..9)
    - Pred label  : angka slot yang paling dekat dengan posisi akhir cube.
    """
    global total_conf_matrix

    cube = cubes[cube_index]
    true_val = cube["val"]    # 1..9
    true_idx = true_val - 1   # 0..8

    fx, fy = cube["pos"]
    pred_val = None
    min_d = 1e9

    # Cari slot terdekat dari posisi akhir kubus
    for s in slots:
        sx, sy = s["pos"]
        d = math.dist((fx, fy), (sx, sy))
        if d < min_d:
            min_d = d
            pred_val = s["val"]

    if pred_val is None:
        return

    pred_idx = pred_val - 1
    total_conf_matrix[true_idx, pred_idx] += 1

def compute_accuracy_from_cm(cm: np.ndarray) -> float:
    total = cm.sum()
    if total == 0:
        return 0.0
    return float(np.trace(cm)) / float(total) * 100.0


def save_training_summary(iteration: int):
    """
    Simpan 1 gambar PNG berisi:
      - plot akurasi (total_conf_matrix) vs iterasi,
      - plot avg error vs iterasi,
      - confusion matrix TOTAL.
    """
    os.makedirs("training_logs", exist_ok=True)

    fig = plt.figure(figsize=(10, 8))
    gs = GridSpec(2, 2, figure=fig)

    ax_acc = fig.add_subplot(gs[0, 0])
    ax_err = fig.add_subplot(gs[0, 1])
    ax_cm  = fig.add_subplot(gs[1, :])

    # --- Plot Akurasi ---
    if hist_accuracy:
        iters = np.arange(1, len(hist_accuracy) + 1)
        ax_acc.plot(iters, hist_accuracy, marker="o")
        ax_acc.set_title("Accuracy dari Confusion Matrix (Total)")
        ax_acc.set_xlabel("Iteration")
        ax_acc.set_ylabel("Accuracy (%)")
        ax_acc.grid(True, alpha=0.3)
    else:
        ax_acc.set_title("Accuracy (Belum ada data)")
        ax_acc.axis("off")

    # --- Plot Error ---
    if hist_error:
        iters_e = np.arange(1, len(hist_error) + 1)
        ax_err.plot(iters_e, hist_error, marker="o", color="red")
        ax_err.set_title("Average Positional Error per Iteration")
        ax_err.set_xlabel("Iteration")
        ax_err.set_ylabel("Error (meter)")
        ax_err.grid(True, alpha=0.3)
    else:
        ax_err.set_title("Error (Belum ada data)")
        ax_err.axis("off")

    # --- Confusion Matrix TOTAL ---
    cm = total_conf_matrix
    im = ax_cm.imshow(cm, cmap="Blues", vmin=0)

    acc_total = compute_accuracy_from_cm(cm)
    ax_cm.set_title(f"Total Confusion Matrix (Acc: {acc_total:.1f}%)")
    ax_cm.set_xlabel("Predicted")
    ax_cm.set_ylabel("Actual")

    tick_labels = [str(i + 1) for i in range(NUM_CLASSES)]
    ax_cm.set_xticks(np.arange(NUM_CLASSES))
    ax_cm.set_yticks(np.arange(NUM_CLASSES))
    ax_cm.set_xticklabels(tick_labels)
    ax_cm.set_yticklabels(tick_labels)

    max_val = cm.max() if cm.size > 0 else 0
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            v = int(cm[i, j])
            if v > 0:
                color = "white" if max_val > 0 and v > max_val / 2 else "black"
                ax_cm.text(j, i, str(v), ha="center", va="center",
                           color=color, fontsize=8)

    fig.colorbar(im, ax=ax_cm, fraction=0.046, pad=0.04)

    plt.tight_layout()
    out_path = os.path.join("training_logs", f"summary_iter_{iteration}.png")
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[TRAIN] Saved summary plot: {out_path}")

# --- 4. GENERATOR SUDOKU 3x3 (PERMUTASI 1..9) ---

def generate_valid_sudoku_3x3():
    nums = list(range(1, 10))
    random.shuffle(nums)
    return nums


# --- 5. MAIN LOOP ---
def main():
    global hist_error

    rl.init_window(SCREEN_WIDTH, SCREEN_HEIGHT, "Neural Sudoku 3x3 (NN + Error Plot + 0.9 Threshold)")
    rl.set_target_fps(60)

    net = PuzzleNet().to(device)
    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    # global best score (semua layout)
    global_best_score = -1.0

    # Load model terbaik kalau ada
    if os.path.exists(MODEL_BEST_PATH):
        try:
            ckpt = torch.load(MODEL_BEST_PATH)
            net.load_state_dict(ckpt['model'])
            global_best_score = ckpt.get('score', -1.0)
            print(f"Brain Loaded (BEST). Global Best Score: {global_best_score:.1f}")
        except Exception as e:
            print("Failed to load best model:", e)

    curr_theta = np.array([-math.pi / 2, 0.0], dtype=np.float32)

    train_X, train_Y = [], []

    iteration = 1
    total_score_iter = 0.0
    speed_idx = 0
    save_msg = "Ready"
    save_col = rl.GRAY
    last_cube_val = 0
    last_cube_score = 0.0

    # Error akumulatif per iterasi
    sum_err_iter = 0.0
    step_count_iter = 0

    # --- LAYOUT SUDOKU GLOBAL YANG AKAN DIGANTI KALAU SUDAH >= 0.9 * MAX ---
    left_vals_global  = generate_valid_sudoku_3x3()
    right_vals_global = generate_valid_sudoku_3x3()
    puzzle_best_score = -1.0  # best score khusus layout ini

    def regenerate_puzzle():
        nonlocal left_vals_global, right_vals_global, puzzle_best_score
        left_vals_global  = generate_valid_sudoku_3x3()
        right_vals_global = generate_valid_sudoku_3x3()
        puzzle_best_score = -1.0
        print("New Sudoku Layout Generated (layout baru, lanjut belajar).")

    def reset_level():
        c_list = []
        s_list = []

        spacing_src = 0.18
        spacing_tgt = 0.22

        src_cx, src_cy = -0.9, 0.0
        tgt_cx, tgt_cy =  0.95, 0.0   # target agak ke kanan

        left_vals = left_vals_global
        right_vals = right_vals_global

        # mapping nilai -> index slot kanan
        val_to_slot_idx = {v: i for i, v in enumerate(right_vals)}

        # CUBES (KIRI)
        for i in range(9):
            val = left_vals[i]
            r = i // 3
            c = i % 3

            off_x = (c - 1) * spacing_src
            off_y = (1 - r) * spacing_src

            cx = src_cx + off_x
            cy = src_cy + off_y

            target_slot_id = val_to_slot_idx[val]

            c_list.append({
                "val": val,
                "pos": [cx, cy],
                "color": COLORS_NUM[val - 1],
                "state": "IDLE",
                "target_slot_id": target_slot_id
            })

        # SLOTS (KANAN)
        for i in range(9):
            r = i // 3
            c = i % 3

            off_x = (c - 1) * spacing_tgt
            off_y = (1 - r) * spacing_tgt

            sx = tgt_cx + off_x
            sy = tgt_cy + off_y

            val = right_vals[i]  # angka yang diminta slot ini

            s_list.append({
                "id": i,
                "pos": [sx, sy],
                "val": val,
                "occupied": False
            })

        return c_list, s_list

    cubes, slots = reset_level()
    cubes.sort(key=lambda x: x["val"])  # robot ambil 1,2,...,9 urut

    path_idx = 0
    grip_active = False
    active_idx = -1
    training_mode = False
    train_timer = 0

    STEPS_TRAVEL = 30
    STEPS_APPROACH = 60
    STEPS_WAIT = 60

    current_task_num = 0
    joint_traj = []
    grip_sched = []

    # --- GENERATOR LINTASAN (TEACHER TRAJECTORY, HANYA LABEL) ---
    def generate_dynamic_trajectory(current_hand_pos, pickup_pos, drop_pos):
        traj = []
        grip = []

        SAFE_HEIGHT = 0.5
        PICK_Y = pickup_pos[1] + 0.05
        DROP_Y = drop_pos[1] + 0.05

        points_config = [
            ([current_hand_pos[0], SAFE_HEIGHT], 0, STEPS_APPROACH),
            ([pickup_pos[0], SAFE_HEIGHT],       0, STEPS_TRAVEL),
            ([pickup_pos[0], PICK_Y],            0, STEPS_APPROACH),
            ([pickup_pos[0], PICK_Y],            1, STEPS_WAIT),
            ([pickup_pos[0], SAFE_HEIGHT],       1, STEPS_APPROACH),
            ([drop_pos[0], SAFE_HEIGHT],         1, STEPS_TRAVEL),
            ([drop_pos[0], DROP_Y],              1, STEPS_APPROACH),
            ([drop_pos[0], DROP_Y],              0, STEPS_WAIT),
            ([drop_pos[0], SAFE_HEIGHT],         0, STEPS_APPROACH),
        ]

        current_pos = current_hand_pos
        current_grip = 0

        for target_pos, target_grip, steps in points_config:
            for s in range(steps):
                alpha = s / max(1, steps)
                pos_x = current_pos[0] * (1 - alpha) + target_pos[0] * alpha
                pos_y = current_pos[1] * (1 - alpha) + target_pos[1] * alpha

                ik = inverse_kinematics(pos_x, pos_y)
                if ik is not None:
                    traj.append(ik)
                else:
                    traj.append(curr_theta.copy())

                if steps == STEPS_WAIT:
                    grip.append(target_grip)
                else:
                    grip.append(current_grip if alpha < 0.5 else target_grip)

            current_pos = target_pos
            current_grip = target_grip

        return np.array(traj), grip

    # --- MAIN LOOP ---
    while not rl.window_should_close():
        # speed control
        if rl.is_key_pressed(rl.KEY_RIGHT):
            speed_idx = min(len(SPEED_LEVELS) - 1, speed_idx + 1)
        if rl.is_key_pressed(rl.KEY_LEFT):
            speed_idx = max(0, speed_idx - 1)
        speed = SPEED_LEVELS[speed_idx]

        for _ in range(speed):

            # ====== FASE EKSEKUSI (ROBOT GERAK, PURE NN) ======
            if not training_mode:
                # 1. Pilih tugas
                if active_idx == -1:
                    if current_task_num < len(cubes):
                        active_idx = current_task_num
                        curr_theta = safe_theta(curr_theta)

                        hand_pos = forward_kinematics(curr_theta[0], curr_theta[1])[1]
                        start_p = hand_pos
                        cube_p = cubes[active_idx]["pos"]
                        slot_p = slots[cubes[active_idx]["target_slot_id"]]["pos"]

                        joint_traj, grip_sched = generate_dynamic_trajectory(start_p, cube_p, slot_p)
                        path_idx = 0
                        current_task_num += 1
                    else:
                        training_mode = True
                        train_timer = 0

                # 2. Gerak sepanjang trajektori (NN control)
                if active_idx != -1 and path_idx < len(joint_traj):
                    ideal_theta_label = joint_traj[path_idx]

                    # Fitur untuk NN: (norm_step, target_id_norm, 0, 0)
                    norm_step = path_idx / max(1, len(joint_traj) - 1)
                    target_id_norm = cubes[active_idx]["val"] / 10.0

                    nn_in = torch.tensor(
                        [norm_step, target_id_norm, 0.0, 0.0],
                        dtype=torch.float32,
                        device=device
                    ).unsqueeze(0)  # (1,4)

                    net.eval()
                    with torch.no_grad():
                        out = net(nn_in)
                        cmd_abs = out.cpu().numpy()[0]  # (2,)

                    # Kumpulkan data teacher (supervised)
                    train_X.append([norm_step, target_id_norm, 0.0, 0.0])
                    train_Y.append(ideal_theta_label)

                    # Perintah sudut dari NN
                    curr_theta = safe_theta(cmd_abs)
                    curr_theta = apply_physical_constraints(curr_theta[0], curr_theta[1])

                    # --- HITUNG ERROR POSISI vs JALUR IDEAL ---
                    _, ideal_pos = forward_kinematics(ideal_theta_label[0], ideal_theta_label[1])
                    _, real_pos  = forward_kinematics(curr_theta[0], curr_theta[1])
                    step_err = math.dist(ideal_pos, real_pos)
                    sum_err_iter += step_err
                    step_count_iter += 1

                    # Gripper
                    should_grasp = grip_sched[path_idx]
                    grip_active = bool(should_grasp)

                    if cubes[active_idx]["state"] == "GRIPPED":
                        hand_pos = forward_kinematics(curr_theta[0], curr_theta[1])[1]
                        cubes[active_idx]["pos"] = list(hand_pos)

                    # 0 -> 1 (ambil cube) dengan toleransi
                    if should_grasp == 1 and cubes[active_idx]["state"] == "IDLE":
                        hand_pos = forward_kinematics(curr_theta[0], curr_theta[1])[1]
                        dist = math.dist(hand_pos, cubes[active_idx]["pos"])
                        if dist < 0.15:
                            cubes[active_idx]["state"] = "GRIPPED"

                    # 1 -> 0 (lepas di slot)
                    if should_grasp == 0 and cubes[active_idx]["state"] == "GRIPPED":
                        cubes[active_idx]["state"] = "DONE"

                        fx, fy = cubes[active_idx]["pos"]
                        slot_target = slots[cubes[active_idx]["target_slot_id"]]
                        tx, ty = slot_target["pos"]
                        dist = math.dist((fx, fy), (tx, ty))

                        scr = max(0.0, (0.15 - dist) * (100.0 / 0.15))

                        last_cube_val = cubes[active_idx]["val"]
                        last_cube_score = scr
                        total_score_iter += scr

                        # --- UPDATE CONFUSION MATRIX TOTAL UNTUK KUBUS INI ---
                        update_confusion_for_cube(cubes, slots, active_idx)

                        active_idx = -1

                    path_idx += 1

                # Fallback: path habis tapi cube belum DONE -> skor 0
                elif active_idx != -1 and path_idx >= len(joint_traj):
                    if cubes[active_idx]["state"] != "DONE":
                        last_cube_val = cubes[active_idx]["val"]
                        last_cube_score = 0.0
                    active_idx = -1
                    path_idx = 0

            # ====== FASE TRAINING ======
            else:
                train_timer += 1
                delay = 60 if speed == 1 else 1
                if train_timer > delay:
                    avg_err_iter = sum_err_iter / max(1, step_count_iter)

                    if len(train_X) > 0:
                        inp = torch.tensor(np.array(train_X), dtype=torch.float32, device=device)
                        lbl = torch.tensor(np.array(train_Y), dtype=torch.float32, device=device)

                        net.train()
                        nan_detected = False
                        for _ in range(80):
                            optimizer.zero_grad()
                            out = net(inp)
                            loss = criterion(out, lbl)
                            if torch.isnan(loss):
                                nan_detected = True
                                break
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
                            optimizer.step()

                        if nan_detected:
                            print("Resetting Network (NaN detected)...")
                            net = PuzzleNet()
                            optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
                            save_msg = "BRAIN RESET"
                            save_col = rl.RED
                        else:
                            # Catat error iterasi ini
                            hist_error.append(avg_err_iter)

                            acc_total = compute_accuracy_from_cm(total_conf_matrix) # [BARU]
                            hist_accuracy.append(acc_total)                         # [BARU]

                            # Update best score untuk layout ini
                            nonlocal_puzzle_best = total_score_iter
                            # python nggak boleh assign langsung outer var di scope ini,
                            # jadi kita pakai closure-style: update lewat list atau nonlocal.
                            # Tapi karena kita di body main(), kita bisa pakai nonlocal di atas:
                            # di sini, workaround sederhana:
                            # kita simpan total_score_iter dulu, lalu di luar if kita assign.

                            # Simpan model terakhir
                            latest_state = {
                                'model': net.state_dict(),
                                'score': total_score_iter,
                                'iteration': iteration,
                                'avg_error': avg_err_iter,
                                'acc_total': acc_total,                            # [BARU]
                            }
                            torch.save(latest_state, MODEL_LATEST_PATH)

                            # Update & simpan model terbaik (global)
                            if total_score_iter > global_best_score:
                                global_best_score = total_score_iter
                                
                                best_state = {
                                    'model': net.state_dict(),
                                    'score': global_best_score,
                                    'iteration': iteration,
                                    'avg_error': avg_err_iter,
                                    'acc_total': acc_total,                        # [BARU]
                                }
                                torch.save(best_state, MODEL_BEST_PATH)
                                save_msg = f"NEW GLOBAL BEST! Score: {global_best_score:.1f}"
                                save_col = rl.GREEN
                            else:
                                save_msg = (
                                    f"Learning... (S:{total_score_iter:.1f}, "
                                    f"E:{avg_err_iter:.4f}, Acc:{acc_total:.1f}%)" # [BARU]
                                )
                                save_col = rl.ORANGE

                    # --- UPDATE puzzle_best_score & CEK THRESHOLD 0.9 ---
                    puzzle_best_score = max(puzzle_best_score, total_score_iter)

                    if puzzle_best_score >= THRESHOLD_PUZZLE:
                        save_training_summary(iteration)                        # [BARU]
                        # Layout ini sudah >= 0.9 * max -> generate puzzle baru
                        save_msg = f"Layout SOLVED (>=90%), new puzzle..."
                        save_col = rl.SKYBLUE
                        regenerate_puzzle()

                    # Reset episode (mulai ulang di layout sekarang, atau sudah baru)
                    train_X, train_Y = [], []
                    total_score_iter = 0.0
                    sum_err_iter = 0.0
                    step_count_iter = 0
                    cubes, slots = reset_level()
                    cubes.sort(key=lambda x: x["val"])
                    current_task_num = 0
                    curr_theta = np.array([-math.pi / 2, 0.0], dtype=np.float32)
                    iteration += 1
                    training_mode = False
                    train_timer = 0

        # --- RENDER ---
        rl.begin_drawing()
        rl.clear_background(COLOR_BG)

        # Deadzone walls
        floor_y = world_to_screen(0, SAFE_MIN_Y)[1]
        rl.draw_rectangle(0, floor_y, SCREEN_WIDTH, SCREEN_HEIGHT - floor_y, COLOR_WALL)
        ceil_y = world_to_screen(0, SAFE_MAX_Y)[1]
        rl.draw_rectangle(0, 0, SCREEN_WIDTH, ceil_y, COLOR_WALL)
        left_x = world_to_screen(SAFE_MIN_X, 0)[0]
        rl.draw_rectangle(0, 0, left_x, SCREEN_HEIGHT, COLOR_WALL)
        right_x = world_to_screen(SAFE_MAX_X, 0)[0]
        rl.draw_rectangle(right_x, 0, SCREEN_WIDTH - right_x, SCREEN_HEIGHT, COLOR_WALL)

        # Slots (kanan)
        for s in slots:
            sx, sy = world_to_screen(s["pos"][0], s["pos"][1])
            rl.draw_rectangle(sx - 20, sy - 20, 40, 40, COLOR_SLOT_BG)
            rl.draw_rectangle_lines_ex(rl.Rectangle(sx - 20, sy - 20, 40, 40), 2, rl.WHITE)
            rl.draw_text(str(s["val"]), sx - 5, sy - 10, 20, (255, 255, 255, 220))

        # Cubes (kiri)
        for c in cubes:
            cx, cy = world_to_screen(c["pos"][0], c["pos"][1])
            rl.draw_rectangle(cx - 14, cy - 14, 28, 28, c["color"])
            rl.draw_rectangle_lines(cx - 14, cy - 14, 28, 28, rl.BLACK)
            rl.draw_text(str(c["val"]), cx - 5, cy - 8, 20, rl.BLACK)

        # Robot
        j1 = BASE_POS
        j2_m, j3_m = forward_kinematics(curr_theta[0], curr_theta[1])
        j2 = world_to_screen(j2_m[0], j2_m[1])
        j3 = world_to_screen(j3_m[0], j3_m[1])

        rl.draw_line_ex(j1, j2, 10, COLOR_ROBOT)
        rl.draw_circle_v(j1, 8, COLOR_JOINT)
        rl.draw_circle_v(j2, 7, COLOR_JOINT)
        rl.draw_line_ex(j2, j3, 6, COLOR_ROBOT)
        rl.draw_circle_v(j3, 8, COLOR_GRIP if grip_active else rl.WHITE)

        # HUD
        rl.draw_text(f"Iter: {iteration}", 10, 10, 20, rl.WHITE)
        rl.draw_text("(NN SELF-CORRECT, 0.9 THRESH)", 200, 10, 20, rl.SKYBLUE)

        sc_col = rl.GREEN if last_cube_score > 50 else rl.RED
        rl.draw_text(f"Cube {last_cube_val} Score: {last_cube_score:.1f}", 10, 40, 20, sc_col)
        rl.draw_text(save_msg, 10, 70, 20, save_col)
        rl.draw_text(f"Speed: {speed}x", 650, 10, 20, rl.YELLOW)
        rl.draw_text(f"Global Best: {global_best_score:.1f}", 10, 100, 20, rl.LIGHTGRAY)
        rl.draw_text(f"Puzzle Best: {puzzle_best_score:.1f}", 10, 130, 20, rl.LIGHTGRAY)
        rl.draw_text(f"Threshold: {THRESHOLD_PUZZLE:.1f}", 10, 160, 18, rl.GRAY)

        # Current avg error (on-the-fly)
        cur_avg_err = sum_err_iter / max(1, step_count_iter) if step_count_iter > 0 else 0.0
        rl.draw_text(f"Curr Avg Err: {cur_avg_err:.4f}", 10, 190, 18, rl.LIGHTGRAY)

        if training_mode:
            rl.draw_text("TRAINING WEIGHTS...", 250, 300, 30, rl.GREEN)

        # Plot error gerakan
        draw_error_plot(hist_error)

        rl.end_drawing()

    rl.close_window()


if __name__ == "__main__":
    main()
