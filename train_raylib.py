# train_tui_sudoku.py

import math
import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# ==========================
#   KONFIGURASI
# ==========================

MODEL_BEST_PATH = "ilc_3x3_best_cli.pth"
MODEL_LAST_PATH = "ilc_3x3_last_cli.pth"

LINK_1 = 1.2
LINK_2 = 1.0

SAFE_MIN_X = -1.8
SAFE_MAX_X =  1.8
SAFE_MIN_Y = -0.8
SAFE_MAX_Y =  1.5

HIDDEN_SIZE = 128

# ----- MODE CEPAT / NORMAL -----
FAST_MODE = True  # ubah True kalau mau lebih cepat

if FAST_MODE:
    EPOCHS      = 10000
    BATCH_SIZE  = 128
else:
    EPOCHS      = 200
    BATCH_SIZE  = 64

LEARNING_RATE = 0.002
EVAL_EVERY    = 5   # setiap berapa epoch di-eval & update TUI

STEPS_TRAVEL   = 30
STEPS_APPROACH = 60
STEPS_WAIT     = 60

MAX_SCORE_THEORETICAL = (9 * 100.0) * 0.8
NUM_CLASSES = 9  # angka 1..9

# Resume control
RESUME_FROM_LAST = True   # lanjut dari last model jika ada
RESUME_FROM_BEST = False  # atau dari best model (jangan dua-duanya True)

# confusion matrix total (global)
total_conf_matrix = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)

# history buat display
hist_loss = []
hist_score = []
hist_error = []
hist_acc = []


# ==========================
#   DEVICE (CPU / CUDA / MPS)
# ==========================

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ==========================
#   MODEL (PUZZLENET)
# ==========================

class PuzzleNet(nn.Module):
    def __init__(self):
        super(PuzzleNet, self).__init__()
        self.fc1 = nn.Linear(4, HIDDEN_SIZE)
        self.ln1 = nn.LayerNorm(HIDDEN_SIZE)
        self.fc2 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.ln2 = nn.LayerNorm(HIDDEN_SIZE)
        self.fc3 = nn.Linear(HIDDEN_SIZE, 2)
        self.relu = nn.LeakyReLU(0.1)

        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)

    def forward(self, x):
        x = self.relu(self.ln1(self.fc1(x)))
        x = self.relu(self.ln2(self.fc2(x)))
        return self.fc3(x)  # -> [theta1, theta2]


# ==========================
#   KINEMATIKA & KONSTRAIN
# ==========================

def safe_theta(theta: np.ndarray) -> np.ndarray:
    if np.isnan(theta).any() or np.isinf(theta).any():
        return np.array([-math.pi / 2, 0.0], dtype=np.float32)
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

def apply_physical_constraints(theta1, theta2):
    _, hand_pos = forward_kinematics(theta1, theta2)
    curr_x, curr_y = hand_pos

    clamped_x = curr_x
    clamped_y = curr_y
    hit_wall = False

    if curr_x > SAFE_MAX_X:
        clamped_x = SAFE_MAX_X; hit_wall = True
    if curr_x < SAFE_MIN_X:
        clamped_x = SAFE_MIN_X; hit_wall = True
    if curr_y < SAFE_MIN_Y:
        clamped_y = SAFE_MIN_Y; hit_wall = True
    if curr_y > SAFE_MAX_Y:
        clamped_y = SAFE_MAX_Y; hit_wall = True

    if hit_wall:
        corrected = inverse_kinematics(clamped_x, clamped_y)
        if corrected is not None:
            return safe_theta(corrected)

    return np.array([theta1, theta2], dtype=np.float32)


# ==========================
#   PUZZLE LAYOUT & TRAJECTORY
# ==========================

def generate_valid_sudoku_3x3():
    nums = list(range(1, 10))
    np.random.shuffle(nums)
    return nums

def build_puzzle(left_vals, right_vals):
    cubes = []
    slots = []

    spacing_src = 0.18
    spacing_tgt = 0.22

    src_cx, src_cy = -0.9, 0.0
    tgt_cx, tgt_cy =  0.95, 0.0

    val_to_slot_idx = {v: i for i, v in enumerate(right_vals)}

    # cubes kiri
    for i in range(9):
        val = left_vals[i]
        r = i // 3
        c = i % 3

        off_x = (c - 1) * spacing_src
        off_y = (1 - r) * spacing_src

        cx = src_cx + off_x
        cy = src_cy + off_y

        cubes.append({
            "val": val,
            "pos": [cx, cy],
            "target_slot_id": val_to_slot_idx[val],
        })

    # slots kanan
    for i in range(9):
        val = right_vals[i]
        r = i // 3
        c = i % 3

        off_x = (c - 1) * spacing_tgt
        off_y = (1 - r) * spacing_tgt

        sx = tgt_cx + off_x
        sy = tgt_cy + off_y

        slots.append({
            "val": val,
            "pos": [sx, sy],
        })

    cubes.sort(key=lambda x: x["val"])
    return cubes, slots

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

    curr_pos = current_hand_pos
    curr_grip = 0

    for target_pos, target_grip, steps in points_config:
        for s in range(steps):
            alpha = s / max(1, steps)
            pos_x = curr_pos[0] * (1 - alpha) + target_pos[0] * alpha
            pos_y = curr_pos[1] * (1 - alpha) + target_pos[1] * alpha

            ik = inverse_kinematics(pos_x, pos_y)
            if ik is not None:
                traj.append(ik)
            else:
                if len(traj) > 0:
                    traj.append(traj[-1].copy())
                else:
                    traj.append(np.array([-math.pi / 2, 0.0], dtype=np.float32))

            if steps == STEPS_WAIT:
                grip.append(target_grip)
            else:
                grip.append(curr_grip if alpha < 0.5 else target_grip)

        curr_pos = target_pos
        curr_grip = target_grip

    return np.array(traj), grip


# ==========================
#   BUILD DATASET (teacher)
# ==========================

def build_offline_dataset(left_vals, right_vals):
    cubes, slots = build_puzzle(left_vals, right_vals)

    X_list = []
    Y_list = []

    home_theta = np.array([-math.pi / 2, 0.0], dtype=np.float32)
    _, home_pos = forward_kinematics(home_theta[0], home_theta[1])

    for cube in cubes:
        cube_pos = cube["pos"]
        slot_pos = slots[cube["target_slot_id"]]["pos"]

        joint_traj, grip_sched = generate_dynamic_trajectory(home_pos, cube_pos, slot_pos)
        n_steps = len(joint_traj)

        for step in range(n_steps):
            ideal_theta = joint_traj[step]

            norm_step = step / max(1, n_steps - 1)
            target_id_norm = cube["val"] / 10.0

            X_list.append([norm_step, target_id_norm, 0.0, 0.0])
            Y_list.append(ideal_theta)

    X = np.array(X_list, dtype=np.float32)
    Y = np.array(Y_list, dtype=np.float32)
    return X, Y


# ==========================
#   CONFUSION MATRIX + EVAL
# ==========================

def update_confusion_for_cube(cube_val, final_pos, slots):
    global total_conf_matrix
    true_val = cube_val
    true_idx = true_val - 1

    fx, fy = final_pos
    pred_val = None
    min_d = 1e9

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

def simulate_episode_and_update_cm(net, device, left_vals, right_vals):
    cubes, slots = build_puzzle(left_vals, right_vals)

    home_theta = np.array([-math.pi / 2, 0.0], dtype=np.float32)
    _, home_pos = forward_kinematics(home_theta[0], home_theta[1])

    total_score = 0.0
    sum_err = 0.0
    step_count = 0

    for cube in cubes:
        curr_theta = home_theta.copy()
        cube_pos = cube["pos"][:]
        cube_state = "IDLE"
        slot_pos = slots[cube["target_slot_id"]]["pos"]

        joint_traj, grip_sched = generate_dynamic_trajectory(home_pos, cube_pos, slot_pos)
        n_steps = len(joint_traj)

        for step in range(n_steps):
            ideal_theta = joint_traj[step]

            norm_step = step / max(1, n_steps - 1)
            target_id_norm = cube["val"] / 10.0

            nn_in = torch.tensor(
                [norm_step, target_id_norm, 0.0, 0.0],
                dtype=torch.float32,
                device=device
            ).unsqueeze(0)

            net.eval()
            with torch.no_grad():
                cmd_abs = net(nn_in)[0].cpu().numpy()

            curr_theta = safe_theta(cmd_abs)
            curr_theta = apply_physical_constraints(curr_theta[0], curr_theta[1])

            _, ideal_pos = forward_kinematics(ideal_theta[0], ideal_theta[1])
            _, real_pos  = forward_kinematics(curr_theta[0], curr_theta[1])
            step_err = math.dist(ideal_pos, real_pos)
            sum_err += step_err
            step_count += 1

            should_grasp = grip_sched[step]
            if should_grasp == 1 and cube_state == "IDLE":
                hand_pos = forward_kinematics(curr_theta[0], curr_theta[1])[1]
                if math.dist(hand_pos, cube_pos) < 0.15:
                    cube_state = "GRIPPED"

            if cube_state == "GRIPPED":
                cube_pos = list(forward_kinematics(curr_theta[0], curr_theta[1])[1])

            if should_grasp == 0 and cube_state == "GRIPPED":
                cube_state = "DONE"
                break

        tx, ty = slot_pos
        fx, fy = cube_pos
        dist_drop = math.dist((fx, fy), (tx, ty))
        cube_score = max(0.0, (0.15 - dist_drop) * (100.0 / 0.15))
        total_score += cube_score

        update_confusion_for_cube(cube["val"], (fx, fy), slots)

    avg_err = sum_err / max(1, step_count)
    return total_score, avg_err


# ==========================
#   TUI DISPLAY
# ==========================

def clear_screen():
    if os.name == "nt":
        os.system("cls")
    else:
        sys.stdout.write("\033[2J\033[H")
        sys.stdout.flush()

def print_confusion_matrix(cm: np.ndarray):
    labels = [str(i+1) for i in range(NUM_CLASSES)]
    row_header = "     " + " ".join(f"{l:>3}" for l in labels)
    print(row_header)
    for i in range(NUM_CLASSES):
        row_str = f"{i+1:>3} |"
        for j in range(NUM_CLASSES):
            v = cm[i, j]
            row_str += f" {v:>3}"
        print(row_str)

def tail(lst, n=10):
    return lst[-n:] if len(lst) > n else lst

def show_tui(epoch, epoch_start, epoch_end,
             avg_loss, eval_score, best_score, avg_err, acc_total):
    clear_screen()
    print("===============================================")
    print("  Neural Sudoku ILC Training (CLI / TUI Mode)  ")
    print("===============================================")
    print(f"FAST_MODE     : {FAST_MODE}")
    print(f"Epoch         : {epoch} (run {epoch-epoch_start+1}/{epoch_end-epoch_start+1})")
    print(f"Loss (train)  : {avg_loss:.6f}")
    print(f"Eval score    : {eval_score:.2f} / {MAX_SCORE_THEORETICAL:.1f}")
    print(f"Best score    : {best_score:.2f}")
    print(f"Avg pos error : {avg_err:.6f}")
    print(f"Acc (CM total): {acc_total:.2f}%")
    print("-----------------------------------------------")

    if hist_loss:
        print("Loss history  :", ", ".join(f"{x:.4f}" for x in tail(hist_loss)))
    if hist_score:
        print("Score history :", ", ".join(f"{x:.1f}" for x in tail(hist_score)))
    if hist_error:
        print("Err history   :", ", ".join(f"{x:.4f}" for x in tail(hist_error)))
    if hist_acc:
        print("Acc history   :", ", ".join(f"{x:.1f}" for x in tail(hist_acc)))

    print("\nConfusion Matrix (Total):")
    print_confusion_matrix(total_conf_matrix)
    print("-----------------------------------------------")
    print("Ctrl+C untuk stop training, model best & last akan disimpan.")


# ==========================
#   MAIN TRAINING LOOP
# ==========================

def main():
    global total_conf_matrix

    np.random.seed(42)
    torch.manual_seed(42)

    device = get_device()

    # ---- INIT / RESUME ----
    left_vals  = None
    right_vals = None
    best_eval_score = -1.0
    start_epoch = 1

    net = PuzzleNet().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)

    resumed_from = None

    if RESUME_FROM_LAST and os.path.exists(MODEL_LAST_PATH):
        ckpt = torch.load(MODEL_LAST_PATH, map_location=device)
        net.load_state_dict(ckpt['model'])
        left_vals  = ckpt.get('left_vals', None)
        right_vals = ckpt.get('right_vals', None)
        prev_epoch = ckpt.get('epoch', 0)
        start_epoch = prev_epoch + 1
        prev_score = ckpt.get('score', None)
        if prev_score is not None:
            best_eval_score = prev_score
        resumed_from = "LAST"
    elif RESUME_FROM_BEST and os.path.exists(MODEL_BEST_PATH):
        ckpt = torch.load(MODEL_BEST_PATH, map_location=device)
        net.load_state_dict(ckpt['model'])
        left_vals  = ckpt.get('left_vals', None)
        right_vals = ckpt.get('right_vals', None)
        prev_epoch = ckpt.get('epoch', 0)
        start_epoch = prev_epoch + 1
        prev_score = ckpt.get('score', None)
        if prev_score is not None:
            best_eval_score = prev_score
        resumed_from = "BEST"

    # kalau belum ada layout dari checkpoint → generate baru
    if left_vals is None or right_vals is None:
        left_vals  = generate_valid_sudoku_3x3()
        right_vals = generate_valid_sudoku_3x3()

    end_epoch = start_epoch + EPOCHS - 1

    print("Using device :", device)
    print("FAST_MODE    :", FAST_MODE)
    if resumed_from:
        print(f"RESUME FROM  : {resumed_from} checkpoint")
        print(f"Start epoch  : {start_epoch}")
        print(f"Prev best    : {best_eval_score:.2f}")
    else:
        print("RESUME FROM  : fresh (no checkpoint used)")
    print("Training layout:")
    print("  Left :", left_vals)
    print("  Right:", right_vals)
    time.sleep(1)

    # Dataset offline dari teacher
    X, Y = build_offline_dataset(left_vals, right_vals)
    print("Dataset size:", X.shape, "(samples x features)")
    time.sleep(1)

    dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(Y))
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    best_state = None
    last_state = None

    try:
        for epoch in range(start_epoch, end_epoch + 1):
            net.train()
            running_loss = 0.0
            count = 0

            for batch_x, batch_y in loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)

                optimizer.zero_grad()
                out = net(batch_x)
                loss = criterion(out, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
                optimizer.step()

                running_loss += loss.item() * batch_x.size(0)
                count += batch_x.size(0)

            avg_loss = running_loss / max(1, count)
            hist_loss.append(avg_loss)

            # last_state buat jaga-jaga
            last_state = {
                'model': net.state_dict(),
                'score': None,
                'epoch': epoch,
                'left_vals': left_vals,
                'right_vals': right_vals
            }

            if epoch % EVAL_EVERY == 0 or epoch == start_epoch or epoch == end_epoch:
                eval_score, eval_err = simulate_episode_and_update_cm(
                    net, device, left_vals, right_vals
                )
                acc_total = compute_accuracy_from_cm(total_conf_matrix)

                hist_score.append(eval_score)
                hist_error.append(eval_err)
                hist_acc.append(acc_total)

                if eval_score > best_eval_score:
                    best_eval_score = eval_score
                    best_state = {
                        'model': net.state_dict(),
                        'score': best_eval_score,
                        'epoch': epoch,
                        'left_vals': left_vals,
                        'right_vals': right_vals
                    }

                # update last_state.score juga
                last_state['score'] = eval_score

                show_tui(epoch, start_epoch, end_epoch,
                         avg_loss, eval_score, best_eval_score,
                         eval_err, acc_total)
                time.sleep(0.2)

    except KeyboardInterrupt:
        print("\n[INFO] Training dihentikan oleh user (Ctrl+C).")

    # setelah loop
    if best_state is None and last_state is not None:
        best_state = last_state
        best_eval_score = best_state.get('score', 0.0) or 0.0

    if best_state is not None:
        torch.save(best_state, MODEL_BEST_PATH)
        print(f"\nBest model saved to: {MODEL_BEST_PATH}")
        print(f"  Best Eval Score: {best_eval_score:.2f}")
        print(f"  Epoch: {best_state['epoch']}")
        print("  Layout:")
        print("    Left :", best_state['left_vals'])
        print("    Right:", best_state['right_vals'])
    else:
        print("\n[WARN] Tidak ada state best yang bisa disimpan.")

    if last_state is not None:
        torch.save(last_state, MODEL_LAST_PATH)
        print(f"Last model state saved to: {MODEL_LAST_PATH}")


if __name__ == "__main__":
    main()
