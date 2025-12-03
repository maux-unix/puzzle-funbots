import math
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# ==========================
#   KONFIGURASI
# ==========================

MODEL_PATH = "puzzlenet_multi_layout_best.pth"

# Fisika lengan 2-link (meter)
LINK_1 = 1.2
LINK_2 = 1.0

# Deadzone / batas workspace
SAFE_MIN_X = -1.8
SAFE_MAX_X =  1.8
SAFE_MIN_Y = -0.8
SAFE_MAX_Y =  1.5

HIDDEN_SIZE = 128

# Berapa banyak layout untuk training & evaluasi
NUM_LAYOUTS_TRAIN = 50   # makin besar = makin general, tapi training lebih lama
NUM_LAYOUTS_EVAL  = 10   # layout acak untuk test

EPOCHS       = 100
BATCH_SIZE   = 128
LEARNING_RATE = 0.002
EVAL_EVERY    = 5   # eval di layout baru tiap beberapa epoch

# Trajectory step (sama dengan yang versi sebelumnya)
STEPS_TRAVEL   = 30
STEPS_APPROACH = 60
STEPS_WAIT     = 60

MAX_SCORE_THEORETICAL = 9 * 100.0  # 9 cube × 100 poin max


# ==========================
#   DEVICE (CPU / CUDA / MPS)
# ==========================

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")   # Mac Metal
    return torch.device("cpu")


# ==========================
#   MODEL (PUZZLENET)
#   Input dim = 6:
#   [norm_step, cx, cy, sx, sy, cube_val_norm]
# ==========================

class PuzzleNet(nn.Module):
    def __init__(self, in_dim=6, hidden=HIDDEN_SIZE):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.ln1 = nn.LayerNorm(hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.ln2 = nn.LayerNorm(hidden)
        self.fc3 = nn.Linear(hidden, 2)  # theta1, theta2
        self.act = nn.LeakyReLU(0.1)

        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)

    def forward(self, x):
        x = self.act(self.ln1(self.fc1(x)))
        x = self.act(self.ln2(self.fc2(x)))
        return self.fc3(x)


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
        cos_angle2 = (target_x**2 + target_y**2
                      - LINK_1**2 - LINK_2**2) / (2 * LINK_1 * LINK_2)
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
    clamped_x, clamped_y = curr_x, curr_y
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
    """Permutasi angka 1..9 acak."""
    nums = list(range(1, 10))
    np.random.shuffle(nums)
    return nums

def build_puzzle(left_vals, right_vals):
    """
    Layout grid 3×3:
      - kiri: cube, posisi ditentukan left_vals
      - kanan: slot, posisi ditentukan right_vals
    """
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
    """
    Teacher trajectory dalam koordinat Cartesian:
    naik -> ke atas cube -> turun -> hold -> naik -> ke atas slot -> turun -> hold -> naik.
    Hasil: list theta teacher (hasil IK).
    """
    traj = []

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

        curr_pos = target_pos

    return np.array(traj)


# ==========================
#   DATASET MULTI-LAYOUT
# ==========================

def build_multi_layout_dataset(num_layouts):
    """
    Multi-layout behavior cloning:
      Untuk banyak layout:
        - generate cube + slot
        - generate teacher trajectory untuk tiap cube
        - setiap step: buat sample
          X = [norm_step, cx, cy, sx, sy, cube_val_norm]
          Y = [theta1_teacher, theta2_teacher]
    """
    X_list = []
    Y_list = []

    for layout_idx in range(num_layouts):
        left_vals  = generate_valid_sudoku_3x3()
        right_vals = generate_valid_sudoku_3x3()
        cubes, slots = build_puzzle(left_vals, right_vals)

        home_theta = np.array([-math.pi / 2, 0.0], dtype=np.float32)
        _, home_pos = forward_kinematics(home_theta[0], home_theta[1])

        for cube in cubes:
            cube_pos = cube["pos"]
            slot_pos = slots[cube["target_slot_id"]]["pos"]

            traj = generate_dynamic_trajectory(home_pos, cube_pos, slot_pos)
            n_steps = len(traj)

            for step in range(n_steps):
                theta = traj[step]
                norm_step = step / max(1, n_steps - 1)

                cx, cy = cube_pos
                sx, sy = slot_pos
                cube_val_norm = cube["val"] / 10.0

                X_list.append([norm_step, cx, cy, sx, sy, cube_val_norm])
                Y_list.append(theta)

    X = np.array(X_list, dtype=np.float32)
    Y = np.array(Y_list, dtype=np.float32)
    return X, Y


# ==========================
#   EVALUASI DI LAYOUT BARU
# ==========================

def evaluate_on_random_layouts(net, device, num_layouts_eval=10):
    """
    Untuk tiap layout eval:
      - generate layout baru
      - generate teacher trajectory untuk scoring & error
      - per step: NN prediksi theta_pred
      - hitung error posisi vs teacher
      - akhir: skor cube berdasarkan drop position vs slot target
    """
    net.eval()
    total_score = 0.0
    total_err = 0.0
    total_steps = 0

    with torch.no_grad():
        for _ in range(num_layouts_eval):
            left_vals  = generate_valid_sudoku_3x3()
            right_vals = generate_valid_sudoku_3x3()
            cubes, slots = build_puzzle(left_vals, right_vals)

            home_theta = np.array([-math.pi / 2, 0.0], dtype=np.float32)
            _, home_pos = forward_kinematics(home_theta[0], home_theta[1])

            for cube in cubes:
                cube_pos = cube["pos"]
                slot_pos = slots[cube["target_slot_id"]]["pos"]

                traj_teacher = generate_dynamic_trajectory(home_pos, cube_pos, slot_pos)
                n_steps = len(traj_teacher)

                # jalanin policy step-by-step (pakai teacher step index, tapi sudut = NN)
                for step in range(n_steps):
                    theta_teacher = traj_teacher[step]

                    norm_step = step / max(1, n_steps - 1)
                    cx, cy = cube_pos
                    sx, sy = slot_pos
                    cube_val_norm = cube["val"] / 10.0

                    inp = torch.tensor(
                        [[norm_step, cx, cy, sx, sy, cube_val_norm]],
                        dtype=torch.float32, device=device
                    )
                    theta_pred = net(inp)[0].cpu().numpy()
                    theta_pred = safe_theta(theta_pred)
                    theta_pred = apply_physical_constraints(theta_pred[0], theta_pred[1])

                    # error posisi vs teacher (buat analisis)
                    _, pos_teacher = forward_kinematics(theta_teacher[0], theta_teacher[1])
                    _, pos_pred    = forward_kinematics(theta_pred[0], theta_pred[1])
                    err = math.dist(pos_teacher, pos_pred)
                    total_err += err
                    total_steps += 1

                # pakai prediksi di step terakhir untuk hitung skor drop cube
                norm_step_final = 1.0
                cx, cy = cube_pos
                sx, sy = slot_pos
                cube_val_norm = cube["val"] / 10.0
                inp_final = torch.tensor(
                    [[norm_step_final, cx, cy, sx, sy, cube_val_norm]],
                    dtype=torch.float32, device=device
                )
                theta_pred_final = net(inp_final)[0].cpu().numpy()
                theta_pred_final = safe_theta(theta_pred_final)
                theta_pred_final = apply_physical_constraints(
                    theta_pred_final[0], theta_pred_final[1]
                )
                _, pos_pred_final = forward_kinematics(
                    theta_pred_final[0], theta_pred_final[1]
                )
                fx, fy = pos_pred_final
                tx, ty = slot_pos
                dist_drop = math.dist((fx, fy), (tx, ty))
                cube_score = max(0.0, (0.15 - dist_drop) * (100.0 / 0.15))
                total_score += cube_score

    avg_err = total_err / max(1, total_steps)
    avg_score = total_score / max(1, num_layouts_eval)
    return avg_score, avg_err


# ==========================
#   MAIN TRAINING LOOP
# ==========================

def main():
    np.random.seed(42)
    torch.manual_seed(42)

    device = get_device()
    print("Device:", device)

    # 1. Build dataset dari banyak layout
    print("Building multi-layout dataset...")
    X, Y = build_multi_layout_dataset(NUM_LAYOUTS_TRAIN)
    print("Dataset:", X.shape, Y.shape)

    dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(Y))
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 2. Init model
    net = PuzzleNet(in_dim=6).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)

    best_score = -1.0

    # 3. Training loop
    for epoch in range(1, EPOCHS+1):
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

        # 4. Evaluasi di layout baru (generalization check)
        if epoch % EVAL_EVERY == 0 or epoch == 1 or epoch == EPOCHS:
            avg_score, avg_err = evaluate_on_random_layouts(
                net, device, NUM_LAYOUTS_EVAL
            )
            print(f"[Epoch {epoch:03d}] Loss={avg_loss:.5f} "
                  f"| EvalScore≈{avg_score:.2f} "
                  f"| AvgErr={avg_err:.4f}")

            if avg_score > best_score:
                best_score = avg_score
                torch.save({
                    'model': net.state_dict(),
                    'best_score': best_score,
                    'in_dim': 6,
                    'note': 'multi-layout training, input=[step,cx,cy,sx,sy,val_norm]',
                }, MODEL_PATH)
                print(f"   -> New best score, model saved to {MODEL_PATH}")
        else:
            print(f"[Epoch {epoch:03d}] Loss={avg_loss:.5f}")

    print("Training complete. Best eval score:", best_score)


if __name__ == "__main__":
    main()
