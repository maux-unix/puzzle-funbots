import math
import random
import numpy as np
import torch
import torch.nn as nn
import os

# ==========================
#   KONFIGURASI DASAR
# ==========================

MODEL_PATH = "ilc_3x3_best.pth"   # atau ganti ke latest jika mau

LINK_1 = 1.2
LINK_2 = 1.0

SAFE_MIN_X = -1.8
SAFE_MAX_X =  1.8
SAFE_MIN_Y = -0.8
SAFE_MAX_Y =  1.5

HIDDEN_SIZE = 128
MAX_SCORE_THEORETICAL = 9 * 100.0

STEPS_TRAVEL   = 30
STEPS_APPROACH = 60
STEPS_WAIT     = 60


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
#   MODEL (SAMA DENGAN TRAINING ONLINE)
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
        return self.fc3(x)  # (..,2) -> [theta1, theta2]


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
#   PUZZLE LAYOUT (SAMA / FIX)
# ==========================

def build_puzzle(left_vals, right_vals):
    """
    Kiri: cubes (angka left_vals)
    Kanan: slots (angka right_vals)
    layout sama dengan training, bukan random terus.
    """
    cubes = []
    slots = []

    spacing_src = 0.18
    spacing_tgt = 0.22

    src_cx, src_cy = -0.9, 0.0
    tgt_cx, tgt_cy =  0.95, 0.0

    # mapping nilai -> index slot kanan
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

    # urutkan cubes supaya robot ambil 1..9
    cubes.sort(key=lambda x: x["val"])
    return cubes, slots


def generate_dynamic_trajectory(current_hand_pos, pickup_pos, drop_pos):
    """
    Sama konsep dengan training: naik, ke X cube, turun, ambil, naik, ke X slot, turun, lepas, naik.
    """
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
#   SIMULASI 1 EPISODE (LAYOUT SAMA)
# ==========================

def simulate_episode_same_layout(net, device, left_vals, right_vals, noise_init=False, verbose=False):
    cubes, slots = build_puzzle(left_vals, right_vals)

    # bisa kasih sedikit noise awal biar nggak 100% sama
    base_theta = np.array([-math.pi / 2, 0.0], dtype=np.float32)
    if noise_init:
        base_theta[0] += np.random.normal(scale=0.05)
        base_theta[1] += np.random.normal(scale=0.05)
    curr_theta = safe_theta(base_theta)

    total_score = 0.0
    sum_err = 0.0
    step_count = 0

    for cube_idx in range(9):
        cube = cubes[cube_idx]
        cube_pos = cube["pos"][:]
        cube_state = "IDLE"

        hand_pos = forward_kinematics(curr_theta[0], curr_theta[1])[1]
        start_p = hand_pos
        cube_p = cube["pos"]
        slot_p = slots[cube["target_slot_id"]]["pos"]

        joint_traj, grip_sched = generate_dynamic_trajectory(start_p, cube_p, slot_p)

        for step in range(len(joint_traj)):
            ideal_theta = joint_traj[step]

            # Fitur sama persis seperti waktu training online:
            norm_step = step / max(1, len(joint_traj) - 1)
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

            # hitung error posisi vs teacher
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

        # skor cube
        tx, ty = slots[cube["target_slot_id"]]["pos"]
        fx, fy = cube_pos
        dist_drop = math.dist((fx, fy), (tx, ty))
        cube_score = max(0.0, (0.15 - dist_drop) * (100.0 / 0.15))
        total_score += cube_score

        if verbose:
            print(f"  Cube {cube['val']} -> score {cube_score:.2f}")

    avg_err = sum_err / max(1, step_count)
    return total_score, avg_err


# ==========================
#   MAIN
# ==========================

def main():
    device = get_device()
    print("Using device:", device)

    net = PuzzleNet().to(device)

    if not os.path.exists(MODEL_PATH):
        print(f"Model file '{MODEL_PATH}' tidak ditemukan.")
        return

    ckpt = torch.load(MODEL_PATH, map_location=device)
    net.load_state_dict(ckpt['model'])
    net.eval()

    print(f"Loaded model from '{MODEL_PATH}'")
    saved_score = ckpt.get('score', None)
    if saved_score is not None:
        print(f"  (saved best score in train: {saved_score:.2f})")

    # Coba ambil layout dari checkpoint, kalau ada
    left_vals  = ckpt.get('left_vals', list(range(1, 10)))
    right_vals = ckpt.get('right_vals', list(range(1, 10)))

    print("Left layout :", left_vals)
    print("Right layout:", right_vals)

    NUM_EPISODES = 10
    scores = []
    errors = []

    for ep in range(1, NUM_EPISODES + 1):
        score, avg_err = simulate_episode_same_layout(
            net, device,
            left_vals, right_vals,
            noise_init=True,
            verbose=False
        )
        scores.append(score)
        errors.append(avg_err)
        print(
            f"[Episode {ep:02d}] "
            f"Score = {score:.2f} / {MAX_SCORE_THEORETICAL:.1f}, "
            f"Avg Pos Error = {avg_err:.5f}"
        )

    mean_score = sum(scores) / len(scores)
    mean_err   = sum(errors) / len(errors)

    print("\n=== SUMMARY (SAME LAYOUT AS TRAIN) ===")
    print(f"Episodes        : {NUM_EPISODES}")
    print(f"Mean Score      : {mean_score:.2f} / {MAX_SCORE_THEORETICAL:.1f}")
    print(f"Mean Pos Error  : {mean_err:.6f}")
    print(f"Best Episode    : {max(scores):.2f}")
    print(f"Worst Episode   : {min(scores):.2f}")


if __name__ == "__main__":
    main()
