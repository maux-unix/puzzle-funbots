# view_raylib_sudoku.py

import math
import numpy as np
import torch
import torch.nn as nn
import pyray as rl

# Harus sama dengan training
MODEL_BEST_PATH = "ilc_3x3_best_cli.pth"

SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
BASE_POS = (400, 300)     # basis robot di tengah
PIXELS_PER_METER = 180.0

LINK_1 = 1.2
LINK_2 = 1.0

SAFE_MIN_X = -1.8
SAFE_MAX_X =  1.8
SAFE_MIN_Y = -0.8
SAFE_MAX_Y =  1.5

HIDDEN_SIZE = 128

STEPS_TRAVEL   = 30
STEPS_APPROACH = 60
STEPS_WAIT     = 60

MAX_SCORE_THEORETICAL = 9 * 100.0

# Warna
COLOR_BG      = (30, 30, 30, 255)
COLOR_ROBOT   = (220, 220, 220, 255)
COLOR_JOINT   = (50, 50, 50, 255)
COLOR_GRIP    = (50, 255, 50, 255)
COLOR_SLOT_BG = (60, 60, 60, 255)

COLORS_NUM = [
    (255, 100, 100, 255), (100, 255, 100, 255), (100, 100, 255, 255),
    (255, 255, 100, 255), (100, 255, 255, 255), (255, 100, 255, 255),
    (255, 150, 50, 255),  (50, 255, 150, 255),  (150, 50, 255, 255)
]


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
        return self.fc3(x)


# ==========================
#   KINEMATIKA & KONVERSI
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

def world_to_screen(x, y):
    px = BASE_POS[0] + (x * PIXELS_PER_METER)
    py = BASE_POS[1] - (y * PIXELS_PER_METER)
    return int(px), int(py)


# ==========================
#   PUZZLE LAYOUT & TRAJECTORY
# ==========================

def build_puzzle(left_vals, right_vals):
    cubes = []
    slots = []

    spacing_src = 0.18
    spacing_tgt = 0.22

    src_cx, src_cy = -0.9, 0.0
    tgt_cx, tgt_cy =  0.95, 0.0

    val_to_slot_idx = {v: i for i, v in enumerate(right_vals)}

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
            "state": "IDLE",
            "target_slot_id": val_to_slot_idx[val],
            "color": COLORS_NUM[val-1],
        })

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
#   MAIN (RAYLIB VIEWER)
# ==========================

def main():
    # Load model
    ckpt = torch.load(MODEL_BEST_PATH, map_location="cpu")
    net = PuzzleNet()
    net.load_state_dict(ckpt['model'])
    net.eval()

    left_vals  = ckpt.get('left_vals', list(range(1, 10)))
    right_vals = ckpt.get('right_vals', list(range(1, 10)))

    cubes, slots = build_puzzle(left_vals, right_vals)

    # Setup Raylib
    rl.init_window(SCREEN_WIDTH, SCREEN_HEIGHT, "Neural Sudoku Viewer (NN Only)")
    rl.set_target_fps(60)

    curr_theta = np.array([-math.pi / 2, 0.0], dtype=np.float32)
    _, home_pos = forward_kinematics(curr_theta[0], curr_theta[1])

    active_idx = 0
    joint_traj = None
    grip_sched = None
    step_idx = 0

    total_score = 0.0
    finished_once = False

    SPEED_LEVELS = [1, 2, 5, 10, 20, 50]
    speed_idx = 0

    while not rl.window_should_close():
        # input speed
        if rl.is_key_pressed(rl.KEY_RIGHT):
            speed_idx = min(len(SPEED_LEVELS)-1, speed_idx+1)
        if rl.is_key_pressed(rl.KEY_LEFT):
            speed_idx = max(0, speed_idx-1)
        speed = SPEED_LEVELS[speed_idx]

        # logic
        for _ in range(speed):
            if active_idx >= len(cubes):
                # semua cube selesai
                finished_once = True
                # tekan R untuk reset demo
                if rl.is_key_pressed(rl.KEY_R):
                    cubes, slots = build_puzzle(left_vals, right_vals)
                    curr_theta = np.array([-math.pi / 2, 0.0], dtype=np.float32)
                    _, home_pos = forward_kinematics(curr_theta[0], curr_theta[1])
                    active_idx = 0
                    joint_traj = None
                    grip_sched = None
                    step_idx = 0
                    total_score = 0.0
                    finished_once = False
                break

            cube = cubes[active_idx]

            if joint_traj is None:
                # generate trajectory untuk cube ini
                _, home_pos = forward_kinematics(curr_theta[0], curr_theta[1])
                cube_pos = cube["pos"]
                slot_pos = slots[cube["target_slot_id"]]["pos"]
                joint_traj, grip_sched = generate_dynamic_trajectory(
                    home_pos, cube_pos, slot_pos
                )
                step_idx = 0
                cube["state"] = "IDLE"

            if step_idx < len(joint_traj):
                ideal_theta = joint_traj[step_idx]
                norm_step = step_idx / max(1, len(joint_traj)-1)
                target_id_norm = cube["val"] / 10.0

                nn_in = torch.tensor(
                    [norm_step, target_id_norm, 0.0, 0.0],
                    dtype=torch.float32
                ).unsqueeze(0)

                with torch.no_grad():
                    cmd_abs = net(nn_in)[0].numpy()

                curr_theta = safe_theta(cmd_abs)
                curr_theta = apply_physical_constraints(curr_theta[0], curr_theta[1])

                should_grasp = grip_sched[step_idx]

                if should_grasp == 1 and cube["state"] == "IDLE":
                    hand_pos = forward_kinematics(curr_theta[0], curr_theta[1])[1]
                    if math.dist(hand_pos, cube["pos"]) < 0.15:
                        cube["state"] = "GRIPPED"

                if cube["state"] == "GRIPPED":
                    hand_pos = forward_kinematics(curr_theta[0], curr_theta[1])[1]
                    cube["pos"] = list(hand_pos)

                if should_grasp == 0 and cube["state"] == "GRIPPED":
                    cube["state"] = "DONE"
                    # hitung skor cube
                    fx, fy = cube["pos"]
                    tx, ty = slots[cube["target_slot_id"]]["pos"]
                    dist_drop = math.dist((fx, fy), (tx, ty))
                    cube_score = max(0.0, (0.15 - dist_drop) * (100.0 / 0.15))
                    total_score += cube_score

                    # lanjut ke cube berikutnya
                    active_idx += 1
                    joint_traj = None
                    grip_sched = None
                    step_idx = 0
                else:
                    step_idx += 1
            else:
                # trajectory habis tapi belum DONE (fallback)
                cube["state"] = "DONE"
                active_idx += 1
                joint_traj = None
                grip_sched = None
                step_idx = 0

        # render
        rl.begin_drawing()
        rl.clear_background(COLOR_BG)

        # draw slots
        for s in slots:
            sx, sy = world_to_screen(s["pos"][0], s["pos"][1])
            rl.draw_rectangle(sx-20, sy-20, 40, 40, COLOR_SLOT_BG)
            rl.draw_rectangle_lines_ex(rl.Rectangle(sx-20, sy-20, 40, 40), 2, rl.WHITE)
            rl.draw_text(str(s["val"]), sx-5, sy-10, 20, (255,255,255,120))

        # draw cubes
        for c in cubes:
            cx, cy = world_to_screen(c["pos"][0], c["pos"][1])
            rl.draw_rectangle(cx-14, cy-14, 28, 28, c["color"])
            rl.draw_rectangle_lines(cx-14, cy-14, 28, 28, rl.BLACK)
            rl.draw_text(str(c["val"]), cx-5, cy-8, 20, rl.BLACK)

        # draw robot
        j1 = BASE_POS
        j2_m, j3_m = forward_kinematics(curr_theta[0], curr_theta[1])
        j2 = world_to_screen(j2_m[0], j2_m[1])
        j3 = world_to_screen(j3_m[0], j3_m[1])

        rl.draw_line_ex(j1, j2, 10, COLOR_ROBOT)
        rl.draw_circle_v(j1, 8, COLOR_JOINT)
        rl.draw_circle_v(j2, 7, COLOR_JOINT)
        rl.draw_line_ex(j2, j3, 6, COLOR_ROBOT)

        grip_on = (active_idx < len(cubes) and cubes[active_idx]["state"] == "GRIPPED")
        rl.draw_circle_v(j3, 8, COLOR_GRIP if grip_on else rl.WHITE)

        rl.draw_text("Neural Sudoku Viewer (NN Only)", 10, 10, 20, rl.WHITE)
        rl.draw_text(f"Speed: {SPEED_LEVELS[speed_idx]}x", 10, 35, 20, rl.YELLOW)
        rl.draw_text(f"Total Score: {total_score:.1f} / {MAX_SCORE_THEORETICAL:.1f}",
                     10, 60, 20, rl.GREEN)

        if finished_once:
            rl.draw_text("FINISHED! Press R to reset demo.",
                         180, 560, 20, rl.SKYBLUE)

        rl.end_drawing()

    rl.close_window()


if __name__ == "__main__":
    main()
