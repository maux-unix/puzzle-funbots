import math
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# TODO: ganti ini sesuai file kamu
# misal: from train_multi_layout import PuzzleNet
from train_raylib import PuzzleNet  

MAX_VIS = 16  # neuron per layer yang digambar

def build_layer_info(model):
    in_dim   = min(model.fc1.in_features, MAX_VIS)
    h1_dim   = min(model.fc1.out_features, MAX_VIS)
    h2_dim   = min(model.fc2.out_features, MAX_VIS)
    out_dim  = min(model.fc3.out_features, MAX_VIS)
    return [("Input", in_dim), ("Hidden1", h1_dim),
            ("Hidden2", h2_dim), ("Output", out_dim)]


def plot_puzzlenet_graph(model: nn.Module, title="PuzzleNet Graph"):
    layers = build_layer_info(model)

    n_layers = len(layers)
    max_neurons = max(n for _, n in layers)

    # posisi node per layer
    node_positions = {}  # (layer_idx, neuron_idx) -> (x, y)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis("off")

    x_spacing = 2.0  # jarak antar layer
    
    # y_spacing dinamis: kalau neuron banyak -> jaraknya diperkecil
    base_height = 10.0  # tinggi total kira2 10 "unit"
    if max_neurons > 1:
        y_spacing = base_height / (max_neurons - 1)
    else:
        y_spacing = 1.0

    # 1. Gambar node
    for li, (lname, n_neurons) in enumerate(layers):
        x = li * x_spacing
        # kita center di vertical
        total_height = (n_neurons - 1) * y_spacing if n_neurons > 1 else 1
        y_start = -total_height / 2.0

        for ni in range(n_neurons):
            y = y_start + ni * y_spacing
            node_positions[(li, ni)] = (x, y)
            ax.scatter(x, y, s=200, zorder=3, edgecolors="black", facecolors="white")
            # label neuron kecil
            ax.text(x, y, f"{ni}", ha="center", va="center", fontsize=8)

        # label nama layer
        ax.text(x, y_start - 0.8, lname, ha="center", va="center", fontsize=10, fontweight="bold")

    # 2. Gambar sambungan antar layer
    #    Pakai weight untuk warna/ketebalan (optional)
    with torch.no_grad():
        W1 = model.fc1.weight.detach().cpu().numpy()  # [h1, in]
        W2 = model.fc2.weight.detach().cpu().numpy()  # [h2, h1]
        W3 = model.fc3.weight.detach().cpu().numpy()  # [out, h2]

        # potong ke neuron yang keliatan
        W1 = W1[:layers[1][1], :layers[0][1]]
        W2 = W2[:layers[2][1], :layers[1][1]]
        W3 = W3[:layers[3][1], :layers[2][1]]


    def draw_weight_layer(weight_matrix, layer_from_idx, layer_to_idx):
        """Gambar semua sambungan dari layer_from -> layer_to."""
        w = weight_matrix
        if w.size == 0:
            return
        w_abs = np.abs(w)
        w_max = np.max(w_abs) if np.max(w_abs) > 0 else 1.0

        out_neurons, in_neurons = w.shape

        for o in range(out_neurons):
            for i in range(in_neurons):
                x1, y1 = node_positions[(layer_from_idx, i)]
                x2, y2 = node_positions[(layer_to_idx, o)]

                val = w[o, i]
                strength = abs(val) / w_max

                # ketebalan garis berdasar |weight|
                lw = 0.3 + 2.0 * strength

                # warna: positif = hijau, negatif = merah
                color = "green" if val >= 0 else "red"

                ax.plot([x1, x2], [y1, y2],
                        linewidth=lw,
                        color=color,
                        alpha=0.4,
                        zorder=1)

    # fc1: input (0) -> hidden1 (1)
    draw_weight_layer(W1, 0, 1)
    # fc2: hidden1 (1) -> hidden2 (2)
    draw_weight_layer(W2, 1, 2)
    # fc3: hidden2 (2) -> output (3)
    draw_weight_layer(W3, 2, 3)

    ax.set_title(title, fontsize=14)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # instansiasi model
    try:
        # kalau versi multi-layout (ada argumen in_dim)
        model = PuzzleNet(in_dim=6)
    except TypeError:
        # kalau versi lama (tanpa argumen)
        model = PuzzleNet()

    # opsional: load checkpoint kalau mau lihat weight hasil training
    # ckpt = torch.load("puzzlenet_multi_layout_best.pth", map_location="cpu")
    # model.load_state_dict(ckpt["model"])

    plot_puzzlenet_graph(model, title="Sambungan Neuron PuzzleNet")
