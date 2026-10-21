from scipy.integrate import odeint
from matplotlib.ticker import AutoMinorLocator, MultipleLocator

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
Range = 3.0
rng = np.random.default_rng(0)


def Phi(x, dt):
    sol = odeint(ode_model, x, [0.0, dt], rtol=1e-8, atol=1e-8)
    return sol[-1]

def ode_model(x, t):
    return [x[1], -x[0] + (1 - x[0] ** 2) * x[1]]


def load_model(model_path):
    # zhuo: read the adaptive subdivision model from json
    with open(model_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    pieces = []
    for p in data:
        exponents = p.get("exponents")
        pieces.append({
            "bounds": [tuple(map(float, b)) for b in p["bounds"]],
            "w": np.asarray(p["w"], dtype=float),
            "degree": int(p.get("degree", 1)),
            "exponents": [tuple(e) for e in exponents] if exponents is not None else None,
            "epsilon": float(p.get("epsilon", 0.0)),
            "n_sample": int(p.get("n_sample", len(p.get("idx", [])))),
            "idx": np.asarray(p.get("idx", []), dtype=int),
        })
    return pieces


def select_piece(xs, pieces):
    # zhuo: here we select the best piece which has the minimum error for the given state xs
    #       tol is the tolerance parameter because sometimes the state will be out of the boundary we learned,
    #       and it can be adjusted to a greater number if the system throws the ValueError below (like 0.1)
    xs = np.asarray(xs, dtype=float).ravel()
    tol = 1e-5
    # tol_vec = np.asarray(tol, dtype=float).ravel()

    d = len(xs)
    models = []
    for p in pieces:
        select = True
        for i in range(d):
            lo, hi = p["bounds"][i]
            # tol = tol_vec[i]
            if not (lo - tol <= xs[i] <= hi + tol):
            # if not (lo <= xs[i] <= hi):
                select = False
                break
        if select:
            models.append(p)

    if not models:
        raise ValueError(f"not able to find a matched model with the given factors: {xs}")
    # if not models:
    #     print("\n=== DEBUG: no matched piece ===")
    #     print("xs:", xs)

    #     for p_idx, p in enumerate(pieces[:5]):
    #         print(f"\n-- checking piece {p_idx} --")
    #         for i in range(d):
    #             lo, hi = p["bounds"][i]
    #             tol = tol_vec[i]
    #             if not (lo - tol <= xs[i] <= hi + tol):
    #                 print(f"[FAIL] dim={i}, xs={xs[i]:.6f}, "
    #                       f"range=[{lo:.6f},{hi:.6f}], tol={tol:.6f}")
    #                 break

    #     raise ValueError(f"not able to find a matched model with the given factors: {xs}")
    return min(models, key=lambda p: p["epsilon"])


def predict(xs, pieces):
    # Zhuo: here we use the pieces to calculate the output of the current state(y_pred) with the w
    xs = np.asarray(xs, dtype=float).ravel()
    p = select_piece(xs, pieces)

    w = p["w"]

    if p["exponents"] is not None:
        phi = np.array([np.prod(xs ** np.asarray(e)) for e in p["exponents"]], dtype=float)
        if w.size != phi.size:
            raise ValueError(f"w size ({w.size}) != feature size ({phi.size})")
        y_pred = float(np.dot(w, phi))
    else:
        if w.size != xs.size + 1:
            raise ValueError(f"w size ({w.size}) != input size + 1 ({xs.size + 1})")
        y_pred = float(w[0] + np.dot(w[1:], xs))
    return y_pred, p


# global x_min, x_max
# x_min = np.array([-Range, -Range, 0.0])
# x_max = np.array([Range, Range, 3.0])
# x_init = [-1.8, -1.5]
# x0 = [x_init[0], x_init[1], 0.0]
# dt = 0.03
# pieces1_path = "PAC_PIECE/pieces_a0.2_b0.01_k100000_e0.01_m3_sresidual_range_NPAC97_y0.json"
# pieces2_path = "PAC_PIECE/pieces_a0.2_b0.01_k100000_e0.01_m3_sresidual_range_NPAC97_y1.json"
# pieces1 = load_model(pieces1_path)
# pieces2 = load_model(pieces2_path)

# ts, pac_traj, true_traj = [], [], []
# plt.figure()
# ts.append(x0[2])
# pac_traj.append(x_init)
# true_traj.append(x_init)

# for i in range(101):
#     x0[2] += dt
#     y1, _ = predict(x0, pieces1)
#     y2, _ = predict(x0, pieces2)
#     y = Phi(x_init, x0[2])

#     ts.append(x0[2])
#     pac_traj.append([y1, y2])
#     true_traj.append(y)

#     plt.plot(y1, y2, "o", color="black")
#     plt.plot(y[0], y[1], "x", color="red")
# plt.xlim(-3.0, 3.0)
# plt.ylim(-3.0, 3.0)
# plt.xlabel("x0")
# plt.ylabel("x1")
# plt.legend(["PAC", "True"], loc="upper left")
# plt.savefig("vanderpol.png", dpi=150)

# ts = np.asarray(ts)
# pac_traj = np.asarray(pac_traj)
# true_traj = np.asarray(true_traj)

# fig, axes = plt.subplots(1, 2, figsize=(10, 4))
# axes[0].plot(ts, pac_traj[:, 0], "o", color="black", label="PAC")
# axes[0].plot(ts, true_traj[:, 0], "x", color="red", label="True")
# axes[0].set_xlabel("t")
# axes[0].set_ylabel("x0")
# axes[0].set_xlim(0, 4.0)
# axes[0].set_ylim(-3.0, 3.0)
# axes[0].legend(loc="upper left")

# axes[1].plot(ts, pac_traj[:, 1], "o", color="black", label="PAC")
# axes[1].plot(ts, true_traj[:, 1], "x", color="red", label="True")
# axes[1].set_xlabel("t")
# axes[1].set_ylabel("x1")
# axes[1].set_xlim(0.0, 4.0)
# axes[1].set_ylim(-3.0, 3.0)
# axes[1].legend(loc="upper left")

# fig.tight_layout()
# fig.savefig("vanderpol_xt.png", dpi=150)
# plt.show()

global x_min, x_max
x_min = np.array([-Range, -Range, 0.0])
x_max = np.array([Range, Range, 3.0])

dt = 0.03
x_init = [-1.8, -1.5]

pieces1_path = "PAC_PIECE/pieces_a0.2_b0.01_k200000_e0.05_m3_sresidual_range_NPAC97_y0.json"
pieces2_path = "PAC_PIECE/pieces_a0.2_b0.01_k100000_e0.1_m3_sresidual_range_NPAC97_y1.json"
pieces1 = load_model(pieces1_path)
pieces2 = load_model(pieces2_path)

# Include both endpoints without stepping outside the learned time domain.
t_end = min(
    3.0,
    max(p["bounds"][2][1] for p in pieces1),
    max(p["bounds"][2][1] for p in pieces2),
)

query_times = np.append(np.arange(dt, t_end, dt), t_end)
ts = np.concatenate(([0.0], query_times))
pac_traj = [x_init.copy()]
for t in query_times:
    xs = [*x_init, float(t)]
    y1, _ = predict(xs, pieces1)
    y2, _ = predict(xs, pieces2)
    pac_traj.append([y1, y2])
pac_traj = np.asarray(pac_traj)
true_traj = odeint(ode_model, x_init, ts, rtol=1e-8, atol=1e-8)

TRUE_COLOR = "#326A9F"
PAC_COLOR = "#D97732"

plot_style = {
    "font.size": 12,
    "axes.labelsize": 15,
    "axes.labelpad": 8,
    "axes.edgecolor": "#737B85",
    "axes.labelcolor": "#303741",
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.axisbelow": True,
    "xtick.color": "#505965",
    "ytick.color": "#505965",
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "legend.frameon": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "pdf.fonttype": 42,
}


def style_axes(ax):
    ax.grid(which="major", color="#DCE1E7", linewidth=0.6,
            linestyle="--", alpha=0.8)
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.tick_params(which="major", length=4, width=0.8)
    ax.tick_params(which="minor", length=2, width=0.6)


def plot_comparison(ax, true_x, true_y, pac_x, pac_y):
    ax.plot(true_x, true_y, color=TRUE_COLOR, linewidth=2.1,
            label="True", zorder=2)
    # Keep every prediction visible without connecting jumps between pieces.
    ax.plot(pac_x, pac_y, color=PAC_COLOR, linestyle="none",
            marker="o", markersize=3.4, markerfacecolor="white",
            markeredgewidth=0.9, label="PAC", zorder=3)


with plt.rc_context(plot_style):
    fig_phase, ax = plt.subplots(figsize=(6.0, 5.2),
                                 layout="constrained")
    plot_comparison(ax, true_traj[:, 0], true_traj[:, 1],
                    pac_traj[:, 0], pac_traj[:, 1])
    ax.scatter(*x_init, s=28, color="#303741", zorder=4)
    ax.annotate(r"$t=0$", xy=x_init, xytext=(8, 9),
                textcoords="offset points", fontsize=11,
                color="#303741")
    ax.set_xlabel(r"$x_0$")
    ax.set_ylabel(r"$x_1$")
    ax.set_aspect("equal", adjustable="box")
    ax.margins(x=0.12, y=0.15)
    ax.xaxis.set_major_locator(MultipleLocator(1.0))
    ax.yaxis.set_major_locator(MultipleLocator(1.0))
    style_axes(ax)
    ax.legend(loc="upper left", handlelength=2.8)
    fig_phase.savefig("vanderpol.png", dpi=300, bbox_inches="tight")
    fig_phase.savefig("vanderpol.pdf", bbox_inches="tight")

    fig_time, axes = plt.subplots(1, 2, figsize=(10.2, 4.0),
                                  sharey=True, layout="constrained")
    for j, ax in enumerate(axes):
        plot_comparison(ax, ts, true_traj[:, j], ts, pac_traj[:, j])
        ax.set_xlabel(r"$t$")
        ax.set_ylabel(rf"$x_{j}$")
        ax.set_xlim(ts[0] - 0.04, ts[-1] + 0.04)
        ax.margins(y=0.12)
        ax.xaxis.set_major_locator(MultipleLocator(0.5))
        ax.yaxis.set_major_locator(MultipleLocator(1.0))
        style_axes(ax)
        ax.legend(loc="upper left", handlelength=2.8)
    axes[1].tick_params(labelleft=True)
    fig_time.savefig("vanderpol_xt.png", dpi=300, bbox_inches="tight")
    fig_time.savefig("vanderpol_xt.pdf", bbox_inches="tight")

plt.show()