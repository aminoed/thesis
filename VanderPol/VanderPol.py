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

def plot_vanderpol():
    TRUE_COLOR = "#326A9F"
    PAC_COLOR = "#D97732"
    FIG_SIZE = (6.0, 4.5)

    plot_style = {
        "font.size": 24,
        "axes.labelsize": 28,
        "axes.labelpad": 8,
        "axes.edgecolor": "#202020",
        "axes.labelcolor": "#303741",
        "axes.linewidth": 1.1,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.axisbelow": True,
        "xtick.color": "#202020",
        "ytick.color": "#202020",
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.labelsize": 24,
        "ytick.labelsize": 24,
        "legend.fontsize": 22,
        "legend.frameon": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.bbox": None,
        "pdf.fonttype": 42,
    }

    def style_axes(ax):
        ax.grid(
            which="major",
            color="#DCE1E7",
            linewidth=0.6,
            linestyle="--",
            alpha=0.8,
        )
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        ax.tick_params(which="major", length=4, width=0.8)
        ax.tick_params(which="minor", length=2, width=0.6)

    def plot_comparison(ax, true_x, true_y, pac_x, pac_y):
        ax.plot(
            true_x,
            true_y,
            color=TRUE_COLOR,
            linewidth=3.0,
            label="ODE Solution",
            zorder=2,
        )
        ax.plot(
            pac_x,
            pac_y,
            color=PAC_COLOR,
            linestyle="none",
            marker="o",
            markersize=4.5,
            markerfacecolor="white",
            markeredgewidth=1.2,
            label="PAC Estimation",
            zorder=3,
        )

    with plt.rc_context(plot_style):
        # State components over time
        for j in range(2):
            fig, ax = plt.subplots(
                figsize=FIG_SIZE,
                layout="constrained",
            )

            plot_comparison(
                ax,
                ts, true_traj[:, j],
                ts, pac_traj[:, j],
            )

            ax.set_xlabel(r"$t$")
            ax.set_ylabel(rf"$s_{j + 1}$")
            ax.set_xlim(ts[0] - 0.04, ts[-1] + 0.04)
            ax.margins(y=0.12)

            ax.xaxis.set_major_locator(MultipleLocator(0.5))
            ax.yaxis.set_major_locator(MultipleLocator(1.0))

            style_axes(ax)
            if j==0:
                ax.legend(
                    loc="upper left",
                    handlelength=1.5,
                    handletextpad=0.5,
                    labelspacing=0.3,
                )
            fig.savefig(
                f"vanderpol_s{j + 1}.pdf",
                bbox_inches=None,
            )

        # State trajectory
        fig_phase, ax = plt.subplots(
            figsize=FIG_SIZE,
            layout="constrained",
        )

        plot_comparison(
            ax,
            true_traj[:, 0], true_traj[:, 1],
            pac_traj[:, 0], pac_traj[:, 1],
        )

        ax.scatter(
            *x_init,
            s=28,
            color="#303741",
            zorder=4,
        )
        ax.annotate(
            r"$t=0$",
            xy=x_init,
            xytext=(8, 9),
            textcoords="offset points",
            fontsize=22,
            color="#303741",
        )

        ax.set_xlabel(r"$s_1$")
        ax.set_ylabel(r"$s_2$")
        ax.set_aspect("equal", adjustable="box")
        ax.margins(x=0.12, y=0.15)

        ax.xaxis.set_major_locator(MultipleLocator(1.0))
        ax.yaxis.set_major_locator(MultipleLocator(1.0))

        style_axes(ax)
        
        fig_phase.savefig(
            "vanderpol.pdf",
            bbox_inches=None,
        )

    plt.show()


global x_min, x_max
x_min = np.array([-Range, -Range, 0.0])
x_max = np.array([Range, Range, 3.0])

dt = 0.03
x_init = [-1.8, -1.5]

pieces1_path = "PAC_PIECE/pieces_a0.2_b0.01_k200000_e0.05_m3_sresidual_range_NPAC97_y0.json"
pieces2_path = "PAC_PIECE/pieces_a0.2_b0.01_k100000_e0.1_m3_sresidual_range_NPAC97_y1.json"
pieces1 = load_model(pieces1_path)
pieces2 = load_model(pieces2_path)
eps_star = [0.05, 0.1]  

t_end = min(
    3.0,
    max(p["bounds"][2][1] for p in pieces1),
    max(p["bounds"][2][1] for p in pieces2),
)

query_times = np.append(np.arange(dt, t_end, dt), t_end)
ts = np.concatenate(([0.0], query_times))
true_traj = odeint(ode_model, x_init, ts, rtol=1e-8, atol=1e-8)

pac_traj = [x_init.copy()]
err1, eps1 = [], []
err2, eps2 = [], []
err3, eps3 = [], []

for i, t in enumerate(query_times):
    xs = [x_init[0], x_init[1], float(t)]
    y1, p1 = predict(xs, pieces1)
    y2, p2 = predict(xs, pieces2)
    pac_traj.append([y1, y2])

    err = np.abs([y1, y2] - true_traj[i + 1])
    eps = [p1["epsilon"], p2["epsilon"]]
    if t < 1:
        err1.append(err); eps1.append(eps)
    elif t < 2:
        err2.append(err); eps2.append(eps)
    else:
        err3.append(err); eps3.append(eps)

pac_traj = np.asarray(pac_traj)

rows = []
for name, err_list, eps_list in [
    ("[0,1)", err1, eps1),
    ("[1,2)", err2, eps2),
    ("[2,3]", err3, eps3),
]:
    err = np.asarray(err_list)
    eps = np.asarray(eps_list)
    for j, state in enumerate(["s1", "s2"]):
        rows.append({
            "t_interval": name,
            "state": state,
            "error_mean": err[:, j].mean(),
            "error_max": err[:, j].max(),
            "epsilon_max": eps[:, j].max(),
            "exceed_eps_star_pct": 100.0 * np.mean(eps[:, j] > eps_star[j]),
        })

summary = pd.DataFrame(rows)
print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
summary.to_csv("vanderpol_summary.csv", index=False)
plot_vanderpol()


