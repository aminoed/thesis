from scipy.integrate import odeint
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MultipleLocator

import numpy as np
import pandas as pd
import json

rng = np.random.default_rng(0)

def Phi(x, dt):
    sol = odeint(ode_model, x, [0.0, dt], rtol=1e-6, atol=1e-6)
    return sol[-1]

def ode_model(x, t):
    return [10 * (x[1] - x[0]), x[0] * (28 - x[2]) - x[1], x[0] * x[1] - 2.67 * x[2]]
    

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


def plot_lorenz():
    TRUE_COLOR = "#326A9F"
    PAC_COLOR = "#D97732"
    FIG_SIZE = (6.0, 4.5)
    y_ticks = (5.0, 10.0, 10.0)

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
        "legend.fontsize": 17,
        "legend.frameon": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "pdf.fonttype": 42,
    }

    with plt.rc_context(plot_style):
        for j in range(3):
            fig, ax = plt.subplots(figsize=FIG_SIZE, layout="constrained")

            ax.plot(
                ts,
                true_traj[:, j],
                color=TRUE_COLOR,
                linewidth=3.0,
                label="ODE Solution",
                zorder=2,
            )
            ax.plot(
                ts,
                pac_traj[:, j],
                color=PAC_COLOR,
                linestyle="none",
                marker="o",
                markersize=1.5,
                markerfacecolor="white",
                markeredgewidth=1.2,
                label="PAC Estimation",
                zorder=3,
            )

            ax.set_xlabel(r"$t$ (s)")
            ax.set_ylabel(rf"$s_{{{j + 1}}}$")
            ax.set_xlim(0, ts[-1])
            ax.margins(y=0.12)
            ax.xaxis.set_major_locator(MultipleLocator(0.1))
            ax.yaxis.set_major_locator(MultipleLocator(y_ticks[j]))
            ax.xaxis.set_minor_locator(AutoMinorLocator(2))
            ax.yaxis.set_minor_locator(AutoMinorLocator(2))
            ax.tick_params(which="major", length=4, width=0.8)
            ax.tick_params(which="minor", length=2, width=0.6)
            ax.grid(
                which="major",
                color="#DCE1E7",
                linewidth=0.6,
                linestyle="--",
                alpha=0.8,
            )
            if j == 0:
                ax.legend(
                    loc="lower center",
                    bbox_to_anchor=(0.5, 1.01),
                    ncol=2,
                    handlelength=1.3,
                    handletextpad=0.5,
                    columnspacing=1.0,
                    borderaxespad=0.0,
                )
            fig.savefig(f"lorenz_s{j + 1}.pdf", bbox_inches="tight")
            plt.close(fig)


global x_min, x_max
x_min = np.array([-19.0, -24.0, 5.0, 0.0])
x_max = np.array([19.0, 24.0, 46.0, 0.5])
x_init = [-4.3, -6.5, 21.5]
x0 = [*x_init, 0.0]
dt = 0.001

pieces1_path = "PAC_PIECE/pieces_a0.2_b0.01_k200000_e0.1_m3_residual_range_NPAC107_X.json"
pieces2_path = "PAC_PIECE/pieces_a0.2_b0.01_k200000_e0.1_m3_residual_range_NPAC107_Y.json"
pieces3_path = "PAC_PIECE/pieces_a0.2_b0.01_k200000_e0.1_m3_residual_range_NPAC107_Z.json"

pieces1 = load_model(pieces1_path)
pieces2 = load_model(pieces2_path)  
pieces3 = load_model(pieces3_path)

t_end = min(
    0.5,
    max(p["bounds"][3][1] for p in pieces1),
    max(p["bounds"][3][1] for p in pieces2),
    max(p["bounds"][3][1] for p in pieces3),
)

query_times = np.append(np.arange(dt, t_end, dt), t_end)
ts = np.concatenate(([0.0], query_times))
true_traj = odeint(ode_model, x_init, ts, rtol=1e-8, atol=1e-8)

pac_traj = [x_init.copy()]
err1, eps1 = [], []
err2, eps2 = [], []
err3, eps3 = [], []
for i, t in enumerate(query_times):
    xs = [x_init[0], x_init[1], x_init[2], float(t)]
    y1, p1 = predict(xs, pieces1)
    y2, p2 = predict(xs, pieces2)
    y3, p3 = predict(xs, pieces3)
    pac_traj.append([y1, y2, y3])

    err = np.abs([y1, y2, y3] - true_traj[i + 1])
    eps = [p1["epsilon"], p2["epsilon"], p3["epsilon"]]
    x0[3] += dt
    if t < 0.1:
        err1.append(err); eps1.append(eps)
    elif t < 0.3:
        err2.append(err); eps2.append(eps)
    else:
        err3.append(err); eps3.append(eps)

pac_traj = np.asarray(pac_traj)

rows = []
for name, err_list, eps_list in [
    ("[0,0.1)", err1, eps1),
    ("[0.1,0.3)", err2, eps2),
    ("[0.3,0.5]", err3, eps3),
]:
    err = np.asarray(err_list)
    eps = np.asarray(eps_list)
    for j, state in enumerate(["s1", "s2", "s3"]):
        rows.append({
            "t_interval": name,
            "state": state,
            "error_mean": err[:, j].mean(),
            "error_max": err[:, j].max(),
            "epsilon_max": eps[:, j].max(),
            "exceed_eps_star_pct": 100.0 * np.mean(eps[:, j] > 0.1),
        })

summary = pd.DataFrame(rows)
print(summary.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
summary.to_csv("lorenz_summary.csv", index=False)
plot_lorenz()