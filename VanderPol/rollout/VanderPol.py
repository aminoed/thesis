import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import odeint
from matplotlib.ticker import AutoMinorLocator, MultipleLocator

Range = 3.0
rng = np.random.default_rng(0)
dt = 0.05
x_min = np.array([-3.0, -3.0])
x_max = np.array([3.0, 3.0])

MODEL_PATHS = [
    "PAC_PIECE/"
    f"pieces_a0.05_b1e-08_k10000_e0.01_m3_sresidual_range_NPAC897_y{j}.json"
    for j in range(2)
]

def Phi(x):
    t = np.linspace(0, dt, 101)
    sol = odeint(ode_model, x, t)
    return sol[-1]

def ode_model(x, t):
    return [x[1], -x[0] + (1 - x[0]**2) * x[1]]



def sample_vanderpol(Range, n):
   X = rng.uniform(-Range, Range, size=(n, 2))
   Y = np.array([Phi(xi) for xi in X])
   return X, Y


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
    tol = 0.1 * (x_max - x_min)
    tol_vec = np.asarray(tol, dtype=float).ravel()

    d = len(xs)
    models = []
    for p in pieces:
        select = True
        for i in range(d):
            lo, hi = p["bounds"][i]
            tol = tol_vec[i]
            if not (lo - tol <= xs[i] <= hi + tol):
                select = False
                break
        if select:
            models.append(p)

    # if not models:
    #     raise ValueError(f"not able to find a matched model with the given factors: {xs}")
    if not models:
        print("\n=== DEBUG: no matched piece ===")
        print("xs:", xs)

        for p_idx, p in enumerate(pieces[:5]):
            print(f"\n-- checking piece {p_idx} --")
            for i in range(d):
                lo, hi = p["bounds"][i]
                tol = tol_vec[i]
                if not (lo - tol <= xs[i] <= hi + tol):
                    print(f"[FAIL] dim={i}, xs={xs[i]:.6f}, "
                          f"range=[{lo:.6f},{hi:.6f}], tol={tol:.6f}")
                    break

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


def plot_vanderpol():
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

    state_ylabels = [rf"$s_{{{j+1}}}$" for j in range(2)]
    state_filenames = [f"s{j+1}" for j in range(2)]

    with plt.rc_context(plot_style):
            for j, ylabel, filename in zip(range(2), state_ylabels, state_filenames):
                fig, ax = plt.subplots(figsize=(6.0, 4.5), layout="constrained")

                ode_ts = ts
                ode_values = true_traj[:, j]

                ax.plot(
                    ode_ts,
                    ode_values,
                    color="#326A9F",
                    linewidth=3.0,
                    label="ODE Solution",
                    zorder=2,
                )
                ax.plot(
                    ts,
                    pac_traj[:, j],
                    color="#D97732",
                    linestyle="none",
                    marker="o",
                    markersize=1.5,
                    markerfacecolor="white",
                    markeredgewidth=1.2,
                    label="PAC Estimation",
                    zorder=3,
                )
                ax.set_xlabel(r"$t$ (s)")
                ax.set_ylabel(ylabel)
                ax.set_xlim(0, ts[-1])
                ax.margins(y=0.12)
                ax.xaxis.set_major_locator(MultipleLocator(5.0))
                ax.yaxis.set_major_locator(MultipleLocator(1.5))
                ax.xaxis.set_minor_locator(AutoMinorLocator(2))
                ax.yaxis.set_minor_locator(AutoMinorLocator(2))
                ax.tick_params(which="major", length=4, width=0.8)
                ax.tick_params(which="minor", length=2, width=0.6)
                ax.grid(
                    which="major", color="#DCE1E7", linewidth=0.6, linestyle="--", alpha=0.8
                )
                if j==1:
                    ax.legend(
                        loc="lower center",
                        bbox_to_anchor=(0.5, 1.01),
                        ncol=2,
                        handlelength=1.3,
                        handletextpad=0.5,
                        columnspacing=1.0,
                        borderaxespad=0.0,
                    )
                fig.savefig(f"vanderpol_{filename}.pdf", bbox_inches="tight")
                fig.savefig(f"vanderpol_{filename}.png", dpi=180, bbox_inches="tight")
                plt.close(fig)

            # s1 vs s2
            fig, ax = plt.subplots(figsize=(6.0, 4.5), layout="constrained")
            ax.plot(
                true_traj[:, 0],
                true_traj[:, 1],
                color="#326A9F",
                linewidth=3.0,
                label="ODE Solution",
                zorder=2,
            )
            ax.plot(
                pac_traj[:, 0],
                pac_traj[:, 1],
                color="#D97732",
                linestyle="none",
                marker="o",
                markersize=1.5,
                markerfacecolor="white",
                markeredgewidth=1.2,
                label="PAC Estimation",
                zorder=3,
            )
            ax.plot(
                true_traj[0, 0],
                true_traj[0, 1],
                color="#404040",
                linestyle="none",
                marker="o",
                markersize=8,
                zorder=4,
            )
            ax.annotate(
                r"$t = 0$",
                xy=(true_traj[0, 0], true_traj[0, 1]),
                xytext=(-40, -20),
                textcoords="offset points",
                fontsize=20,
                color="#303741",
            )
            ax.set_xlabel(r"$s_{1}$")
            ax.set_ylabel(r"$s_{2}$")
            ax.margins(0.12)
            ax.xaxis.set_major_locator(MultipleLocator(1.0))
            ax.yaxis.set_major_locator(MultipleLocator(1.0))
            ax.xaxis.set_minor_locator(AutoMinorLocator(2))
            ax.yaxis.set_minor_locator(AutoMinorLocator(2))
            ax.tick_params(which="major", length=4, width=0.8)
            ax.tick_params(which="minor", length=2, width=0.6)
            ax.grid(
                which="major", color="#DCE1E7", linewidth=0.6, linestyle="--", alpha=0.8
            )
            fig.savefig("vanderpol_phase.pdf", bbox_inches="tight")
            fig.savefig("vanderpol_phase.png", dpi=180, bbox_inches="tight")
            plt.close(fig)


# X, Y = sample_vanderpol(Range, 10000)
# pd.DataFrame(np.hstack((X, Y[:,0].reshape(-1,1))), columns=['x0', 'x1', 'y0']).to_csv("vanderpol_X.csv", index=False)
# pd.DataFrame(np.hstack((X, Y[:,1].reshape(-1,1))), columns=['x0', 'x1', 'y1']).to_csv("vanderpol_Y.csv", index=False)
models = [load_model(path) for path in MODEL_PATHS]

x_init = [-1.8,-1.5]
x0 = np.asarray(x_init, dtype=float).copy()
x1 = x0.copy()
pac_traj = [x0.copy()]
true_traj = [x1.copy()]

for i in range(500):
    x0 = np.array([predict(x0, pieces)[0] for pieces in models])
    x1 = Phi(x1)
    pac_traj.append(x0.copy())
    true_traj.append(x1.copy())
ts = dt * np.arange(500 + 1)
pac_traj = np.asarray(pac_traj)
true_traj = np.asarray(true_traj)

state_names = ["s0", "s1"]
error = pac_traj - true_traj
abs_error = np.abs(error)
summary = pd.DataFrame(
    {
        "state": np.tile(state_names, len(ts)),
        "error": error.flatten(),
        "abs_error": abs_error.flatten(),
    }
)
summary.to_csv("vanderpol_summary.csv", index=False)
# print(summary.to_string(index=False, float_format=lambda v: f"{v:.6g}"))
trajectory = {"t": ts}
for j, name in enumerate(state_names):
    trajectory[f"ode_{name}"] = true_traj[:, j]
    trajectory[f"pac_{name}"] = pac_traj[:, j]
    trajectory[f"abs_error_{name}"] = abs_error[:, j]
pd.DataFrame(trajectory).to_csv("vanderpol_trajectory.csv", index=False)

plot_vanderpol()