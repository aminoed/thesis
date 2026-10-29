from scipy.integrate import odeint
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
from matplotlib.ticker import AutoMinorLocator, MultipleLocator

rng = np.random.default_rng(91723)
x_min = np.array([-19, -24, 5])
x_max = np.array([19, 24, 46])
dt = 0.001
T = 7
n_steps = int(round(T / dt))
MODEL_PATHS = [
    "PAC_PIECE/"
    f"pieces_a0.2_b0.01_k100000"
    f"_e{0.0001 if j == 0 else 0.001}"
    f"_m3_sresidual_range_NPAC97_y{j}.json"
    for j in range(3)
]
def Phi(x):
    t = np.linspace(0, dt, 101)
    sol = odeint(ode_model, x, t)
    return sol[-1]

def ode_model(x, t):
    return [10 * (x[1] - x[0]), x[0] * (28 - x[2]) - x[1], x[0] * x[1] - 2.67 * x[2]]
    

def sample_lorenz(n, low=x_min, high=x_max):
   X = rng.uniform(low, high, size=(n, 3))
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
    tol = 1e-5

    d = len(xs)
    models = []
    for p in pieces:
        select = True
        for i in range(d):
            lo, hi = p["bounds"][i]
            if not (lo - tol <= xs[i] <= hi + tol):
                select = False
                break
        if select:
            models.append(p)

    if not models:
        print(f"[stop] no matched piece for xs={xs}")
        return None
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


# X, Y = sample_lorenz(100000)
# pd.DataFrame(np.hstack((X, Y[:,0].reshape(-1,1))), columns=['x0', 'x1', 'x2', 'y0']).to_csv("lorenz_X.csv", index=False)
# pd.DataFrame(np.hstack((X, Y[:,1].reshape(-1,1))), columns=['x0', 'x1', 'x2', 'y1']).to_csv("lorenz_Y.csv", index=False)
# pd.DataFrame(np.hstack((X, Y[:,2].reshape(-1,1))), columns=['x0', 'x1', 'x2', 'y2']).to_csv("lorenz_Z.csv", index=False)

models = [load_model(path) for path in MODEL_PATHS]
print(f"len(models[0]): {len(models[0])}") 
print(f"len(models[1]): {len(models[1])}") 
print(f"len(models[2]): {len(models[2])}") 
x_init = [-4.3, -6.5, 21.5]

x0 = np.asarray(x_init, dtype=float).copy()
x1 = x0.copy()
pac_traj = [x0.copy()]
true_traj = [x1.copy()]

for i in range(n_steps):
    x0 = np.array([predict(x0, pieces)[0] for pieces in models])
    x1 = Phi(x1)
    pac_traj.append(x0.copy())
    true_traj.append(x1.copy())
ts = dt * np.arange(n_steps + 1)
pac_traj = np.asarray(pac_traj)
true_traj = np.asarray(true_traj)

state_names = ["s0", "s1", "s2"]
error = pac_traj - true_traj
abs_error = np.abs(error)
summary = pd.DataFrame(
    {
        "state": np.tile(state_names, len(ts)),
        "error": error.flatten(),
        "abs_error": abs_error.flatten(),
    }
)
summary.to_csv("lorenz_summary.csv", index=False)
# print(summary.to_string(index=False, float_format=lambda v: f"{v:.6g}"))
trajectory = {"t": ts}
for j, name in enumerate(state_names):
    trajectory[f"ode_{name}"] = true_traj[:, j]
    trajectory[f"pac_{name}"] = pac_traj[:, j]
    trajectory[f"abs_error_{name}"] = abs_error[:, j]
pd.DataFrame(trajectory).to_csv("lorenz_trajectory.csv", index=False)

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

state_ylabels = [rf"$x_{{{j+1}}}$" for j in range(3)]
state_filenames = [f"s{j+1}" for j in range(3)]
y_ticks = (5.0, 10.0, 5.0)

with plt.rc_context(plot_style):
        for j, ylabel, filename in zip(range(3), state_ylabels, state_filenames):
            fig, ax = plt.subplots(figsize=(6.0, 4.5), layout="constrained")

            ax.plot(
                ts,
                true_traj[:, j],
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
            ax.xaxis.set_major_locator(MultipleLocator(1.0))
            ax.yaxis.set_major_locator(MultipleLocator(y_ticks[j]))
            ax.xaxis.set_minor_locator(AutoMinorLocator(2))
            ax.yaxis.set_minor_locator(AutoMinorLocator(2))
            ax.tick_params(which="major", length=4, width=0.8)
            ax.tick_params(which="minor", length=2, width=0.6)
            ax.grid(
                which="major", color="#DCE1E7", linewidth=0.6, linestyle="--", alpha=0.8
            )
            # if j == 1:
            #     ax.legend(
            #         loc="lower center",
            #         bbox_to_anchor=(0.5, 1.01),
            #         ncol=2,
            #         handlelength=1.3,
            #         handletextpad=0.5,
            #         columnspacing=1.0,
            #         borderaxespad=0.0,
            #     )
            fig.savefig(f"lorenz_{filename}.pdf", bbox_inches="tight")
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
        ax.set_xlabel(r"$x_{1}$")
        ax.set_ylabel(r"$x_{2}$")
        ax.margins(0.12)
        ax.xaxis.set_major_locator(MultipleLocator(5.0))
        ax.yaxis.set_major_locator(MultipleLocator(10.0))
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        ax.tick_params(which="major", length=4, width=0.8)
        ax.tick_params(which="minor", length=2, width=0.6)
        ax.grid(
            which="major", color="#DCE1E7", linewidth=0.6, linestyle="--", alpha=0.8
        )
        # ax.legend(
        #     loc="lower center",
        #     bbox_to_anchor=(0.5, 1.01),
        #     ncol=2,
        #     handlelength=1.3,
        #     handletextpad=0.5,
        #     columnspacing=1.0,
        #     borderaxespad=0.0,
        # )
        fig.savefig("lorenz.pdf", bbox_inches="tight")
        plt.close(fig)