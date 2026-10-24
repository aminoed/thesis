import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
from scipy.integrate import odeint

rng = np.random.default_rng(0)
x_min = np.array([-0.05, -20.0])
x_max = np.array([9.1, 20.0])
dt = 0.01
t_end = 5.0
x_init = [9.0, -6.9]
eps_star = [0.001, 0.03]  # [height, velocity]
MODEL_PATHS = [
    "PAC_PIECE/"
    f"pieces_a0.1_b0.013_k200000"
    f"_e{0.001 if j == 0 else 0.03}"
    f"_m3_sresidual_range_NPAC167_Y{j}.json"
    for j in range(2)
]


def Phi_hybrid(x):
    y = odeint(model, x, [0, dt])[-1]
    if y[0] <= 0:
        y[0] = 0.0
        y[1] = -0.8 * y[1]
    return y


def sample_bounce_ball(n):
    X = rng.uniform(x_min, x_max, size=(n, 2))
    Y = np.array([Phi_hybrid(xi) for xi in X])
    return X, Y


def model(x, t):
    dxdt = [x[1], -9.81]
    return dxdt


def load_model(model_path):
    # zhuo: read the adaptive subdivision model from json
    with open(model_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    pieces = []
    for p in data:
        exponents = p.get("exponents")
        pieces.append(
            {
                "bounds": [tuple(map(float, b)) for b in p["bounds"]],
                "w": np.asarray(p["w"], dtype=float),
                "degree": int(p.get("degree", 1)),
                "exponents": (
                    [tuple(e) for e in exponents] if exponents is not None else None
                ),
                "epsilon": float(p.get("epsilon", 0.0)),
                "n_sample": int(p.get("n_sample", len(p.get("idx", [])))),
                "idx": np.asarray(p.get("idx", []), dtype=int),
            }
        )
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

    if p is None:
        return None, None

    w = p["w"]

    if p["exponents"] is not None:
        phi = np.array(
            [np.prod(xs ** np.asarray(e)) for e in p["exponents"]], dtype=float
        )
        if w.size != phi.size:
            raise ValueError(f"w size ({w.size}) != feature size ({phi.size})")
        y_pred = float(np.dot(w, phi))
    else:
        if w.size != xs.size + 1:
            raise ValueError(f"w size ({w.size}) != input size + 1 ({xs.size + 1})")
        y_pred = float(w[0] + np.dot(w[1:], xs))
    return y_pred, p


# X, Y = sample_bounce_ball(200000)
# pd.DataFrame(np.hstack((X, Y[:,0].reshape(-1,1))), columns=['x0', 'x1', 'y0']).to_csv("bounce_ball_X_200000.csv", index=False)
# pd.DataFrame(np.hstack((X, Y[:,1].reshape(-1,1))), columns=['x0', 'x1', 'y1']).to_csv("bounce_ball_Y_200000.csv", index=False)

models = [load_model(path) for path in MODEL_PATHS]

n_steps = round(t_end / dt)
x0 = np.asarray(x_init, dtype=float).copy()
x1 = x0.copy()
pac_traj = [x0.copy()]
true_traj = [x1.copy()]
all_err, all_eps, all_t = [], [], []

for i in range(n_steps):
    ys, epss = [], []
    for pieces in models:
        y, p = predict(x0, pieces)
        ys.append(y)
        epss.append(p["epsilon"])
    x0 = np.asarray(ys, dtype=float)
    x1 = Phi_hybrid(x1)
    pac_traj.append(x0.copy())
    true_traj.append(x1.copy())

    t = (i + 1) * dt
    all_t.append(t)
    all_err.append(np.abs(x0 - x1))
    all_eps.append(epss)

ts = dt * np.arange(n_steps + 1)
pac_traj = np.asarray(pac_traj)
true_traj = np.asarray(true_traj)

# Landing-phase boundaries: velocity jumps at bounce impacts (~0.83 s, ~3.27 s)
impact_times = ts[np.flatnonzero(np.diff(true_traj[:, 1]) > 1) + 1]
t_splits = [0.0, *[float(t) for t in impact_times], float(t_end)]
# e.g. [0, 0.83), [0.83, 3.27), [3.27, 5]
interval_names = [
    f"[{t_splits[k]:.2f},{t_splits[k + 1]:.2f}{')' if k < len(t_splits) - 2 else ']'}"
    for k in range(len(t_splits) - 1)
]
pd.DataFrame(
    {
        "phase": [f"landing_{k + 1}" for k in range(len(interval_names))],
        "t_interval": interval_names,
        "t_lo": t_splits[:-1],
        "t_hi": t_splits[1:],
    }
).to_csv("bounce_ball_time_intervals.csv", index=False)

phase_err = [[] for _ in interval_names]
phase_eps = [[] for _ in interval_names]
for t, err, epss in zip(all_t, all_err, all_eps):
    for k in range(len(t_splits) - 1):
        lo, hi = t_splits[k], t_splits[k + 1]
        if (t < hi) if k < len(t_splits) - 2 else (t <= hi):
            if t >= lo:
                phase_err[k].append(err)
                phase_eps[k].append(epss)
            break

rows = []
for name, err_list, eps_list in zip(interval_names, phase_err, phase_eps):
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
print(summary.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
summary.to_csv("bounce_ball_summary.csv", index=False)

state_names = ["s1", "s2"]
state_units = ["m", "m/s"]
abs_error = np.abs(pac_traj - true_traj)
trajectory = {"t": ts}
for j, name in enumerate(state_names):
    trajectory[f"ode_{name}"] = true_traj[:, j]
    trajectory[f"pac_{name}"] = pac_traj[:, j]
    trajectory[f"abs_error_{name}"] = abs_error[:, j]
pd.DataFrame(trajectory).to_csv("bounce_ball_trajectory.csv", index=False)

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

state_ylabels = [rf"$s_{{{j+1}}}$ ({unit})" for j, unit in enumerate(state_units)]
state_filenames = [f"s{j+1}" for j in range(2)]

with plt.rc_context(plot_style):
        for j, ylabel, filename in zip(range(2), state_ylabels, state_filenames):
            fig, ax = plt.subplots(figsize=(6.0, 4.5), layout="constrained")

            ode_ts = ts
            ode_values = true_traj[:, j]
            if j == 1:
                # Draw impact connectors as dashed segments instead of solid lines.
                impacts = np.flatnonzero(np.diff(ode_values) > 1) + 1
                ode_ts = np.insert(ts, impacts, np.nan)
                ode_values = np.insert(ode_values, impacts, np.nan)
                for impact in impacts:
                    ax.plot(
                        ts[impact - 1:impact + 1],
                        true_traj[impact - 1:impact + 1, j],
                        color="#326A9F",
                        linestyle="--",
                        linewidth=2.0,
                        zorder=2,
                    )

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
            ax.xaxis.set_major_locator(MultipleLocator(1.0))
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
            fig.savefig(f"bounce_ball_{filename}.pdf", bbox_inches="tight")
            plt.close(fig)