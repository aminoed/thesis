import json
import time
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MultipleLocator

g = 9.81
R = 0.1
l = 0.5
M_rotor = 0.1
M = 1
m = M + 4*M_rotor

Jx = Jy = 2/5 * M * (R**2) + 2 * M_rotor*(l**2)
Jz = 2/5*M*R**2 + 4*M_rotor*(l**2)

TARGET_HEIGHT = 1.0

dt = 0.025
rng = np.random.default_rng(0)


x_min = np.array([
    -3.0, -3.0, -0.5, -2.0, -2.0, -3.0,
    -0.2, -0.2, -0.4, -0.5, -0.5, -0.5,
])
x_max = np.array([
    3.0, 3.0, 1.6, 2.0, 2.0, 1.0,
    0.2, 0.2, 0.4, 0.5, 0.5, 0.5,
])
x_min[:2] = [-5.0, -5.0]
x_max[:2] = [5.0, 5.0]
MODEL_PATHS = [
    "PAC_PIECE/"
    f"pieces_a0.05_b1e-08_k50000"
    f"_e{0.0001 if j in (7, 8, 11) else 0.001 if j in (0, 9) else 0.01}"
    f"_m3_sresidual_range_NPAC1297_y{j}_new.json"
    for j in range(12)
]
# MODEL_PATHS = [
#     "PAC_PIECE/"
#     f"pieces_a0.05_b1e-08_k200000_e0.01_m3_sresidual_range_NPAC1297_y{j}.json"
#     for j in range(12)
# ]


def Phi(x):
    x = np.asarray(x, dtype=float)
    if x.shape != (12,) or not np.all(np.isfinite(x)):
        raise ValueError("x must contain 12 finite physical states")

    solution = solve_ivp(
        quadrotor_ode,
        t_span=(0.0, dt),
        y0=np.append(x, 0.0),
        t_eval=[dt],
        method="RK45",
        max_step=dt,
        rtol=1e-9,
        atol=1e-11,
    )
    if not solution.success:
        raise RuntimeError(f"Integration failed: {solution.message}")
    next_state = solution.y[:12, -1]
    
    return next_state

def quadrotor_ode(t, state):
    x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, clock = state

    F, T_x, T_y = pd_controller(state)
    dx1 = np.cos(x8)*np.cos(x9)*x4 + (np.sin(x7)*np.sin(x8)*np.cos(x9) - np.cos(x7)*np.sin(x9))*x5 + (np.cos(x7)*np.sin(x8)*np.cos(x9) + np.sin(x7)*np.sin(x9))*x6
    dx2 = np.cos(x8)*np.sin(x9)*x4 + (np.sin(x7)*np.sin(x8)*np.sin(x9) + np.cos(x7)*np.cos(x9))*x5 + (np.cos(x7)*np.sin(x8)*np.sin(x9) - np.sin(x7)*np.cos(x9))*x6
    dx3 = np.sin(x8)*x4 - np.sin(x7)*np.cos(x8)*x5 - np.cos(x7)*np.cos(x8)*x6
    dx4 = x12*x5 - x11*x6 - g*np.sin(x8)    
    dx5 = x10*x6 - x12*x4 + g*np.cos(x8)*np.sin(x7)
    dx6 = x11*x4 - x10*x5 + g*np.cos(x8)*np.cos(x7) - F/m
    
    dx7 = x10 + (np.sin(x7)*(np.sin(x8)/np.cos(x8)))*x11 + (np.cos(x7)*(np.sin(x8)/np.cos(x8)))*x12
    dx8 = np.cos(x7)*x11 - np.sin(x7)*x12
    dx9 = (np.sin(x7)/np.cos(x8))*x11 + (np.cos(x7)/np.cos(x8))*x12
    
    dx10 = (Jy-Jz)/Jx*x11*x12 + T_x/Jx
    dx11 = (Jz-Jx)/Jy*x10*x12 + T_y/Jy
    dx12 = 0
    dt = 1.0

    return [dx1, dx2, dx3, dx4, dx5, dx6, dx7, dx8, dx9, dx10, dx11, dx12, dt]


def pd_controller(state, target_height=TARGET_HEIGHT, target_roll=0.0, target_pitch=0.0):
    x3= state[2]
    x6 = state[5]
    x7, x8 = state[6], state[7]
    x10, x11 = state[9], state[10]

    F = m * g - 10.0 * (x3 - target_height) + 3.0 * x6
    T_x = -(x7 - target_roll) - x10
    T_y = -(x8 - target_pitch) - x11
    return F, T_x, T_y

def sample_quadrotor(n, low=x_min, high=x_max):
    X = rng.uniform(low, high, size=(n, 12))
    Y = np.array([Phi(xi) for xi in X])
    return X, Y


def load_model(model_path):
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
    xs = np.asarray(xs, dtype=float).ravel()
    tol = 1e-12
    models = []
    for p in pieces:
        if len(p["bounds"]) != xs.size:
            raise ValueError("model input dimension does not match state dimension")
        if all(
            lo - tol <= xs[i] <= hi + tol
            for i, (lo, hi) in enumerate(p["bounds"])
        ):
            models.append(p)
    if not models:
        raise ValueError(f"not able to find a matched model for state: {xs}")
    return min(models, key=lambda p: p["epsilon"])


def predict(xs, pieces):
    xs = np.asarray(xs, dtype=float).ravel()
    p = select_piece(xs, pieces)
    w = p["w"]
    if p["exponents"] is not None:
        phi = np.array([
            np.prod(xs ** np.asarray(e)) for e in p["exponents"]
        ], dtype=float)
        if w.size != phi.size:
            raise ValueError(f"w size ({w.size}) != feature size ({phi.size})")
        y_pred = float(np.dot(w, phi))
    else:
        if w.size != xs.size + 1:
            raise ValueError(f"w size ({w.size}) != input size + 1 ({xs.size + 1})")
        y_pred = float(w[0] + np.dot(w[1:], xs))
    if not np.isfinite(y_pred):
        raise ValueError("model returned a non-finite prediction")
    return y_pred, p


# X, Y = sample_quadrotor(50000)
# input_columns = [f"x{i}" for i in range(12)]
# for j in range(12):
#     data = np.hstack((X, Y[:, j].reshape(-1, 1)))
#     filename = f"quadrotor_y{j}_new.csv"
#     pd.DataFrame(data, columns=input_columns + [f"y{j}"]).to_csv(
#         filename, index=False
#     )
models = [load_model(path) for path in MODEL_PATHS]

initial_state = np.zeros(12)
initial_state[:6] = rng.uniform(-0.4, 0.4, size=6)
initial_state[6] = 0.05  
initial_state[7] = -0.04
x0 = np.asarray(initial_state, dtype=float).copy()
x1 = x0.copy()
pac_traj = [x0.copy()]
true_traj = [x1.copy()]
for i in range(200):
    # Compute every component from the same old x0 before replacing it.
    x0 = np.array([predict(x0, pieces)[0] for pieces in models])
    x1 = Phi(x1)
    pac_traj.append(x0.copy())
    true_traj.append(x1.copy())
ts = dt * np.arange(200 + 1)
pac_traj = np.asarray(pac_traj)
true_traj = np.asarray(true_traj)
    # plt.plot(ts, pac_traj[:, 2], "o", color="black", markersize=3, label="PAC")
    # plt.plot(ts, true_traj[:, 2], "-", color="red", label="True")
    # plt.axhline(TARGET_HEIGHT, color="gray", linestyle="--", label="Target: 1 m")
    # plt.xlabel("Time t (s)")
    # plt.ylabel("Height x3 (m)")
    # plt.grid()
    # plt.legend()
    # plt.savefig("height.png", dpi=150)
    # plt.show()
state_names = ["x", "y", "z", "u", "v", "w", "roll", "pitch", "yaw", "p", "q", "r"]
state_units = ["m"] * 3 + ["m/s"] * 3 + ["rad"] * 3 + ["rad/s"] * 3
error = pac_traj - true_traj
abs_error = np.abs(error)
summary = pd.DataFrame({
    "state": state_names,
    "unit": state_units,
    "rmse": np.sqrt(np.mean(error[1:] ** 2, axis=0)),
    "mae": np.mean(abs_error[1:], axis=0),
    "max_abs_error": np.max(abs_error[1:], axis=0),
    "final_abs_error": abs_error[-1],
})
summary.to_csv("quadrotor_error_summary.csv", index=False)
print(summary.to_string(index=False, float_format=lambda v: f"{v:.6g}"))
trajectory = {"t": ts}
for j, name in enumerate(state_names):
    trajectory[f"ode_{name}"] = true_traj[:, j]
    trajectory[f"pac_{name}"] = pac_traj[:, j]
    trajectory[f"abs_error_{name}"] = abs_error[:, j]
pd.DataFrame(trajectory).to_csv("quadrotor_trajectory.csv", index=False)
    # plt.plot(ts, pac_traj[:, 2], "o", color="black", markersize=3, label="PAC")
    # plt.plot(ts, true_traj[:, 2], "-", color="red", label="True")
    # plt.axhline(TARGET_HEIGHT, color="gray", linestyle="--", label="Target: 1 m")
    # plt.xlabel("Time t (s)")
    # plt.ylabel("Height x3 (m)")
    # plt.grid()
    # plt.legend()
    # plt.savefig("height.png", dpi=150)
    # plt.show()
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

state_ylabels = [
    rf"$s_{{{j+1}}}$ ({unit})" for j, unit in enumerate(state_units)
]
state_filenames = [f"x{j+1}" for j in range(12)]

with plt.rc_context(plot_style):
    for j, ylabel, filename in zip(range(12), state_ylabels, state_filenames):
        fig, ax = plt.subplots(figsize=(6.0, 4.5), layout="constrained")

        if j == 2:
            ax.axhline(
                TARGET_HEIGHT,
                color="red",
                linestyle=(0, (5, 4)),
                linewidth=1.8,
                zorder=1,
            )
            ax.annotate(
                "Target: 1 m",
                xy=(0.98, TARGET_HEIGHT),
                xycoords=ax.get_yaxis_transform(),
                xytext=(0, 10),
                textcoords="offset points",
                ha="right", va="bottom",
                fontsize=13, color="#202020",
            )
            ax.yaxis.set_major_locator(MultipleLocator(0.5))

        ax.plot(
            ts, true_traj[:, j],
            color="#326A9F",
            linewidth=3.0,
            label="ODE Solution",
            zorder=2,
        )
        ax.plot(
            ts, pac_traj[:, j],
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
        ax.set_xlim(ts[0] - 0.08, ts[-1] + 0.08)
        ax.margins(y=0.12)
        ax.xaxis.set_major_locator(MultipleLocator(1.0))
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        ax.tick_params(which="major", length=4, width=0.8)
        ax.tick_params(which="minor", length=2, width=0.6)
        ax.grid(
            which="major", color="#DCE1E7", linewidth=0.6,
            linestyle="--", alpha=0.8,
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

        fig.savefig(f"{filename}.png", dpi=300)
        fig.savefig(f"{filename}.pdf")


# Compare one full 12-output PAC step with an ODE step on the SAME states.
# Warm-up and JSON loading are excluded; no changes to the rollout above.
[predict(true_traj[0], pieces)[0] for pieces in models]
Phi(true_traj[0])
pac_ms, ode_ms = [], []
for state in true_traj[:-1]:
    start = time.perf_counter()
    [predict(state, pieces)[0] for pieces in models]
    pac_ms.append(1000.0 * (time.perf_counter() - start))
    start = time.perf_counter()
    Phi(state)
    ode_ms.append(1000.0 * (time.perf_counter() - start))
timing_summary = pd.DataFrame([
    {"method": name, "n_evaluations": len(values),
     "mean_ms": np.mean(values), "median_ms": np.median(values),
     "p95_ms": np.percentile(values, 95)}
    for name, values in [("PAC (12 outputs)", pac_ms), ("ODE", ode_ms)]
])
timing_summary.to_csv("quadrotor_timing_summary.csv", index=False)
pd.DataFrame({"pac_ms": pac_ms, "ode_ms": ode_ms}).to_csv(
    "quadrotor_step_times.csv", index=False)
print(timing_summary.to_string(index=False, float_format=lambda v: f"{v:.6g}"))

plt.show()