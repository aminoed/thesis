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
MAX_HEIGHT = 1.4
MIN_HEIGHT = 0.9
GOAL_LO, GOAL_HI = 0.97, 1.03  # goal region for height s3
T_MIN_HEIGHT = 1.0             # after this time, s3 must stay above MIN_HEIGHT

dt = 0.025
rng = np.random.default_rng(0)
CSV_PATH = "height_trajectories.csv"
RECOMPUTE = False  # True: re-run rollouts; False: plot from CSV only


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
    f"_m3_sresidual_range_NPAC1297_y{j}.json"
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

if RECOMPUTE:
    models = [load_model(path) for path in MODEL_PATHS]
    altitude_runs = []
    for run in range(1000):
        initial_state = np.zeros(12)
        initial_state[:6] = rng.uniform(-0.4, 0.4, size=6)

        x0 = np.asarray(initial_state, dtype=float).copy()
        pac_traj = [x0.copy()]
        for i in range(200):
            x0 = np.array([predict(x0, pieces)[0] for pieces in models])
            pac_traj.append(x0.copy())

        pac_traj = np.asarray(pac_traj)
        altitude_runs.append(pac_traj[:, 2])

    ts = dt * np.arange(len(altitude_runs[0]))
    df = pd.DataFrame({"t": ts})
    for i, alt in enumerate(altitude_runs):
        df[f"run_{i}"] = alt
    df.to_csv(CSV_PATH, index=False)
else:
    df = pd.read_csv(CSV_PATH)


# Orange + blue complementary palette
COLOR_BOUND = "#2F5F8A"  # deep blue — safety bounds
COLOR_GOAL = "#6BA3D0"   # soft blue — goal region
COLOR_TRAJ = "#E07A3D"   # orange — PAC rollouts

ts = df["t"].to_numpy()
run_cols = [c for c in df.columns if c.startswith("run_")]

fig, ax = plt.subplots(figsize=(8, 5))
for col in run_cols:
    ax.plot(ts, df[col], color=COLOR_TRAJ, alpha=0.18, linewidth=0.7, zorder=1)

# Goal region: above trajectories so it stays visible
ax.axhspan(
    GOAL_LO, GOAL_HI, color=COLOR_GOAL, alpha=0.40,
    label=r"Goal region $[0.97,\ 1.03]$", zorder=3,
)
# Safety bounds (full-width lines)
ax.axhline(
    MAX_HEIGHT, color=COLOR_BOUND, linestyle="--", linewidth=1.5, zorder=4,
    label=r"$s_3 \leq 1.4$ (all $t$)",
)
ax.axhline(
    MIN_HEIGHT, color=COLOR_BOUND, linestyle="--", linewidth=1.5, zorder=4,
    label=r"$s_3 \geq 0.9$ ($t \geq 1$)",
)

# Red value markers on the left (bounds only)
for y, text in [(MAX_HEIGHT, "1.4"), (MIN_HEIGHT, "0.9")]:
    ax.annotate(
        text, xy=(0.0, y), xycoords=ax.get_yaxis_transform(),
        xytext=(-6, 0), textcoords="offset points",
        ha="right", va="center", fontsize=10, color="red",
        clip_on=False,
    )

ax.set_xlabel(r"$t$ (s)")
ax.set_ylabel(r"Height $s_3$ (m)")
ax.set_xlim(0, 5)
ax.grid(alpha=0.25)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(frameon=False, loc="lower right")
fig.tight_layout()
fig.subplots_adjust(left=0.14) 
fig.savefig("height.pdf", bbox_inches="tight")
plt.show()