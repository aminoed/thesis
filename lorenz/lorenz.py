from scipy.integrate import odeint
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json

rng = np.random.default_rng(0)

def Phi(x, dt):
    sol = odeint(ode_model, x, [0.0, dt], rtol=1e-8, atol=1e-8)
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


global x_min, x_max
x_min = np.array([-25.0, -25.0, 0.0, 0.0])
x_max = np.array([25.0, 25.0, 50.0, 3.0])
x_init = [15.0, 15.0, 36.0]
x0 = [x_init[0], x_init[1], x_init[2], 0.0]
dt = 0.03

pieces1_path = "PAC_PIECE/pieces_a0.2_b0.01_k100000_e0.01_m3_residual_range_NPAC97_X.json"
pieces2_path = "PAC_PIECE/pieces_a0.2_b0.01_k100000_e0.01_m3_residual_range_NPAC97_Y.json"
pieces3_path = "PAC_PIECE/pieces_a0.2_b0.01_k100000_e0.01_m3_residual_range_NPAC97_Z.json"

pieces1 = load_model(pieces1_path)
pieces2 = load_model(pieces2_path)  
pieces3 = load_model(pieces3_path)

ts, pac_traj, true_traj = [], [], []
ts.append(x0[3])
pac_traj.append(x_init)
true_traj.append(x_init)

for i in range(101):
    x0[3] += dt
    y1, _ = predict(x0, pieces1)
    y2, _ = predict(x0, pieces2)
    y3, _ = predict(x0, pieces3)

    y = Phi(x0[:3], x0[3])
    ts.append(x0[3])
    pac_traj.append([y1, y2, y3])
    true_traj.append(y)

    plt.plot(y1, y2, 'o', color='black')
    plt.plot(y[0], y[1], 'x', color='red')
plt.xlim(-25.0, 25.0)
plt.ylim(-25.0, 25.0)
plt.xlabel("x0")
plt.ylabel("x1")
# plt.zlabel("x2")
plt.legend(["PAC", "True"], loc="upper left")
plt.savefig("lorenz_10k.png", dpi=150)

ts = np.asarray(ts)
pac_traj = np.asarray(pac_traj)
true_traj = np.asarray(true_traj)

fig, axes = plt.subplots(1, 3, figsize=(10, 4))
axes[0].plot(ts, pac_traj[:, 0], "o", color="black", label="PAC")
axes[0].plot(ts, true_traj[:, 0], "x", color="red", label="True")
axes[0].set_xlabel("t")
axes[0].set_ylabel("x0")
axes[0].set_xlim(0, 4.0)
axes[0].set_ylim(-25.0, 25.0)
axes[0].legend(loc="upper left")

axes[1].plot(ts, pac_traj[:, 1], "o", color="black", label="PAC")
axes[1].plot(ts, true_traj[:, 1], "x", color="red", label="True")
axes[1].set_xlabel("t")
axes[1].set_ylabel("x1")
axes[1].set_xlim(0.0, 4.0)
axes[1].set_ylim(-25.0, 25.0)
axes[1].legend(loc="upper left")

axes[2].plot(ts, pac_traj[:, 2], "o", color="black", label="PAC")
axes[2].plot(ts, true_traj[:, 2], "x", color="red", label="True")
axes[2].set_xlabel("t")
axes[2].set_ylabel("x2")
axes[2].set_xlim(0.0, 4.0)
axes[2].set_ylim(0.0, 50.0)
axes[2].legend(loc="upper left")
fig.tight_layout()
fig.savefig("lorenz_xt.png", dpi=150)
plt.show()
