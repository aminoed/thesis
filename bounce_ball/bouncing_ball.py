import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import pandas as pd
import json

rng = np.random.default_rng(0)
x_min = np.array([-3.0, -20.0])
x_max = np.array([10.0, 20.0])


def Phi_hybrid(x):
    y = odeint(model, x, step)[-1]
    if y[0] <= 0:
        y[0] = 0.0         
        y[1] = -0.8 * y[1]
    return y

def sample_bounce_ball(n):
    X = rng.uniform(x_min, x_max, size=(n, 2))
    Y = np.array([Phi_hybrid(xi) for xi in X])
    return X, Y
mode = 0

def model(x,t):
    dxdt = [x[1], -9.81]
    return dxdt


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
        phi = np.array([np.prod(xs ** np.asarray(e)) for e in p["exponents"]], dtype=float)
        if w.size != phi.size:
            raise ValueError(f"w size ({w.size}) != feature size ({phi.size})")
        y_pred = float(np.dot(w, phi))
    else:
        if w.size != xs.size + 1:
            raise ValueError(f"w size ({w.size}) != input size + 1 ({xs.size + 1})")
        y_pred = float(w[0] + np.dot(w[1:], xs))
    return y_pred, p


N = 500 # total number of time steps
i = 0

step = np.linspace(0,0.01)
x0 = np.array([10.0, 0.0], dtype=float)
x1 = x0.copy()

# X, Y = sample_bounce_ball(100000)
# pd.DataFrame(np.hstack((X, Y[:,0].reshape(-1,1))), columns=['x0', 'x1', 'y0']).to_csv("bounce_ball_X.csv", index=False)
# pd.DataFrame(np.hstack((X, Y[:,1].reshape(-1,1))), columns=['x0', 'x1', 'y1']).to_csv("bounce_ball_Y.csv", index=False)
pieces1_path = "PAC_PIECE/pieces_a0.05_b1e-08_k100000_e0.1_m3_sresidual_range_NPAC897_X.json"
pieces2_path = "PAC_PIECE/pieces_a0.05_b1e-08_k100000_e0.5_m3_sresidual_range_NPAC897_Y.json"
pieces1 = load_model(pieces1_path)
pieces2 = load_model(pieces2_path)

ts, pac_h, pac_v, true_h, true_v = [], [], [], [], []

while i < N:
    
    y1, _ = predict(x1, pieces1)
    y2, _ = predict(x1, pieces2)
    if y1 is None or y2 is None:
        print(f"[stop] PAC can't predict at step {i}, state={x1}")
        break

    x1 = np.array([y1, y2], dtype=float)
    x0 = Phi_hybrid(x0)

    t = 0.01 * i
    ts.append(t)
    pac_h.append(x1[0]); pac_v.append(x1[1])
    true_h.append(x0[0]); true_v.append(x0[1])
    i += 1

if ts:
    plt.plot(ts, pac_h, 'o-', color='black', markersize=3, label='PAC height')
    plt.plot(ts, true_h, 'x--', color='red', markersize=3, label='True height')
    plt.plot(ts, pac_v, 'o-', color='gray', markersize=2, label='PAC vel')
    plt.plot(ts, true_v, 'x--', color='orange', markersize=2, label='True vel')


plt.xlabel('t')
plt.ylabel('state')
plt.legend(loc='upper left')
plt.savefig("bounce_ball.png", dpi=150)
plt.show()