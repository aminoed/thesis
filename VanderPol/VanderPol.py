from scipy.integrate import odeint
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
Range = 3.0
rng = np.random.default_rng(0)

def Phi(x):
    t = np.linspace(0, 0.05, 101)
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

# X, Y = sample_vanderpol(Range, 10000)
# pd.DataFrame(np.hstack((X, Y[:,0].reshape(-1,1))), columns=['x0', 'x1', 'y0']).to_csv("vanderpol_X.csv", index=False)
# pd.DataFrame(np.hstack((X, Y[:,1].reshape(-1,1))), columns=['x0', 'x1', 'y1']).to_csv("vanderpol_Y.csv", index=False)

global x_min, x_max
x_min = np.array([-3.0, -3.0])
x_max = np.array([3.0, 3.0])
x0 = [1.25, 2.35]
x1 = x0.copy()
pieces1_path = "PAC_PIECE/pieces_a0.05_b1e-08_k10000_e0.01_m3_sresidual_range_NPAC897_y0.json"
pieces2_path = "PAC_PIECE/pieces_a0.05_b1e-08_k10000_e0.01_m3_sresidual_range_NPAC897_y1.json"
pieces1 = load_model(pieces1_path)
pieces2 = load_model(pieces2_path)
for i in range(1500):
    y1, _ = predict(x0, pieces1)
    y2, _ = predict(x0, pieces2)
    x0 = [y1, y2]
    
    y = Phi(x1)
    x1 = y
    plt.plot(x0[0], x0[1], 'o', color='black')
    plt.plot(x1[0], x1[1], 'x', color='red')
plt.legend(['PAC', 'True'], loc='upper left')
plt.savefig("vanderpol.png", dpi=150)
plt.show()
