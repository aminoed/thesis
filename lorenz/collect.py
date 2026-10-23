import numpy as np
import pandas as pd
from scipy.integrate import odeint

rng = np.random.default_rng(91724)
x_min = np.array([-19.0, -24.0, 5.0, 0.0])
x_max = np.array([19.0, 24.0, 46.0, 0.5])
n = 200000

def Phi(x, dt):
    sol, info = odeint(ode_model, x, [0.0, dt], rtol=1e-6, atol=1e-6, full_output=1)
    if info["message"] != "Integration successful.":
        return None
    y = sol[-1]
    if not np.all(np.isfinite(y)):
        return None
    return y


def sample_lorenz(n, low=x_min, high=x_max):
    X_good, Y_good = [], []
    while len(X_good) < n:
        xi = rng.uniform(low, high)
        y = Phi(xi[:3], xi[3])
        if y is None:
            raise RuntimeError(f"integration failed, input: {xi.tolist()}")
        X_good.append(xi)
        Y_good.append(y)
    return np.array(X_good), np.array(Y_good)


def ode_model(x, t):
    return [10 * (x[1] - x[0]), x[0] * (28 - x[2]) - x[1], x[0] * x[1] - 2.67 * x[2]]


X, Y = sample_lorenz(n)
pd.DataFrame(
    np.hstack((X, Y[:, 0].reshape(-1, 1))), columns=["x0", "x1", "x2", "tau", "y0"]
).to_csv(f"lorenz_X_{n}.csv", index=False)

pd.DataFrame(
    np.hstack((X, Y[:, 1].reshape(-1, 1))), columns=["x0", "x1", "x2", "tau", "y1"]
).to_csv(f"lorenz_Y_{n}.csv", index=False)

pd.DataFrame(
    np.hstack((X, Y[:, 2].reshape(-1, 1))), columns=["x0", "x1", "x2", "tau", "y2"]
).to_csv(f"lorenz_Z_{n}.csv", index=False)

