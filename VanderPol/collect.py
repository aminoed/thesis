from scipy.integrate import odeint
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json

Range = 3.0
rng = np.random.default_rng(0)


def Phi(x, dt):
    sol = odeint(ode_model, x, [0.0, dt], rtol=1e-8, atol=1e-8)
    return sol[-1]


def ode_model(x, t):
    return [x[1], -x[0] + (1 - x[0] ** 2) * x[1]]


def sample_vanderpol(Range, n):
    # X = rng.uniform(-Range, Range, size=(n, 2))
    # tau = 0.05 * rng.integers(1, 101, size=n)
    # Y = np.array([Phi(xi, ti) for xi, ti in zip(X, tau)])
    # X = np.column_stack((X, tau))

    low = np.array([-Range, -Range, 0.0])
    high = np.array([Range, Range, 3.0])
    X = rng.uniform(low, high, size=(n, 3))
    Y = np.array([Phi(xi[:2], xi[2]) for xi in X])
    
    return X, Y


X, Y = sample_vanderpol(Range, 200000)
pd.DataFrame(
    np.hstack((X, Y[:, 0].reshape(-1, 1))), columns=["x0", "x1", "tau", "y0"]
).to_csv("vanderpol_X.csv", index=False)

pd.DataFrame(
    np.hstack((X, Y[:, 1].reshape(-1, 1))), columns=["x0", "x1", "tau", "y1"]
).to_csv("vanderpol_Y.csv", index=False)

plt.scatter(X[:, 0], Y[:, 0], s=1)
plt.xlabel("x0")
plt.ylabel("y0")
plt.savefig("vanderpol_xy.png", dpi=150)
plt.show()
