from scipy.integrate import odeint
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

rng = np.random.default_rng(0)
x_min = np.array([-3.0, -20.0, 0.0])
x_max = np.array([10.0, 20.0, 5])


def Phi_hybrid(x, dt):
    sol = odeint(ode_model, x, [0.0, dt], rtol=1e-6, atol=1e-6)
    
    y = sol[-1]
    if y[0] <= 0:
        y[0] = 0.0
        y[1] = -0.8 * y[1]

    return y


def sample_bounce_ball(nx):
    X = rng.uniform(x_min, x_max, size=(nx, 3))
    Y = np.array([Phi_hybrid(xi[:2], xi[2]) for xi in X])
    return X, Y


def ode_model(x, t):
    dxdt = [x[1], -9.81]
    return dxdt


n = 200000
X, Y = sample_bounce_ball(n)
pd.DataFrame(
    np.hstack((X, Y[:, 0].reshape(-1, 1))), columns=["x0", "x1", "tau", "y0"]
).to_csv(f"bounce_ball_X_{n}.csv", index=False)

pd.DataFrame(
    np.hstack((X, Y[:, 1].reshape(-1, 1))), columns=["x0", "x1", "tau", "y1"]
).to_csv(f"bounce_ball_Y_{n}.csv", index=False)


plt.scatter(X[:, 0], Y[:, 0], s=1)
plt.xlabel("x")
plt.ylabel("y")
plt.savefig(f"bounce_ball_xy_{n}.png", dpi=150)
plt.show()
