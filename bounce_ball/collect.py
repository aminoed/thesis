import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

rng = np.random.default_rng(0)
x_min = np.array([-1.0, -10.0, 0.0])
x_max = np.array([10.0, 10.0, 5.0])
g = 9.81
alpha = 0.8

def Phi_hybrid(x, dt):
    h = float(x[0])
    v = float(x[1])
    t_left = float(dt)

    if t_left <= 0.0:
        return np.array([h, v], dtype=float)

    while t_left > 0.0:
        if h <= 0.0 and v <= 0.0:
            h = 0.0
            v = -0.8 * v
            if v <= 1e-12:
                return np.array([0.0, 0.0], dtype=float)
        if h == 0.0:
            t_rest = 2.0 * v / (9.81 * (1.0 - 0.8))
            if t_left >= t_rest:
                return np.array([0.0, 0.0], dtype=float)
        
        #  h(t) = h(0) + v(0) t - 0.5 * 9.81 * t^2
        #  v(t) = v(0) - g t
        #  solve for t when h(t) = 0
        args = [0.5 * 9.81, -v, -h]
        t_hit = np.roots(args)
        # t = symbols('t')
        # eq = Eq(h + v * t - 0.5 * 9.81 * t**2, 0)
        # t_hit = solve(eq, t)

        # keep the smallest positive real root
        pos = [float(r.real) for r in t_hit if np.isreal(r) and r.real > 0.0]
        t_hit = min(pos) if pos else None


        # no impact in remaining time 
        if t_hit is None or t_hit > t_left:
            h = h + v * t_left - 0.5 * 9.81 * t_left ** 2
            v = v - 9.81 * t_left
            if h < 0.0:
                h = 0.0
                v = -0.8 * v
            break

        # fly to impact, then bounce
        v = v - 9.81 * t_hit
        h = 0.0
        v = -0.8 * v
        t_left -= t_hit

        if abs(v) <= 1e-12:
            return np.array([0.0, 0.0], dtype=float)

    return np.array([h, v], dtype=float)

def sample_bounce_ball(nx):
    X = rng.uniform(x_min, x_max, size=(nx, 3))
    Y = np.array([Phi_hybrid(xi[:2], xi[2]) for xi in X])
    return X, Y


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


# test the Phi_hybrid function
# x_init = [9.9, 0.0]
# query_times = np.arange(0.0, 5.0, 0.01)
# true_traj = np.array([Phi_hybrid(x_init, t) for t in query_times])

# plt.plot(query_times, true_traj[:, 0], label="x")
# plt.plot(query_times, true_traj[:, 1], label="y")
# plt.legend()
# plt.savefig(f"bounce_ball_traj.png", dpi=150)
# plt.show()