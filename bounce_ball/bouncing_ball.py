import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import pandas as pd
import json
from matplotlib.ticker import AutoMinorLocator, MultipleLocator

global x_min, x_max
rng = np.random.default_rng(0)
x_min = np.array([-1.0, -10.0, 0.0])
x_max = np.array([10.0, 10.0, 5.0])


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


def ode_model(x, t):
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


def plot_bounce_ball():
    TRUE_COLOR = "#326A9F"
    PAC_COLOR = "#D97732"
    FIG_SIZE = (6.0, 4.5)

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
        "legend.fontsize": 22,
        "legend.frameon": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.bbox": None,
        "pdf.fonttype": 42,
    }

    def style_axes(ax):
        ax.grid(
            which="major",
            color="#DCE1E7",
            linewidth=0.6,
            linestyle="--",
            alpha=0.8,
        )
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        ax.tick_params(which="major", length=4, width=0.8)
        ax.tick_params(which="minor", length=2, width=0.6)

    def plot_comparison(ax, true_x, true_y, pac_x, pac_y):
        ax.plot(
            true_x,
            true_y,
            color=TRUE_COLOR,
            linewidth=3.0,
            label="ODE Solution",
            zorder=2,
        )
        ax.plot(
            pac_x,
            pac_y,
            color=PAC_COLOR,
            linestyle="none",
            marker="o",
            markersize=4.5,
            markerfacecolor="white",
            markeredgewidth=1.2,
            label="PAC Estimation",
            zorder=3,
        )

    
    s1_lim = (-0.5, 11.0)
    s2_lim = (-20.0, 45.0)
    t_lim = (ts[0] - 0.1, ts[-1] + 0.1)
    y_lims = (s1_lim, s2_lim)
    y_ticks = (2.0, 10.0)

    with plt.rc_context(plot_style):
        # State components over time
        for j in range(2):
            fig, ax = plt.subplots(
                figsize=FIG_SIZE,
                layout="constrained",
            )

            plot_comparison(
                ax,
                ts, true_traj[:, j],
                ts, pac_traj[:, j],
            )

            ax.set_xlabel(r"$t$")
            ax.set_ylabel(rf"$s_{j + 1}$")
            ax.set_xlim(*t_lim)
            ax.set_ylim(*y_lims[j])

            ax.xaxis.set_major_locator(MultipleLocator(1.0))
            ax.yaxis.set_major_locator(MultipleLocator(y_ticks[j]))

            style_axes(ax)
            if j == 1:
                ax.legend(
                    loc="lower center",
                    bbox_to_anchor=(0.5, 1.0),
                    ncol=2,
                    handlelength=1,
                    handletextpad=0.5,
                    columnspacing=1.0,
                    labelspacing=0.3,
                )
            fig.savefig(
                f"bounce_ball_s{j + 1}.pdf",
                bbox_inches=None,
            )

    plt.show()


dt = 0.01
x_init = [9.0, -6.9]
eps_star = [0.02, 0.03]  # [height, velocity]


pieces1_path = (
    "PAC_PIECE/pieces_a0.1_b0.01_k200000_e0.05_m3_sresidual_range_NPAC193_y0.json"
)
pieces2_path = (
    "PAC_PIECE/pieces_a0.1_b0.01_k200000_e0.5_m3_sresidual_range_NPAC193_y1.json"
)
pieces1 = load_model(pieces1_path)
pieces2 = load_model(pieces2_path)

t_end = min(
    5.0,
    max(p["bounds"][2][1] for p in pieces1),
    max(p["bounds"][2][1] for p in pieces2),
)

query_times = np.append(np.arange(dt, t_end, dt), t_end)
ts = np.concatenate(([0.0], query_times))
true_traj = np.array([x_init] + [Phi_hybrid(x_init, t) for t in query_times])

pac_traj = [x_init.copy()]
err1, eps1 = [], []
err2, eps2 = [], []
err3, eps3 = [], []

for i, t in enumerate(query_times):
    xs = [x_init[0], x_init[1], float(t)]
    y1, p1 = predict(xs, pieces1)
    y2, p2 = predict(xs, pieces2)
    pac_traj.append([y1, y2])

    err = np.abs([y1, y2] - true_traj[i + 1])
    eps = [p1["epsilon"], p2["epsilon"]]
    if t < 1:
        err1.append(err); eps1.append(eps)
    elif t < 3:
        err2.append(err); eps2.append(eps)
    else:
        err3.append(err); eps3.append(eps)

pac_traj = np.asarray(pac_traj)

rows = []
for name, err_list, eps_list in [
    ("[0,1)", err1, eps1),
    ("[1,3)", err2, eps2),
    ("[3,5]", err3, eps3),
]:
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
print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
summary.to_csv("bounce_ball_summary.csv", index=False)
plot_bounce_ball()


