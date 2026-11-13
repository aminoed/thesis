from pathlib import Path

import numpy as np
from scipy.optimize import LinearConstraint, linprog, minimize
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def fit_linear(x, y):
    """Minimize max absolute error, with unrestricted affine coefficients."""
    design = np.column_stack((np.ones(len(x)), x))
    matrix = np.vstack((
        np.column_stack((design, -np.ones(len(x)))),
        np.column_stack((-design, -np.ones(len(x)))),
    ))
    result = linprog(
        [0.0, 0.0, 1.0], A_ub=matrix, b_ub=np.r_[y, -y],
        bounds=[(None, None), (None, None), (0.0, None)],
        method="highs",
    )
    if not result.success:
        raise RuntimeError(result.message)
    w = result.x[:2]
    epsilon = float(result.x[2])
    # Three independent positive dual constraints certify a unique optimum.
    support = np.abs(result.ineqlin.marginals) > 1e-8
    unique = np.linalg.matrix_rank(matrix[support]) == 3
    if not unique:
        qp = minimize(
            lambda v: float(v @ v), w, jac=lambda v: 2 * v,
            constraints=[LinearConstraint(design, y - epsilon, y + epsilon)],
            method="SLSQP", options={"ftol": 1e-12, "maxiter": 2000},
        )
        if not qp.success:
            raise RuntimeError("Minimum-norm selection failed: " + qp.message)
        w = qp.x
    assert np.max(np.abs(design @ w - y)) <= epsilon + 1e-8
    return {"w": w.tolist(), "epsilon": epsilon, "status": "fitted"}


def predict(model, x):
    return model["w"][0] + model["w"][1] * np.asarray(x)


def draw_problem_setting(output_dir):
    boundaries = [0, np.pi / 2, 3 * np.pi / 2, 2 * np.pi]
    x = np.linspace(0, 2 * np.pi, 20)
    y = np.sin(x)

    single_fit = fit_linear(x, y)
    region_index = np.digitize(x, boundaries[1:-1])
    local_fits = []
    for index, (low, high) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        in_region = region_index == index
        local_fits.append((low, high, fit_linear(x[in_region], y[in_region])))
    figures = [
        ("example_piecewise.pdf", True),
    ]

    domain = np.array([0, 2 * np.pi])
    axis_left, axis_bottom = -0.75, -1.40
    local_color = "#1f77b4"
    single_color = "#6b737d"
    guide_color = "#a7b0ba"
    axis_color = "#404750"
    axis_arrow = {"arrowstyle": "-|>", "color": axis_color, "linewidth": 0.95}
    label_background = {"facecolor": "white", "edgecolor": "none", "pad": 1}
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "Times"],
        "mathtext.fontset": "stix", "font.size": 10, "pdf.fonttype": 42,
        "axes.spines.top": False, "axes.spines.right": False,
    })
    for filename, show_local_fits in figures:
        fig = plt.figure(figsize=(4.6, 3.15))
        ax = fig.add_axes((0.02, 0.03, 0.96, 0.94))
        ax.set(xlim=(-1.2, 8.9), ylim=(-1.96, 2.02))
        ax.set_axis_off()

        ax.annotate("", (7.40, axis_bottom), (axis_left, axis_bottom), arrowprops=axis_arrow)
        ax.annotate("", (axis_left, 1.46), (axis_left, axis_bottom), arrowprops=axis_arrow)
        ax.text(7.52, axis_bottom - 0.01, "$x$", fontsize=12, va="center")
        ax.text(axis_left - 0.14, 1.48, "$y$", fontsize=12, ha="right")

        ax.vlines(domain, axis_bottom - 0.42, 1.23, color=guide_color, linestyle=":", linewidth=0.8)
        ax.annotate("", (domain[1], axis_bottom - 0.30), (domain[0], axis_bottom - 0.30),
                    arrowprops={"arrowstyle": "<->", "color": axis_color, "linewidth": 0.8})
        ax.text(np.pi, axis_bottom - 0.30, "$X$", ha="center", va="center",
                bbox=label_background)
        ax.text(domain[0], axis_bottom - 0.06, "$0$", ha="center", va="top", bbox=label_background)
        ax.text(domain[1], axis_bottom - 0.06, r"$2\pi$", ha="center", va="top", bbox=label_background)

        ax.plot(domain, predict(single_fit, domain), color=single_color, linewidth=1.55)
        ax.text(domain[1] + 0.28, predict(single_fit, domain[1]),
                "Single linear\napproximation", color=axis_color, va="center", fontsize=9)
        ax.scatter(x, y, s=19, facecolor="white", edgecolor=axis_color, linewidth=0.75, zorder=5)

        if show_local_fits:
            ax.vlines(boundaries[1:-1], axis_bottom, 1.23, color=guide_color, linewidth=0.75)
            ax.text(np.pi, 1.76, "Local linear approximations", color=local_color,
                    ha="center", va="bottom")
            for index, (low, high, model) in enumerate(local_fits):
                endpoints = np.array([low, high])
                ax.plot(endpoints, predict(model, endpoints), color=local_color,
                        linestyle="--", linewidth=1.65)
                label_point = low + 0.40 * (high - low)
                ax.annotate("", (label_point, predict(model, label_point)),
                            (np.pi + (index - 1) * 0.85, 1.69),
                            arrowprops={"arrowstyle": "-", "color": single_color, "linewidth": 0.7})
        path = output_dir / filename
        fig.savefig(path)
        plt.close(fig)
        max_error = single_fit["epsilon"]
        if show_local_fits:
            max_error = max(model["epsilon"] for _, _, model in local_fits)
        print(f"{path} (maximum sample error: {max_error:.3f})")



draw_problem_setting(Path(__file__).parent)
