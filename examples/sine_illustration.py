"""Make two connected sine examples for a thesis.

Run: python sine_illustration.py --output-dir sine_figures
For the two separate Problem Setting figures:
python sine_illustration.py --problem-setting --output-dir Master_Thesis_self/figures
Dependencies: numpy, scipy, matplotlib.
Figure 1 illustrates one current box and its two local models using samples.
Figure 2 compares this subdivision with a simple maximum-error rule.
No true-function curve is drawn. A subset of actual samples is displayed.
The baseline chooses the leftmost tied maximum and cuts immediately after it.
All reported error bounds refer to sampled constraints, not proved adaptive PAC
guarantees. Both methods use the same data, LP, and sample-count rule.
"""

import argparse
import json
from collections import deque
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


def choose_split(x, r, minimum, rule, tie_tol):
    """Return an index after which to cut; x is strictly increasing."""
    counts = np.arange(1, len(x))
    valid = ((np.maximum(counts, len(x) - counts) >= minimum)
             & (x[:-1] < x[1:]))
    if not np.any(valid):
        return None
    if rule == "range":
        left_range = np.maximum.accumulate(r) - np.minimum.accumulate(r)
        right_range = (
            np.maximum.accumulate(r[::-1]) - np.minimum.accumulate(r[::-1])
        )[::-1]
        scores = np.maximum(left_range[:-1], right_range[1:])
        optimum = scores[valid].min()
        candidates = np.flatnonzero(valid & (np.abs(scores - optimum) <= tie_tol))
        index = min(candidates, key=lambda j: (abs(2 * (j + 1) - len(x)), j))
    elif rule == "maxpoint":
        # A fixed, transparent baseline. Ignore roundoff within tied LP errors.
        tied = np.flatnonzero(np.abs(r - r.max()) <= tie_tol)
        candidates = [int(j) for j in tied if j < len(x) - 1 and valid[j]]
        if not candidates:
            return None
        index = candidates[0]
    else:
        raise ValueError(rule)
    threshold = float(x[index] + (x[index + 1] - x[index]) / 2)
    return int(index), threshold


def split_children(x, y, low, high, parent, split, minimum):
    index, threshold = split
    children = []
    for a, b, sl in [(low, threshold, slice(0, index + 1)),
                     (threshold, high, slice(index + 1, len(x)))]:
        xc, yc = x[sl], y[sl]
        if len(xc) >= minimum:
            model = fit_linear(xc, yc)
            assert model["epsilon"] <= parent["epsilon"] + 1e-8
        else:
            model = dict(parent, status="inherited")
        children.append(dict(model, low=float(a), high=float(b), count=len(xc)))
    assert sum(child["count"] for child in children) == len(x)
    return children


def adaptive_models(x, y, domain, minimum, target, tie_tol):
    queue = deque([(domain[0], domain[1], x, y)])
    leaves = []
    while queue:
        low, high, xc, yc = queue.popleft()
        model = fit_linear(xc, yc)
        if model["epsilon"] <= target:
            leaves.append(dict(model, low=low, high=high, count=len(xc)))
            continue
        errors = np.abs(predict(model, xc) - yc)
        split = choose_split(xc, errors, minimum, "range", tie_tol)
        if split is None:
            leaves.append(dict(model, low=low, high=high, count=len(xc),
                               status="no_admissible_split"))
            continue
        index, threshold = split
        for child, sl in zip(
            split_children(xc, yc, low, high, model, split, minimum),
            [slice(0, index + 1), slice(index + 1, len(xc))],
        ):
            if child["status"] == "inherited":
                leaves.append(child)
            else:
                queue.append((child["low"], child["high"], xc[sl], yc[sl]))
    return sorted(leaves, key=lambda leaf: leaf["low"])


def draw_schematic(ax, parent, children, x, y, domain, threshold,
                   baseline=False):
    """Conference-style axes, sample dots and affine lines; no sin curve."""
    low, high = domain
    span = high - low
    x_axis = low - 0.16 * span
    bottom = -1.30
    blue = "#2026A5"
    # Only display a sparse subset; every displayed dot is a training sample.
    shown = np.unique(np.linspace(0, len(x) - 1, min(18, len(x)), dtype=int))
    ax.scatter(x[shown], y[shown], color="#A53625", s=22, zorder=5)
    grid = np.array([low, high])
    ax.plot(grid, predict(parent, grid), color=blue, linewidth=1.6, zorder=3)
    for child in children:
        local = np.array([child["low"], child["high"]])
        ax.plot(local, predict(child, local), color=blue,
                linestyle=(0, (2.5, 2)), linewidth=1.5, zorder=4)
    for boundary in domain:
        ax.plot([boundary, boundary], [bottom - 0.25, 1.18],
                color="black", linestyle=(0, (2, 2)), linewidth=0.85)
    ax.plot([threshold, threshold], [bottom, 1.18], color="#E53935", linewidth=1)
    ax.annotate("", xy=(high + 0.22 * span, bottom), xytext=(x_axis, bottom),
                arrowprops={"arrowstyle": "-|>", "color": "black", "lw": 1.2})
    ax.annotate("", xy=(x_axis, 1.40), xytext=(x_axis, bottom),
                arrowprops={"arrowstyle": "-|>", "color": "black", "lw": 1.2})
    ax.text(high + 0.23 * span, bottom - 0.03, "$x$", fontsize=13, va="top")
    ax.text(x_axis - 0.025 * span, 1.42, "$y$", fontsize=13, ha="right")
    # Domain bracket with a white center for the X' label.
    ax.annotate("", xy=(high, bottom - 0.16), xytext=(low, bottom - 0.16),
                arrowprops={"arrowstyle": "<->", "color": "black", "lw": 0.9})
    ax.text((low + high) / 2, bottom - 0.16, "$X'$", ha="center", va="center",
            fontsize=13, bbox={"facecolor": "white", "edgecolor": "none", "pad": 3})
    ax.text(high + 0.07 * span, -0.18,
            "Parent linear PAC\napproximation\n$p(\\mathbf{x},\\mathbf{w}^{*})$",
            fontsize=10, ha="left", va="center")
    if baseline:
        ax.annotate("Cut after a maximum-error sample\n(one sample in the left child)",
                    xy=(threshold, 1.16), xytext=(low + 0.47 * span, 1.61),
                    fontsize=10, ha="center", va="bottom",
                    arrowprops={"arrowstyle": "-", "lw": 0.9, "color": "black"})
    else:
        label_x = low + 0.48 * span
        ax.text(label_x, 1.59, "Better linear PAC approximations\nin each subdivision",
                fontsize=10, ha="center", va="bottom")
        for k, child in enumerate(children):
            target_x = child["low"] + 0.54 * (child["high"] - child["low"])
            ax.annotate("", xy=(target_x, float(predict(child, target_x))),
                        xytext=(label_x + (-0.10 if k == 0 else 0.10) * span, 1.54),
                        arrowprops={"arrowstyle": "-", "color": "black", "lw": 0.85})
    ax.set_xlim(low - 0.23 * span, high + 0.43 * span)
    ax.set_ylim(bottom - 0.38, 1.94)
    ax.set_axis_off()


def draw_problem_setting(output_dir):
    """Compare a single linear fit with regional fits on one sine period."""
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
        ("problem_setting_single_fit.pdf", False),
        ("problem_setting_piecewise_fit.pdf", True),
    ]

    domain = np.array([0, 2 * np.pi])
    axis_left, axis_bottom = -0.75, -1.40
    local_color = "#1f77b4"
    single_color = "#6b737d"
    guide_color = "#a7b0ba"
    axis_color = "#404750"
    axis_arrow = {"arrowstyle": "-|>", "color": axis_color, "linewidth": 0.95}
    label_background = {"facecolor": "white", "edgecolor": "none", "pad": 1}
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "pdf.fonttype": 42})
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent)
    parser.add_argument("--problem-setting", action="store_true",
                        help="Draw the single and piecewise fit example as separate PDFs.")
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--beta", type=float, default=0.01)
    parser.add_argument("--target", type=float, default=0.15)
    args = parser.parse_args()
    if args.problem_setting:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        draw_problem_setting(args.output_dir)
        return
    if not (0 < args.alpha < 1 and 0 < args.beta < 1 and args.target > 0):
        parser.error("alpha and beta must be in (0,1), and target must be positive")
    minimum = int(np.ceil(2 / args.alpha * (np.log(1 / args.beta) + 3)))
    if args.samples < minimum:
        parser.error(f"Collect at least {minimum} initial samples")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    x = np.sort(np.random.default_rng(args.seed).uniform(0, 2 * np.pi, args.samples))
    y = np.sin(x)
    domain = (0.0, float(2 * np.pi))
    tie_tol = 1e-10
    global_model = fit_linear(x, y)
    leaves = adaptive_models(x, y, domain, minimum, args.target, tie_tol)
    assert sum(leaf["count"] for leaf in leaves) == len(x)
    # Use a box from the actual adaptive construction, rather than choose
    # a different dataset for the conference-style two-child illustration.
    full_errors = np.abs(predict(global_model, x) - y)
    first_split = choose_split(x, full_errors, minimum, "range", tie_tol)
    illustration_domain = domain
    if first_split is not None:
        first_children = split_children(x, y, *domain, global_model, first_split, minimum)
        for child in first_children:
            if child["status"] == "fitted" and child["epsilon"] > args.target:
                illustration_domain = (child["low"], child["high"])
                break
    inside = (x >= illustration_domain[0]) & (x <= illustration_domain[1])
    box_x, box_y = x[inside], y[inside]
    root = fit_linear(box_x, box_y)
    errors = np.abs(predict(root, box_x) - box_y)
    comparisons = {}
    for rule in ("maxpoint", "range"):
        split = choose_split(box_x, errors, minimum, rule, tie_tol)
        if split is None:
            raise RuntimeError(f"No admissible first split for {rule}")
        children = split_children(box_x, box_y, *illustration_domain, root, split, minimum)
        predictions = np.r_[predict(children[0], box_x[:split[0] + 1]),
                            predict(children[1], box_x[split[0] + 1:])]
        comparisons[rule] = {
            "index": split[0], "threshold": split[1], "children": children,
            "sample_mean_absolute_error": float(np.mean(np.abs(predictions - box_y))),
            "sample_max_absolute_error": float(np.max(np.abs(predictions - box_y))),
        }
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    fig1, ax = plt.subplots(figsize=(5.5, 4.0), layout="constrained")
    selected = comparisons["range"]
    draw_schematic(ax, root, selected["children"], box_x, box_y,
                   illustration_domain, selected["threshold"])
    fig2, axes = plt.subplots(1, 2, figsize=(10.4, 4.0), layout="constrained")
    for ax, rule, title in zip(axes, ("maxpoint", "range"),
                              ("(a) Maximum-error-point split", "(b) Impurity-based split")):
        comparison = comparisons[rule]
        draw_schematic(ax, root, comparison["children"], box_x, box_y,
                       illustration_domain, comparison["threshold"], baseline=(rule == "maxpoint"))
        ax.set_title(title, fontsize=11, pad=8)
    for fig, name in [(fig1, "sine_piecewise"), (fig2, "sine_split_comparison")]:
        for extension in ("png", "pdf"):
            path = args.output_dir / f"{name}.{extension}"
            fig.savefig(path, dpi=240, bbox_inches="tight")
            print(path)
        plt.close(fig)
    result = {
        "settings": {"N": args.samples, "seed": args.seed, "alpha": args.alpha,
                     "beta": args.beta, "epsilon_min": args.target,
                     "N_alpha_beta": minimum, "tie_tolerance": tie_tol},
        "global": global_model, "piecewise": leaves,
        "illustration_box": {"domain": illustration_domain, "count": len(box_x), "parent": root},
        "split_comparison": comparisons,
        "baseline": "leftmost tied maximum; cut at following sample midpoint",
    }
    result_path = args.output_dir / "sine_illustration_results.json"
    result_text = json.dumps(result, indent=2)
    if not result_path.exists() or result_path.read_text(encoding="utf-8") != result_text:
        result_path.write_text(result_text, encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
