import argparse
import ast
import math
import time
from collections import deque
from itertools import combinations_with_replacement
from pathlib import Path

import matplotlib
import numpy as np
from scipy import optimize

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def load_estimator(path):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    definitions = [
        node for node in tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef))
        or (isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "DATA")
    ]
    namespace = dict(np=np, math=math, time=time, deque=deque,
                     combinations_with_replacement=combinations_with_replacement,
                     optimize=optimize)
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


def first_split(api, x, y, alpha, beta, strategy):
    X = x[:, None]
    ids = np.arange(x.size)
    min_samples = api["n_pac_for_degree"](alpha, beta, 1, 1)
    weights, _ = api["fit_polynomial_pac"](X, y, 1)
    errors = api["compute_errors"](X, y, weights, 1)
    _, cut, left, right, cut_index = api["best_split"](
        X, y, ids, min_samples, errors, ids, strategy
    )
    result = []
    for sample_ids, bounds in zip((left[left != cut_index], right),
                                  ((x[0], cut), (cut, x[-1]))):
        w, bound = api["fit_polynomial_pac"](X[sample_ids], y[sample_ids], 1)
        result.append({"bounds": [bounds], "w": w, "epsilon": bound})
    return result


def worst_curve_error(pieces):
    worst = (-1.0, None, None)
    for piece in pieces:
        low, high = piece["bounds"][0]
        intercept, slope = piece["w"]
        points = [low, high]
        if abs(slope) <= 1.0:
            root = np.arccos(slope)
            points.extend(x for x in (root, 2.0 * np.pi - root) if low < x < high)
        x = np.asarray(points)
        pred = intercept + slope * x
        errors = np.abs(np.sin(x) - pred)
        i = int(np.argmax(errors))
        if errors[i] > worst[0]:
            worst = errors[i], x[i], pred[i]
    return worst


def draw_comparison(x, y, rows, threshold):
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix", "font.size": 10, "pdf.fonttype": 42,
        "axes.spines.top": False, "axes.spines.right": False,
    })
    ink, samples, fit, cut, accent = "#242B30", "#78838B", "#23688E", "#68727A", "#733E77"
    fig, axes = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(6.7, 5.0))
    fig.subplots_adjust(left=0.11, right=0.965, top=0.85, bottom=0.125, wspace=0.17, hspace=0.42)
    grid = np.linspace(x[0], x[-1], 450)
    names = ("Residual-range", "Maximum-residual")
    for row, (axes_row, results) in enumerate(zip(axes, rows)):
        for column, (ax, pieces) in enumerate(zip(axes_row, results)):
            error, x_error, fitted = worst_curve_error(pieces)
            ax.plot(grid, np.sin(grid), color=ink, linewidth=1.1, zorder=2)
            ax.scatter(x[::6], y[::6], color=samples, s=10, linewidth=0, zorder=3)
            for i, piece in enumerate(pieces):
                low, high = piece["bounds"][0]
                xx = np.array([low, high])
                w = piece["w"]
                ax.plot(xx, w[0] + w[1] * xx, color=fit, linewidth=1.6, zorder=4)
                if i < len(pieces) - 1:
                    ax.axvline(high, color=cut, linestyle=":", linewidth=0.9, zorder=1)
            # ax.vlines(x_error, fitted, np.sin(x_error), color=accent, linewidth=1.6, zorder=5)
            stage = "one split" if row == 0 else "second split"
            letter = "abcd"[2 * row + column]
            ax.set_title(f"({letter}) {names[column]}, {stage}", y=-0.3 if row == 0 else -0.45,
                         loc="left", fontweight="bold", fontsize=10.2, pad=7)
            # ax.text(0.035, 0.06, f"Curve error = {error:.3f}",
            #         transform=ax.transAxes, va="bottom", fontsize=9, color=accent,
            #         bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.1})
            ax.set_ylim(-1.13, 1.18)
            ax.set_xlim(x[0] - 0.05, x[-1] + 0.05)
            ax.set_yticks([-1, 0, 1])
            ax.set_xticks(np.arange(4) * np.pi / 2, [r"$0$", r"$\pi/2$", r"$\pi$", r"$3\pi/2$"])
            ax.set_axisbelow(True)
            ax.grid(axis="y", color="#E4E8EB", linewidth=0.6)
            ax.set_ylabel(r"$\varphi(x)$")
            ax.set_xlabel(r"Input $x$")
    # fig.supxlabel(rf"Input $x$   |   Final fits use sample threshold $\epsilon_\mathrm{{min}}={threshold:.2f}$",
    #               y=0.029, fontsize=10)
    handles = [
        Line2D([], [], color=ink, label=r"$\varphi(x) = \sin x$"),
        Line2D([], [], color=samples, linestyle="none", marker="o",
               markersize=3.6, label="Samples"),
        Line2D([], [], color=fit, label="Local fits"),
        Line2D([], [], color=cut, linestyle=":", label="Split"),
        # Line2D([], [], color=accent, label="Curve error"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.55, 0.985),
               ncol=5, frameon=False, fontsize=9.4, columnspacing=1.05, handletextpad=0.55)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("../../Simulator2/AdaSub_poly.py"))
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    api = load_estimator(args.source)
    x = np.linspace(0.0, 1.5 * np.pi, 121)
    y = np.sin(x)
    alpha, beta, threshold = 0.5, 0.1, 0.20
    strategies = ("residual_range", "maximum_residual")
    first = [first_split(api, x, y, alpha, beta, strategy) for strategy in strategies]
    final = [
        sorted(api["adaSubDiv"](x[:, None], y, [(x[0], x[-1])], threshold,
                                alpha, beta, 1, strategy),
               key=lambda p: p["bounds"][0][0])
        for strategy in strategies
    ]
    for stage, results in (("One split", first), ("Final", final)):
        for strategy, pieces in zip(strategies, results):
            cuts = [round(float(piece["bounds"][0][1] / np.pi), 4) for piece in pieces[:-1]]
            sample_error = max(piece["epsilon"] for piece in pieces)
            curve_error = worst_curve_error(pieces)[0]
            print(f"{stage}, {strategy}: cuts/pi={cuts}, "
                  f"sample error={sample_error:.6f}, curve error={curve_error:.6f}")

    fig = draw_comparison(x, y, (first, final), threshold)
    fig.savefig(args.output_dir / "sine_split_stages.pdf")
    fig.savefig(args.output_dir / "sine_split_stages.png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
