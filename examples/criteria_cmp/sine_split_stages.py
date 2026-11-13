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


def first_split(api, x, y, alpha, beta, strategy, parent_errors):
    X = x[:, None]
    ids = np.arange(x.size)
    min_samples = api["n_pac_for_degree"](alpha, beta, 1, 1)
    _, cut, left, right, cut_index = api["best_split"](
        X, y, ids, min_samples, parent_errors, ids, strategy
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


def draw_parent_fits(ax, parent_pieces, parent_weights, color, grid):
    """Dashed reference: global fit for first split; previous-stage pieces later."""
    if parent_pieces is None:
        ax.plot(grid, parent_weights[0] + parent_weights[1] * grid,
                color=color, linestyle="--", linewidth=1.1, zorder=2)
        return
    for piece in parent_pieces:
        low, high = piece["bounds"][0]
        xx = np.array([low, high])
        w = piece["w"]
        ax.plot(xx, w[0] + w[1] * xx, color=color, linestyle="--", linewidth=1.1, zorder=2)


def draw_comparison(x, y, rows, threshold, parent_weights):
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "Times"],
        "mathtext.fontset": "stix", "font.size": 10, "pdf.fonttype": 42,
        "axes.spines.top": False, "axes.spines.right": False,
    })
    ink, samples, fit, cut, accent = "#242B30", "#242B30", "#23688E", "#68727A", "#733E77"
    parent_color = "#A05B33"
    figures = []
    grid = np.linspace(x[0], x[-1], 450)
    names = ("Residual-range", "Maximum-residual")
    handles = [
        # Line2D([], [], color=ink, label=r"$\varphi(x) = \sin x$"),
        Line2D([], [], color=samples, linestyle="none", marker="o",
               markersize=3.6, label="Samples"),
        Line2D([], [], color=parent_color, linestyle="--", label="Previous fit"),
        Line2D([], [], color=fit, label="Local fits"),
        Line2D([], [], color=cut, linestyle=":", label="Split"),
        # Line2D([], [], color=accent, label="Curve error"),
    ]

    for row, results in enumerate(rows):
        for column, pieces in enumerate(results):
            fig, ax = plt.subplots(figsize=(3.4, 2.1))
            fig.subplots_adjust(left=0.17, right=0.94, bottom=0.27, top=0.95)
            error, x_error, fitted = worst_curve_error(pieces)
            # ax.plot(grid, np.sin(grid), color=ink, linewidth=1.1, zorder=2)
            parent_pieces = None if row == 0 else rows[row - 1][column]
            draw_parent_fits(ax, parent_pieces, parent_weights, parent_color, grid)
            ax.scatter(x, y, color=samples, s=2, linewidth=0, zorder=3)
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
            # fig.text(0.5, 0.055, f"({letter}) {names[column]}, {stage}",
            #          ha="center", fontsize=9.5, fontweight="bold")
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
            ax.set_xlabel(r"$x$")
            if letter == "a":
                ax.legend(handles=handles, loc="lower left", ncol=2,
                          frameon=False, fontsize=7.0, columnspacing=0.7,
                          handlelength=1.5, handletextpad=0.45)
            figures.append((letter, fig))
    # fig.supxlabel(rf"Input $x$   |   Final fits use sample threshold $\epsilon_\mathrm{{min}}={threshold:.2f}$",
    #               y=0.029, fontsize=10)
    return figures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("../../Simulator2/AdaSub_poly.py"))
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    api = load_estimator(args.source)
    x = np.linspace(0.0, 1.5 * np.pi, 121)
    y = np.sin(x)
    alpha, beta, threshold = 0.5, 0.1, 0.2
    strategies = ("residual_range", "maximum_residual")
    parent_weights, _ = api["fit_polynomial_pac"](x[:, None], y, 1)
    parent_errors = api["compute_errors"](x[:, None], y, parent_weights, 1)
    first = [first_split(api, x, y, alpha, beta, strategy, parent_errors)
             for strategy in strategies]
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

    for letter, fig in draw_comparison(x, y, (first, final), threshold, parent_weights):
        fig.savefig(args.output_dir / f"sin_split_{letter}.pdf")
        fig.savefig(args.output_dir / f"sin_split_{letter}.png", dpi=300)
        plt.close(fig)


if __name__ == "__main__":
    main()
