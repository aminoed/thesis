"""Plot PAC piecewise-linear fits (m_max=1) as two separate figures."""

import csv
import json
import numpy as np
import matplotlib.pyplot as plt

# y0 = uniform, y1 = gaussian
TAG = "a0.1_b0.01_k2000_e0.01_m1_sresidual_range_NPAC153_y{y}"

COLOR_TRUE = "#9A9A9A"
COLOR_UNIFORM = "#1F77B4"
COLOR_GAUSSIAN = "#D62728"


def load_pieces(y):
    with open(f"PAC_PIECE/pieces_{TAG.format(y=y)}.json") as f:
        return json.load(f)


def load_perf(y):
    path = f"PAC_PERF/pac_build_{TAG.format(y=y)}.json"
    with open(path) as f:
        return json.load(f)


def summarize(name, pieces, perf):
    loos = [p["bounds"][0][0] for p in pieces]
    his = [p["bounds"][0][1] for p in pieces]
    eps = [p["epsilon"] for p in pieces]
    errs = [p["max_error"] for p in pieces]
    return {
        "distribution": name,
        "n_pieces": len(pieces),
        "n_splits": perf.get("n_splits"),
        "coverage_low": min(loos),
        "coverage_high": max(his),
        "epsilon_low": min(eps),
        "epsilon_high": max(eps),
        "max_error": max(errs),
        "build_time_s": perf.get("t_total"),
    }


def save_summary(rows, out_path="compare_summary.csv"):
    fields = list(rows[0].keys())
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved {out_path}")
    # also print a quick view
    print()
    print(f"{'distribution':<12} {'pieces':>6} {'eps_low':>10} {'eps_high':>10} {'max_err':>10} {'cover':>18}")
    for r in rows:
        cover = f"[{r['coverage_low']:.3f}, {r['coverage_high']:.3f}]"
        print(
            f"{r['distribution']:<12} {r['n_pieces']:>6} "
            f"{r['epsilon_low']:>10.4f} {r['epsilon_high']:>10.4f} "
            f"{r['max_error']:>10.4f} {cover:>18}"
        )


def shade_pieces(ax, pieces, color):
    bounds = set()
    for piece in pieces:
        lo, hi = piece["bounds"][0]
        ax.axvspan(lo, hi, color=color, alpha=0.12, zorder=0, linewidth=0)
        bounds.add(lo)
        bounds.add(hi)
    for b in sorted(bounds):
        ax.axvline(b, color=color, ls="-", lw=0.8, alpha=0.35, zorder=0)


def plot_pieces(ax, pieces, color, label):
    first = True
    for piece in pieces:
        lo, hi = piece["bounds"][0]
        w = piece["w"]
        xs = np.array([lo, hi])
        ys = np.array([sum(w[k] * (x ** k) for k in range(len(w))) for x in xs])
        ax.plot(
            xs, ys,
            color=color,
            lw=2.0,
            solid_capstyle="round",
            marker="o",
            ms=5.5,
            markerfacecolor=color,
            markeredgecolor="white",
            markeredgewidth=1.0,
            label=label if first else None,
            zorder=3,
        )
        first = False


def style_axes(ax):
    ax.set_xlim(-1.02, 1.02)
    ax.set_ylim(-0.05, 1.08)
    ax.set_xlabel(r"$x$", fontsize=12)
    ax.set_ylabel(r"$y$", fontsize=12)
    ax.tick_params(labelsize=10, length=3.5, width=0.8)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#666666")
        ax.spines[spine].set_linewidth(0.9)


def make_figure(pieces, color, model_label, out_path):
    plt.rcParams.update({"font.family": "sans-serif"})
    x_true = np.linspace(-1, 1, 500)
    y_true = x_true ** 2

    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=150)
    shade_pieces(ax, pieces, color)
    ax.plot(
        x_true, y_true,
        color=COLOR_TRUE, lw=3.0, ls="-",
        label=r"True function $\varphi(x)=x^2$",
        zorder=2,
    )
    plot_pieces(ax, pieces, color=color, label=model_label)
    style_axes(ax)
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(0.05, 1.0),
        ncol=1,
        frameon=True,
        fancybox=False,
        edgecolor="#D0D0D0",
        framealpha=0.95,
        fontsize=10,
        handlelength=2.2,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved {out_path}")


pieces_uni = load_pieces(y=0)
pieces_gauss = load_pieces(y=1)

make_figure(
    pieces_uni,
    color=COLOR_UNIFORM,
    model_label="Uniform-sampled model",
    out_path="compare_fit_uniform.pdf",
)
make_figure(
    pieces_gauss,
    color=COLOR_GAUSSIAN,
    model_label="Gaussian-sampled model",
    out_path="compare_fit_gaussian.pdf",
)

save_summary([
    summarize("uniform", pieces_uni, load_perf(y=0)),
    summarize("gaussian", pieces_gauss, load_perf(y=1)),
])
