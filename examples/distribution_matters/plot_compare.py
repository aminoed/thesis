import csv
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# y0 = uniform, y1 = gaussian
TAG = "a0.1_b0.01_k2000_e0.01_m1_sresidual_range_NPAC153_y{y}"

COLOR_TRUE = "#9A9A9A"
COLOR_UNIFORM = "#1F77B4"
COLOR_GAUSSIAN = "#D62728"

YLIM = (-0.05, 1.08)
YTICKS = np.arange(0.0, 1.0 + 1e-9, 0.2)


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
    ax.set_ylim(*YLIM)
    ax.set_yticks(YTICKS)  
    ax.set_xlabel(r"$x$", fontsize=12)
    ax.set_ylabel(r"$y$", fontsize=12)
    ax.tick_params(labelsize=10, length=3.5, width=0.8)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#666666")
        ax.spines[spine].set_linewidth(0.9)


def plot_sample_hist(ax, x_samples, color, bin_width=0.1):
    ax_r = ax.twinx()
    low, high = -1.0, 1.0
    n_bins = int(round((high - low) / bin_width))
    edges = np.linspace(low, high, n_bins + 1)
    counts, _ = np.histogram(x_samples, bins=edges)

    heights = counts / (float(len(x_samples)) * bin_width)
    ax_r.bar(
        edges[:-1], heights,
        width=bin_width,
        align="edge",
        color=color,
        alpha=0.22,
        linewidth=0,
        zorder=0,
    )
    ax_r.set_ylabel("Sample density", color=color, fontsize=12)
    ax_r.tick_params(axis="y", labelsize=10, length=3.5, width=0.8, colors=color)
    ax_r.spines["top"].set_visible(False)
    ax_r.spines["right"].set_color(color)
    ax_r.spines["right"].set_linewidth(0.9)
    ax_r.spines["left"].set_visible(False)
    ax_r.spines["bottom"].set_visible(False)
    ax_r.set_xlim(ax.get_xlim())
    ax_r.set_ylim(*YLIM)  
    ax_r.set_yticks(YTICKS)

    ax.set_zorder(ax_r.get_zorder() + 1)
    ax.patch.set_visible(False)
    return ax_r


def make_figure(pieces, x_samples, color, model_label, out_path):
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "Times"],
        "mathtext.fontset": "stix", "font.size": 10, "pdf.fonttype": 42,
        "axes.spines.top": False, "axes.spines.right": False,
    })
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
    plot_sample_hist(ax, x_samples, color)
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(0.02, 1.0),
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
x_uni = pd.read_csv("uniform.csv")["x0"].to_numpy()
x_gauss = pd.read_csv("gaussian.csv")["x0"].to_numpy()

make_figure(
    pieces_uni,
    x_samples=x_uni,
    color=COLOR_UNIFORM,
    model_label="Uniform-sampled model",
    out_path="compare_fit_uniform.pdf",
)
make_figure(
    pieces_gauss,
    x_samples=x_gauss,
    color=COLOR_GAUSSIAN,
    model_label="Gaussian-sampled model",
    out_path="compare_fit_gaussian.pdf",
)

save_summary([
    summarize("uniform", pieces_uni, load_perf(y=0)),
    summarize("gaussian", pieces_gauss, load_perf(y=1)),
])
