import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

COLOR_TRUE = "#9A9A9A"
COLOR_UNIFORM = "#1F77B4"
COLOR_GAUSSIAN = "#D62728"

LOW, HIGH = -np.pi, np.pi


def phi(x):
    return np.sin(x)


def fit_linear(x, y):
    # ŷ = w0 + w1 x  (ordinary least squares)
    A = np.column_stack([np.ones_like(x), x])
    w, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    return w  # [w0, w1]


def predict(w, x):
    return w[0] + w[1] * x


def max_abs_error(w, grid=2000):
    xs = np.linspace(LOW, HIGH, grid)
    return float(np.max(np.abs(predict(w, xs) - phi(xs))))


def summarize(name, w, x, y):
    resid = y - predict(w, x)
    return {
        "distribution": name,
        "w0": float(w[0]),
        "w1": float(w[1]),
        "train_mse": float(np.mean(resid ** 2)),
        "train_max_abs": float(np.max(np.abs(resid))),
        "grid_max_abs": max_abs_error(w),
        "x_mean": float(np.mean(x)),
        "x_std": float(np.std(x)),
    }


def save_summary(rows, out_path="compare_summary.csv"):
    fields = list(rows[0].keys())
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved {out_path}")
    print()
    print(
        f"{'distribution':<12} {'w0':>10} {'w1':>10} "
        f"{'train_mse':>12} {'train_max':>10} {'grid_max':>10}"
    )
    for r in rows:
        print(
            f"{r['distribution']:<12} {r['w0']:>10.4f} {r['w1']:>10.4f} "
            f"{r['train_mse']:>12.4f} {r['train_max_abs']:>10.4f} {r['grid_max_abs']:>10.4f}"
        )


def style_axes(ax, ylim):
    pad = 0.05 * (HIGH - LOW)
    ax.set_xlim(LOW - pad, HIGH + pad)
    ax.set_ylim(*ylim)
    ax.set_xlabel(r"$x$", fontsize=12)
    ax.set_ylabel(r"$y$", fontsize=12)
    ax.tick_params(labelsize=10, length=3.5, width=0.8)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#666666")
        ax.spines[spine].set_linewidth(0.9)


def density_heights(x_samples, bin_width=0.25):
    n_bins = int(round((HIGH - LOW) / bin_width))
    edges = np.linspace(LOW, HIGH, n_bins + 1)
    counts, _ = np.histogram(x_samples, bins=edges)
    heights = counts / (float(len(x_samples)) * bin_width)
    return edges, heights


def plot_sample_hist(ax, x_samples, color, dens_ylim, bin_width=0.25):
    ax_r = ax.twinx()
    edges, heights = density_heights(x_samples, bin_width)
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
    ax_r.set_ylim(*dens_ylim)

    ax.set_zorder(ax_r.get_zorder() + 1)
    ax.patch.set_visible(False)
    return ax_r


def make_figure(w, x_samples, color, model_label, out_path, ylim, dens_ylim):
    plt.rcParams.update({"font.family": "sans-serif"})
    x_true = np.linspace(LOW, HIGH, 500)

    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=150)
    ax.plot(
        x_true, phi(x_true),
        color=COLOR_TRUE, lw=3.0, ls="-",
        label=r"True $\varphi(x)=\sin x$",
        zorder=2,
    )
    ax.plot(
        x_true, predict(w, x_true),
        color=color, lw=2.2,
        label=model_label,
        zorder=3,
    )
    style_axes(ax, ylim)
    plot_sample_hist(ax, x_samples, color, dens_ylim)
    ax.legend(
        loc="upper left",
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


def make_overlay(w_uni, w_gauss, ylim, out_path="compare_fit_overlay.pdf"):
    plt.rcParams.update({"font.family": "sans-serif"})
    x_true = np.linspace(LOW, HIGH, 500)

    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=150)
    ax.plot(
        x_true, phi(x_true),
        color=COLOR_TRUE, lw=3.0,
        label=r"True $\varphi(x)$",
        zorder=2,
    )
    ax.plot(
        x_true, predict(w_uni, x_true),
        color=COLOR_UNIFORM, lw=2.2,
        label=rf"Uniform fit: $y={w_uni[0]:.2f}+{w_uni[1]:.2f}x$",
        zorder=3,
    )
    ax.plot(
        x_true, predict(w_gauss, x_true),
        color=COLOR_GAUSSIAN, lw=2.2,
        label=rf"Gaussian fit: $y={w_gauss[0]:.2f}+{w_gauss[1]:.2f}x$",
        zorder=3,
    )
    style_axes(ax, ylim)
    ax.legend(
        loc="upper left",
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


df_uni = pd.read_csv("uniform.csv")
df_gauss = pd.read_csv("gaussian.csv")
x_uni = df_uni["x0"].to_numpy()
y_uni = df_uni["y"].to_numpy()
x_gauss = df_gauss["x0"].to_numpy()
y_gauss = df_gauss["y"].to_numpy()

w_uni = fit_linear(x_uni, y_uni)
w_gauss = fit_linear(x_gauss, y_gauss)

# shared y-limit covering true peak and both lines
x_grid = np.linspace(LOW, HIGH, 500)
y_all = np.concatenate([
    phi(x_grid),
    predict(w_uni, x_grid),
    predict(w_gauss, x_grid),
])
pad = 0.05 * (y_all.max() - y_all.min())
ylim = (float(y_all.min() - pad), float(y_all.max() + pad))

# shared density axis so the two figures are comparable
_, h_uni = density_heights(x_uni)
_, h_gauss = density_heights(x_gauss)
dens_ylim = (0.0, float(max(h_uni.max(), h_gauss.max()) * 1.15))

make_figure(
    w_uni, x_uni, COLOR_UNIFORM,
    model_label=rf"Uniform LR: $y={w_uni[0]:.2f}+{w_uni[1]:.2f}x$",
    out_path="compare_fit_uniform.pdf",
    ylim=ylim,
    dens_ylim=dens_ylim,
)
make_figure(
    w_gauss, x_gauss, COLOR_GAUSSIAN,
    model_label=rf"Gaussian LR: $y={w_gauss[0]:.2f}+{w_gauss[1]:.2f}x$",
    out_path="compare_fit_gaussian.pdf",
    ylim=ylim,
    dens_ylim=dens_ylim,
)
make_overlay(w_uni, w_gauss, ylim)

save_summary([
    summarize("uniform", w_uni, x_uni, y_uni),
    summarize("gaussian", w_gauss, x_gauss, y_gauss),
])
