import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

N = 2000
LOW, HIGH = -np.pi, np.pi


def phi(x):
    return np.sin(x)


def sample_uniform(N, LOW, HIGH):
    # uniform
    x_uniform = np.random.uniform(LOW, HIGH, N)
    y_uniform = phi(x_uniform)
    # pd.DataFrame({"x0": x_uniform, "y": y_uniform}).to_csv("uniform.csv", index=False)
    return x_uniform, y_uniform

# # Gaussian (mean 0, std 0.4)
# scale = 0.4
# rng = np.random.default_rng()
# x_gauss = []
# while len(x_gauss) < N:
#     cand = rng.normal(loc=0.0, scale=scale, size=N)
#     keep = cand[(cand >= LOW) & (cand <= HIGH)]
#     x_gauss.extend(keep.tolist())
# x_gauss = np.array(x_gauss[:N])
# y_gauss = phi(x_gauss)
# pd.DataFrame({"x0": x_gauss, "y": y_gauss}).to_csv("gaussian.csv", index=False)

# fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

# axes[0].hist(x_uniform, bins=40, density=True, color="steelblue", edgecolor="white")
# axes[0].set_title(r"Uniform $x \sim U[-\pi,\pi]$")
# axes[0].set_xlabel("x")
# axes[0].set_ylabel("density")

# axes[1].hist(x_gauss, bins=40, density=True, color="darkorange", edgecolor="white")
# xs = np.linspace(LOW, HIGH, 300)
# pdf = (1.0 / (scale * np.sqrt(2 * np.pi))) * np.exp(-0.5 * (xs / scale) ** 2)
# axes[1].plot(xs, pdf, "k-", lw=2, label=f"$N(0, {scale}^2)$")
# axes[1].set_title(r"Gaussian $x$")
# axes[1].set_xlabel("x")
# axes[1].legend()

# plt.tight_layout()
# plt.savefig("sample_distribution.png", dpi=150)
# plt.show()

# print(f"uniform: {len(x_uniform)} points, x in [{x_uniform.min():.3f}, {x_uniform.max():.3f}]")
# print(f"gauss:   {len(x_gauss)} points, x in [{x_gauss.min():.3f}, {x_gauss.max():.3f}]")
