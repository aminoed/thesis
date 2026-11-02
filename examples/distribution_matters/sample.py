import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

N = 2000
LOW, HIGH = -1.0, 1.0

# uniform
x_uniform = np.random.uniform(LOW, HIGH, N)
y_uniform = x_uniform ** 2
pd.DataFrame({"x0": x_uniform, "y": y_uniform}).to_csv("uniform.csv", index=False)

# Gaussian (mean 0, std 0.4)
rng = np.random.default_rng()
x_gauss = []
while len(x_gauss) < N:
    cand = rng.normal(loc=0.0, scale=0.4, size=N)
    keep = cand[(cand >= LOW) & (cand <= HIGH)]
    x_gauss.extend(keep.tolist())
x_gauss = np.array(x_gauss[:N])
y_gauss = x_gauss ** 2
pd.DataFrame({"x0": x_gauss, "y": y_gauss}).to_csv("gaussian.csv", index=False)

fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

axes[0].hist(x_uniform, bins=40, density=True, color="steelblue", edgecolor="white")
axes[0].set_title("Uniform x ~ U[-1, 1]")
axes[0].set_xlabel("x")
axes[0].set_ylabel("density")

axes[1].hist(x_gauss, bins=40, density=True, color="darkorange", edgecolor="white")
# overlay the theoretical N(0, 0.3^2) pdf (untruncated, for visual check)
xs = np.linspace(LOW, HIGH, 300)
pdf = (1.0 / (0.3 * np.sqrt(2 * np.pi))) * np.exp(-0.5 * (xs / 0.3) ** 2)
axes[1].plot(xs, pdf, "k-", lw=2, label=r"$N(0, 0.3^2)$")
axes[1].set_title("Gaussian x (truncated to [-1, 1])")
axes[1].set_xlabel("x")
axes[1].legend()

plt.tight_layout()
plt.savefig("sample_distribution.png", dpi=150)
plt.show()

print(f"uniform: {len(x_uniform)} points, x in [{x_uniform.min():.3f}, {x_uniform.max():.3f}]")
print(f"gauss:   {len(x_gauss)} points, x in [{x_gauss.min():.3f}, {x_gauss.max():.3f}]")
