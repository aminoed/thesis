from pathlib import Path

import numpy as np
import pandas as pd

from GPUPowerFNN import load_fnn, predict_power_fnn

# (n_gpu, num_samples)
CONFIGS = [
    (1, 20000),
    (2, 40000),
    (4, 80000),
]

GPU_FREQS = np.arange(495, 1380 + 1, 15)  # 495, 510, ..., 1380
CPU_FREQ = 2000.0
OUTPUT_TEMPLATE = "FNN_{n_gpu}_{k}.csv"

np.random.seed(0)


def model_path(n_gpu):
    return f"fnn_gpu_{n_gpu}.pt"


def ref_csv_path(n_gpu):
    return f"gpu_{n_gpu}.csv"


def sample_feature_matrix(df_ref, x_cols, n_gpu, n_samples):
    idx = np.random.randint(0, len(df_ref), size=n_samples)
    X = df_ref[x_cols].to_numpy(dtype=np.float32)[idx].copy()

    col_index = {c: i for i, c in enumerate(x_cols)}

    for gid in range(n_gpu):
        X[:, col_index[f"gpu{gid}_freq"]] = np.random.choice(GPU_FREQS, size=n_samples)

    if "cpu_freq" in col_index:
        X[:, col_index["cpu_freq"]] = CPU_FREQ

    return X


def generate_dataset(n_gpu, n_samples):
    pt_path = model_path(n_gpu)
    csv_path = ref_csv_path(n_gpu)

    print(f"\n=== n_gpu={n_gpu}, samples={n_samples} ===")
    print(f"Model: {pt_path}")
    print(f"Ref CSV: {csv_path}")

    model, device, scaler, state = load_fnn(pt_path)
    x_cols = state["x_cols"]
    df_ref = pd.read_csv(csv_path)

    X = sample_feature_matrix(df_ref, x_cols, n_gpu, n_samples)
    y = predict_power_fnn(model, X, device, scaler)

    df_out = pd.DataFrame(X, columns=x_cols)
    df_out["total_power"] = np.asarray(y, dtype=np.float32).reshape(-1)
    return df_out



for n_gpu, n_samples in CONFIGS:
    df = generate_dataset(n_gpu, n_samples)
    out_path = OUTPUT_TEMPLATE.format(n_gpu=n_gpu, k=n_samples)
    df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}  shape={df.shape}")
