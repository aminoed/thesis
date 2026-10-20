import re

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


DATA_PATH = "gpu_4.csv"

GPU_FEATURE_SUFFIXES = ("freq", "SM_util", "mem_util")
SHARED_FEATURES = ("cpu_freq", "cpu_util")
Y_COL = "total_power"

np.random.seed(0)
torch.manual_seed(0)


def discover_x_cols(columns):
    cols = set(columns)

    gpu_ids = sorted(
        {int(m.group(1)) for c in cols if (m := re.fullmatch(r"gpu(\d+)_freq", c))}
    )
    n_gpu = len(gpu_ids)

    x_cols = []
    for gid in gpu_ids:
        for suffix in GPU_FEATURE_SUFFIXES:
            x_cols.append(f"gpu{gid}_{suffix}")

    for col in SHARED_FEATURES:
        if col in cols:
            x_cols.append(col)

    fan_cols = sorted(
        (c for c in cols if re.fullmatch(r"fan\d+_speed", c)),
        key=lambda c: int(re.fullmatch(r"fan(\d+)_speed", c).group(1)),
    )
    x_cols.extend(fan_cols)
    return x_cols, n_gpu


def model_path_for(n_gpu):
    return f"fnn_gpu_{n_gpu}.pt"


def load_data(csv_path):
    df = pd.read_csv(csv_path)

    x_cols, n_gpu = discover_x_cols(df.columns)
    y_col = Y_COL

    df = df.dropna(subset=x_cols + [y_col]).reset_index(drop=True)

    X = df[x_cols].values.astype(np.float32)
    y = df[y_col].values.astype(np.float32).reshape(-1, 1)

    print(f"n_gpu={n_gpu}, features({len(x_cols)}): {x_cols}")
    return df, X, y, x_cols, n_gpu


class PowerFNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x)


def train_fnn(X, y, epochs=344, batch_size=256, lr=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    n = len(X)
    idx = np.random.permutation(n)
    train_idx = idx[: int(0.8 * n)]
    val_idx = idx[int(0.8 * n) :]

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train), torch.tensor(y_train)),
        batch_size=batch_size,
        shuffle=True,
    )

    val_loader = DataLoader(
        TensorDataset(torch.tensor(X_val), torch.tensor(y_val)), batch_size=batch_size
    )

    model = PowerFNN(X.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=100)
    criterion = nn.MSELoss()
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)

            pred = model(xb)
            loss = criterion(pred, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * xb.size(0)

        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0

        all_preds = []
        all_targets = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                val_loss += criterion(pred, yb).item() * xb.size(0)

                all_preds.append(pred.cpu().numpy())
                all_targets.append(yb.cpu().numpy())
        val_loss /= len(val_loader.dataset)

        all_preds = np.vstack(all_preds)
        all_targets = np.vstack(all_targets)
        acc = regression_accuracy(all_targets, all_preds, threshold=1.0)

        if epoch % 10 == 0:
            print(
                f"Epoch {epoch:03d} | Train MSE: {train_loss:.6f} | Val MSE: {val_loss:.6f} | Val Acc(<1W): {acc:.4f}"
            )
    return model, val_idx, scaler


def regression_accuracy(y_true, y_pred, threshold=1.0):
    return np.mean(np.abs(y_true - y_pred) < threshold)


def save_model(model, path, x_cols, scaler, n_gpu):
    torch.save(
        {
            "state_dict": model.state_dict(),
            "x_cols": x_cols,
            "scaler": scaler,
            "n_gpu": n_gpu,
        },
        path,
    )


def load_fnn(path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    state = torch.load(path, map_location=device, weights_only=False)

    model = PowerFNN(len(state["x_cols"])).to(device)
    model.load_state_dict(state["state_dict"])
    model.eval()

    scaler = state["scaler"]
    return model, device, scaler, state


def predict_power_fnn(model, x_raw, device, scaler):
    x = np.array(x_raw, dtype=np.float32)

    if x.ndim == 1:
        x = x[None, :]

    x = scaler.transform(x)

    x_tensor = torch.tensor(x).to(device)

    with torch.no_grad():
        y_pred = model(x_tensor).cpu().numpy()

    return y_pred.squeeze()


def evaluate_model(model, df, X, y, val_idx, device, scaler):
    model.eval()

    X_val = scaler.transform(X[val_idx])
    y_val = y[val_idx]

    with torch.no_grad():
        p_pred = (
            model(torch.tensor(X_val, dtype=torch.float32).to(device)).cpu().numpy()
        )

    p_true = y_val
    error = p_pred - p_true

    mae = np.mean(np.abs(error))
    rmse = np.sqrt(np.mean(error**2))

    acc_1w = np.mean(np.abs(error) < 1.0)
    acc_5w = np.mean(np.abs(error) < 5.0)

    ss_res = np.sum((p_true - p_pred) ** 2)
    ss_tot = np.sum((p_true - np.mean(p_true)) ** 2)
    r2 = 1 - ss_res / ss_tot

    print("\n===== Evaluation (on total_power) =====")
    print(f"MAE: {mae:.4f} W")
    print(f"RMSE: {rmse:.4f} W")
    print(f"Acc (<1W): {acc_1w:.4f}")
    print(f"Acc (<5W): {acc_5w:.4f}")
    print(f"R^2: {r2:.4f}")

    return {
        "error": error,
        "mae": mae,
        "rmse": rmse,
        "acc_1w": acc_1w,
        "acc_5w": acc_5w,
        "r2": r2,
    }


if __name__ == "__main__":
    print(f"Loading data from {DATA_PATH}...")
    df, X, y, x_cols, n_gpu = load_data(DATA_PATH)
    model_path = model_path_for(n_gpu)

    print("Start training...")
    model, val_idx, scaler = train_fnn(X, y, epochs=79, batch_size=256, lr=0.003)

    print(f"Saving model to {model_path}...")
    save_model(model, model_path, x_cols, scaler, n_gpu)

    print("Training finished!")

    model, device, scaler, state = load_fnn(model_path)

    sample_raw = df.iloc[0][x_cols].values
    pred = predict_power_fnn(model, sample_raw, device, scaler)
    print("sample:", sample_raw)
    print("true power:", df.iloc[0]["total_power"])
    print("Example prediction:", pred)
    evaluate_model(model, df, X, y, val_idx, device, scaler)
