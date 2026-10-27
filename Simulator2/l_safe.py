import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PAC_DIR = os.path.join(BASE_DIR, "PAC_PIECE")
OUTPUT_DIR = os.path.join(BASE_DIR, "L_SAFE")

COEFFICIENTS = {
    1: {"gpu": [0.275430],
        "cpu": 0.272998},
    2: {"gpu": [0.245124, 0.294542],
        "cpu": 0.198053},
    4: {"gpu": [0.213441, 0.240409, 0.239106, 0.239705],
        "cpu": 0.129515},
}

CONTROL_HORIZON = 4
PREDICTION_HORIZON = 2

Q = 500
R = 0.01

# Same frequency limits as Simulator I.
min_GPU_freq, max_GPU_freq = 495, 1380
min_CPU_freq, max_CPU_freq = 1100, 2100

N_TEST = 360
L_MAX = 100
STATE_NOISE = 0.01

SAFE_MARGIN_PER_GPU = 25

INIT_BELOW_TARGET_LO = 100
INIT_BELOW_TARGET_HI = 50

NO_MATCH_REPORT_LIMIT = 3
no_match_reports = 0


def get_power_target(n_gpu):
    return 600 + 150 * n_gpu


def get_power_limit(n_gpu):
    return get_power_target(n_gpu) + SAFE_MARGIN_PER_GPU * n_gpu


def system_power(GPU_freqs, CPU_freq, gpu_coefs, cpu_coef):
    # Actual power from Simulator I, used only for safety monitoring.
    return float(np.dot(GPU_freqs, gpu_coefs) + CPU_freq * cpu_coef)


def compute_power(delta_gpu_freqs, delta_cpu_freqs, current_power,
                  gpu_coefs, cpu_coef):
    return (sum(d * c for d, c in zip(delta_gpu_freqs, gpu_coefs))
            + delta_cpu_freqs * cpu_coef + current_power)


def get_dynamic_bounds(GPU_freqs, CPU_freq):
    n_gpu = len(GPU_freqs)

    lower_bounds = []
    upper_bounds = []

    for i in range(n_gpu):
        lower_bounds.append(min_GPU_freq - GPU_freqs[i])
        upper_bounds.append(max_GPU_freq - GPU_freqs[i])

    lower_bounds.append(min_CPU_freq - CPU_freq)
    upper_bounds.append(max_CPU_freq - CPU_freq)

    lower_bounds = np.array(lower_bounds * CONTROL_HORIZON)
    upper_bounds = np.array(upper_bounds * CONTROL_HORIZON)

    if np.any(lower_bounds >= upper_bounds):
        raise ValueError("Dynamic bounds are invalid. Check the logic.")

    return lower_bounds, upper_bounds


def conservative_control(current_GPU_freqs, current_CPU_freqs):
    cpu_ctl = min_CPU_freq - current_CPU_freqs
    return np.array(
        [min_GPU_freq - f for f in current_GPU_freqs] + [cpu_ctl]
    )


def mpc(current_power, target, current_GPU_freqs, current_CPU_freqs,
        gpu_coefs, cpu_coef):
    n_gpu = len(current_GPU_freqs)
    PROCESSOR_NUM = n_gpu + 1

    def objective(x):
        control_sequence = x.reshape((CONTROL_HORIZON, PROCESSOR_NUM))
        predicted_GPU_freqs = np.array(current_GPU_freqs)
        predicted_CPU_freqs = current_CPU_freqs
        cost = 0

        for k in range(PREDICTION_HORIZON):
            if k < CONTROL_HORIZON:
                delta = control_sequence[k]
            else:
                delta = np.zeros(PROCESSOR_NUM)

            for i in range(n_gpu):
                predicted_GPU_freqs[i] = np.clip(
                    predicted_GPU_freqs[i] + delta[i],
                    min_GPU_freq, max_GPU_freq
                )

            predicted_CPU_freqs = np.clip(
                predicted_CPU_freqs + delta[-1],
                min_CPU_freq, max_CPU_freq
            )

            mpc_power = compute_power(
                [predicted_GPU_freqs[i] - current_GPU_freqs[i]
                 for i in range(n_gpu)],
                predicted_CPU_freqs - current_CPU_freqs,
                current_power,
                gpu_coefs,
                cpu_coef
            )

            cost += Q * (mpc_power - target) ** 2

            if k < CONTROL_HORIZON:
                cost += R * (
                    np.sum((max_GPU_freq - predicted_GPU_freqs) ** 2)
                    + (max_CPU_freq - predicted_CPU_freqs) ** 2
                )

        return cost

    x0 = np.zeros(CONTROL_HORIZON * PROCESSOR_NUM)

    lower_bounds, upper_bounds = get_dynamic_bounds(
        current_GPU_freqs, current_CPU_freqs
    )
    bounds = list(zip(lower_bounds, upper_bounds))

    result = minimize(
        objective, x0, bounds=bounds, method="SLSQP",
        options={"maxiter": 3000}
    )

    if result.success and result.x is not None and len(result.x) > 0:
        return result.x[:PROCESSOR_NUM]

    print("Optimization failed:", result.message)
    return np.zeros(PROCESSOR_NUM)


def load_model(model_path):
    with open(model_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    pieces = []

    for p in data:
        exponents = p.get("exponents")

        pieces.append({
            "bounds": [tuple(map(float, b)) for b in p["bounds"]],
            "w": np.asarray(p["w"], dtype=float),
            "epsilon": float(p.get("epsilon", 0.0)),
            "exponents": (
                np.asarray(exponents, dtype=int)
                if exponents is not None else None
            ),
        })

    if not pieces:
        raise ValueError(f"Empty PAC model: {model_path}")

    return pieces


def select_piece(xs, pieces, verbose=True):
    xs = np.asarray(xs, dtype=float).ravel()

    # Same tolerance as Simulator I online control.
    tol = 0.1 * (x_max - x_min)

    models = []

    for p in pieces:
        select = True

        for i in range(len(xs)):
            lo, hi = p["bounds"][i]

            if not (lo - tol[i] <= xs[i] <= hi + tol[i]):
                select = False
                break

        if select:
            models.append(p)

    if not models:
        if verbose:
            print("No matched PAC piece. State:", xs)
        return None

    return min(models, key=lambda p: p["epsilon"])


def constrained_pac(pac_path):
    global pieces, piece_lo, piece_hi, x_min, x_max

    pieces = load_model(pac_path)
    d = len(pieces[0]["bounds"])

    piece_lo = np.array(
        [min(p["bounds"][i][0] for p in pieces) for i in range(d)],
        dtype=float
    )
    piece_hi = np.array(
        [max(p["bounds"][i][1] for p in pieces) for i in range(d)],
        dtype=float
    )

    x_min = piece_lo.copy()
    x_max = piece_hi.copy()

    # Simulator I has one frequency per GPU, followed by CPU frequency.
    x_min[:-1] = min_GPU_freq
    x_max[:-1] = max_GPU_freq
    x_min[-1] = min_CPU_freq
    x_max[-1] = max_CPU_freq


def predict(xs):
    xs = np.asarray(xs, dtype=float).ravel()
    p = select_piece(xs, pieces, verbose=False)

    if p is None:
        return None, None

    w = p["w"]

    if p["exponents"] is not None:
        phi = np.prod(xs[None, :] ** p["exponents"], axis=1)

        if w.size != phi.size:
            raise ValueError("Polynomial coefficient count does not match.")

        y_pred = float(np.dot(w, phi))

    else:
        if w.size != xs.size + 1:
            raise ValueError("Linear coefficient count does not match.")

        y_pred = float(w[0] + np.dot(w[1:], xs))

    return y_pred, p


def random_state_update(state, x_min, x_max):
    # Same state update as Simulator I online control.
    state = state.copy()
    ranges = x_max - x_min

    noise = np.random.uniform(
        -STATE_NOISE, STATE_NOISE, size=state.shape
    )
    state = np.clip(state + noise * ranges, x_min, x_max)

    return state


def load_dataset(n_gpu):
    filename = f"Simulator1_gpu{n_gpu}_{20000 * n_gpu}.csv"
    df = pd.read_csv(os.path.join(BASE_DIR, filename))

    x_cols = [f"gpu{i}_freq" for i in range(1, n_gpu + 1)]
    x_cols.append("cpu_freq")

    X = df[x_cols].to_numpy(dtype=float)
    y = df["total_power"].to_numpy(dtype=float)

    return X, y, X.min(axis=0), X.max(axis=0)


def select_init_state(X, y, P_target, P_limit, n_gpu, max_tries=50):
    lo = P_target - INIT_BELOW_TARGET_LO * n_gpu
    hi = P_target - INIT_BELOW_TARGET_HI * n_gpu

    candidates = np.flatnonzero((y > lo) & (y <= hi))

    if candidates.size == 0:
        candidates = np.flatnonzero((y > lo) & (y <= P_limit))

    if candidates.size == 0:
        raise ValueError(
            "No initial state satisfies the specified power range."
        )

    for _ in range(max_tries):
        init_idx = int(np.random.choice(candidates))

        if select_piece(X[init_idx], pieces, verbose=False) is not None:
            return init_idx

    return int(np.random.choice(candidates))


def mpc_with_pac(current_state, current_GPU_freqs, current_CPU_freqs,
                 gpu_coefs, cpu_coefs, target):
    power_pred, pac_piece = predict(current_state)

    if pac_piece is None:
        return conservative_control(
            current_GPU_freqs, current_CPU_freqs
        ), "CON"

    u = mpc(
        power_pred, target, current_GPU_freqs, current_CPU_freqs,
        gpu_coefs, cpu_coefs
    )

    return u, "PAC"


PAC_PATTERN = re.compile(
    r"pieces_NGPU(?P<ngpu>\d+)_a(?P<alpha>[\d.eE+-]+)"
    r"_b(?P<beta>[\d.eE+-]+)_k(?P<k>\d+)"
    r"_e(?P<eps>[\d.eE+-]+)_m(?P<m>\d+)_s?"
    r"(?P<split>residual_range|maximum_residual)"
    r"_NPAC(?P<npac>\d+)\.json$"
)


def parse_pac_filename(path):
    match = PAC_PATTERN.fullmatch(os.path.basename(path))

    if not match:
        return None

    info = match.groupdict()

    return {
        "ngpu": int(info["ngpu"]),
        "alpha": float(info["alpha"]),
        "beta": float(info["beta"]),
        "k": int(info["k"]),
        "eps": float(info["eps"]),
        "m": int(info["m"]),
        "split": info["split"],
        "npac": int(info["npac"]),
        "path": path,
    }



all_pac_files = sorted(
    glob.glob(os.path.join(PAC_DIR, "*.json"))
)

pac_db = []

for path in all_pac_files:
    info = parse_pac_filename(path)

    if info is not None:
        pac_db.append(info)

if not pac_db:
    raise ValueError(f"No matching PAC models found in {PAC_DIR}")

np.random.seed(0)
os.makedirs(OUTPUT_DIR, exist_ok=True)

log_path = os.path.join(
    OUTPUT_DIR, "output_lsafe_simulator1.log"
)
summary_path = os.path.join(
    OUTPUT_DIR, "lsafe_summary.csv"
)

results = []

stdout_orig = sys.stdout
stderr_orig = sys.stderr
log_file = open(log_path, "w", encoding="utf-8")

sys.stdout = log_file
sys.stderr = log_file

try:
    for pac_m in pac_db:
        constrained_pac(pac_m["path"])
        no_match_reports = 0

        n_gpu = pac_m["ngpu"]

        if len(piece_lo) != n_gpu + 1:
            raise ValueError(
                f"PAC dimension mismatch: {pac_m['path']}"
            )

        # Keep the frequency bounds set by constrained_pac().
        X, y, _, _ = load_dataset(n_gpu)

        gpu_coefs = COEFFICIENTS[n_gpu]["gpu"]
        cpu_coefs = COEFFICIENTS[n_gpu]["cpu"]

        P_target = get_power_target(n_gpu)
        P_safe = get_power_limit(n_gpu)
        P_limit = P_safe
        
        L_max = L_MAX
        L_safe = L_max
        NPAC = N_TEST

        print(f"===== Model: {pac_m['path']}")
        print(f"P_safe={P_safe}, P_limit={P_limit}")

        while NPAC > 0:
            NPAC -= 1

            init_idx = select_init_state(
                X, y, P_target, P_limit, n_gpu
            )
            init_state = X[init_idx].copy()

            current_GPU_freqs = init_state[:-1].copy()
            current_CPU_freq = float(init_state[-1])

            steps = L_safe

            for step in range(steps):
                current_state = np.concatenate(
                    [current_GPU_freqs, [current_CPU_freq]]
                )

                # Actual power is used only for safety monitoring.
                power = system_power(
                    current_GPU_freqs, current_CPU_freq,
                    gpu_coefs, cpu_coefs
                )

                print(f"Step {step}: power={power:.3f}")

                if power > P_limit:
                    L_safe = step
                    break

                # The controller receives only the PAC prediction.
                u, mode = mpc_with_pac(
                    current_state,
                    current_GPU_freqs,
                    current_CPU_freq,
                    gpu_coefs,
                    cpu_coefs,
                    P_target
                )

                if mode != "PAC":
                    # Stop before applying conservative control.
                    L_safe = step
                    break

                current_GPU_freqs = np.clip(
                    current_GPU_freqs + u[:-1],
                    min_GPU_freq, max_GPU_freq
                )
                current_CPU_freq = float(np.clip(
                    current_CPU_freq + u[-1],
                    min_CPU_freq, max_CPU_freq
                ))

                full_state = np.concatenate(
                    [current_GPU_freqs, [current_CPU_freq]]
                )
                full_state = random_state_update(
                    full_state, x_min, x_max
                )

                # Same PAC-domain clipping as online control.
                full_state = np.clip(
                    full_state, piece_lo, piece_hi
                )

                current_GPU_freqs = full_state[:-1].copy()
                current_CPU_freq = float(full_state[-1])

                # Check the updated state, including the last step.
                power = system_power(
                    current_GPU_freqs, current_CPU_freq,
                    gpu_coefs, cpu_coefs
                )

                if power > P_limit:
                    L_safe = step
                    print(
                        f"Unsafe updated state: power={power:.3f}"
                    )
                    break

            if L_safe == 0:
                break

        result = {
            key: value
            for key, value in pac_m.items()
            if key != "path"
        }

        result.update({
            "pac_model": os.path.basename(pac_m["path"]),
            "P_safe": P_safe,
            "P_limit": P_limit,
            "L_max": L_max,
            "L_safe": L_safe,
            "n_test_required": N_TEST,
            "n_test_completed": N_TEST - NPAC,
            "state_noise": STATE_NOISE,
        })

        results.append(result)
        pd.DataFrame(results).to_csv(
            summary_path, index=False
        )

        print(f"===== Final result: {result}\n")
        print(
            f"GPU={n_gpu}, model={result['pac_model']}, "
            f"L_safe={L_safe}",
            file=stdout_orig
        )

finally:
    sys.stdout = stdout_orig
    sys.stderr = stderr_orig
    log_file.close()

print(f"Results saved to {summary_path}")
