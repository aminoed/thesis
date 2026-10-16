import json
import glob
import os
import re
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import sys

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

# Define LQR parameters
Q = 500  # State cost
R = 0.01  # Control effort cost
powercontrol = []

# Frequency limits
min_GPU_freq, max_GPU_freq = 495, 1380
min_CPU_freq, max_CPU_freq = 1100, 2100


def get_power_target(n_gpu):
    return 600 + n_gpu * 150


def get_safe_power(n_gpu):
    return get_power_target(n_gpu) + 25


# Constrained optimization function
def compute_power(delta_gpu_freqs, delta_cpu_freqs, current_power, gpu_coefs, cpu_coef):
    return sum(d * c for d, c in zip(delta_gpu_freqs, gpu_coefs)) + delta_cpu_freqs * cpu_coef + current_power


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

    # Ensure bounds are valid
    if np.any(lower_bounds >= upper_bounds):
        raise ValueError("Dynamic bounds are invalid. Check the logic.")
    return lower_bounds, upper_bounds


def conservative_control(current_GPU_freqs, current_CPU_freqs):
    cpu_ctl = min_CPU_freq - current_CPU_freqs
    return np.array([min_GPU_freq - f for f in current_GPU_freqs] + [cpu_ctl])


def mpc(current_power, target, current_GPU_freqs, current_CPU_freqs, gpu_coefs, cpu_coef):
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
                delta = np.zeros(PROCESSOR_NUM)  # No control after control horizon

            # Update frequencies without snapping
            for i in range(n_gpu):
                predicted_GPU_freqs[i] = np.clip(
                    predicted_GPU_freqs[i] + delta[i],
                    min_GPU_freq, max_GPU_freq
                )

            predicted_CPU_freqs = np.clip(
                predicted_CPU_freqs + delta[-1],
                min_CPU_freq, max_CPU_freq
            )

            # Compute power without snapping
            power = compute_power(
                [predicted_GPU_freqs[i] - current_GPU_freqs[i] for i in range(n_gpu)],
                predicted_CPU_freqs - current_CPU_freqs,
                current_power,
                gpu_coefs,
                cpu_coef
            )
            # Tracking error
            cost += Q * (power - target) ** 2

            # Control effort
            if k < CONTROL_HORIZON:
                cost += R * (1 * np.sum((max_GPU_freq - predicted_GPU_freqs) ** 2) +
                             (max_CPU_freq - predicted_CPU_freqs) ** 2)
        return cost

    x0 = np.zeros(CONTROL_HORIZON * PROCESSOR_NUM)

    # Get dynamic bounds
    lower_bounds, upper_bounds = get_dynamic_bounds(current_GPU_freqs, current_CPU_freqs)
    # Convert bounds to a list of (min, max) tuples
    bounds = list(zip(lower_bounds, upper_bounds))
    result = minimize(objective, x0, bounds=bounds, method='SLSQP',
                      options={'maxiter': 3000})

    if result.success and result.x is not None and len(result.x) > 0:
        # Extract the first control action
        return result.x[:PROCESSOR_NUM]
    else:
        print("Optimization failed:", result.message)
        return np.zeros(PROCESSOR_NUM)


def load_model(model_path):
    # zhuo: read the adaptive subdivision model from json
    with open(model_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    pieces = []
    for p in data:
        exponents = p.get("exponents")
        pieces.append({
            "bounds": [tuple(map(float, b)) for b in p["bounds"]],
            "w": np.asarray(p["w"], dtype=float),
            "degree": int(p.get("degree", 1)),
            "exponents": [tuple(e) for e in exponents] if exponents is not None else None,
            "epsilon": float(p.get("epsilon", 0.0)),
            "n_sample": int(p.get("n_sample", len(p.get("idx", [])))),
            "idx": np.asarray(p.get("idx", []), dtype=int),
        })
    return pieces


def select_piece(xs, pieces):
    # zhuo: here we select the best piece which has the minimum error for the given state xs
    #       tol is the tolerance parameter because sometimes the state will be out of the boundary we learned,
    #       and it can be adjusted to a greater number if the system throws the ValueError below (like 0.1)
    xs = np.asarray(xs, dtype=float).ravel()
    tol = 0.1 * (x_max - x_min)
    tol_vec = np.asarray(tol, dtype=float).ravel()

    d = len(xs)
    models = []
    for p in pieces:
        select = True
        for i in range(d):
            lo, hi = p["bounds"][i]
            tol = tol_vec[i]
            if not (lo - tol <= xs[i] <= hi + tol):
                select = False
                break
        if select:
            models.append(p)

    # if not models:
    #     raise ValueError(f"not able to find a matched model with the given factors: {xs}")
    if not models:
        print("\n=== DEBUG: no matched piece ===")
        print("xs:", xs)

        for p_idx, p in enumerate(pieces[:5]):
            print(f"\n-- checking piece {p_idx} --")
            for i in range(d):
                lo, hi = p["bounds"][i]
                tol = tol_vec[i]
                if not (lo - tol <= xs[i] <= hi + tol):
                    print(f"[FAIL] dim={i}, xs={xs[i]:.6f}, "
                          f"range=[{lo:.6f},{hi:.6f}], tol={tol:.6f}")
                    break

        raise ValueError(f"not able to find a matched model with the given factors: {xs}")
    return min(models, key=lambda p: p["epsilon"])


def constrained_pac(pac_path):
    global pieces, piece_lo, piece_hi, x_min, x_max
    # Zhuo: here is another constraint to prevent the state is out of the bounds, no need to change it
    pieces = load_model(pac_path)

    piece_lo = np.array([min(p["bounds"][i][0] for p in pieces)
                         for i in range(len(pieces[0]["bounds"]))], dtype=float)
    piece_hi = np.array([max(p["bounds"][i][1] for p in pieces)
                         for i in range(len(pieces[0]["bounds"]))], dtype=float)

    x_min = piece_lo.copy()
    x_max = piece_hi.copy()

    x_min[:-1] = min_GPU_freq
    x_max[:-1] = max_GPU_freq
    x_min[-1] = min_CPU_freq
    x_max[-1] = max_CPU_freq


def predict(xs):
    # Zhuo: here we use the pieces to calculate the output of the current state(y_pred) with the w
    xs = np.asarray(xs, dtype=float).ravel()
    p = select_piece(xs, pieces)

    w = p["w"]

    if p["exponents"] is not None:
        phi = np.array([np.prod(xs ** np.asarray(e)) for e in p["exponents"]], dtype=float)
        if w.size != phi.size:
            raise ValueError(f"w size ({w.size}) != feature size ({phi.size})")
        y_pred = float(np.dot(w, phi))
    else:
        if w.size != xs.size + 1:
            raise ValueError(f"w size ({w.size}) != input size + 1 ({xs.size + 1})")
        y_pred = float(w[0] + np.dot(w[1:], xs))
    return y_pred, p


def random_state_update(state, x_min, x_max):
    # zhuo: we aid a small noise to the state and shows that the system will remain stable
    #       even if we add noises to the ML tasks, if it's 0,
    #       then there shouldn't have any fluctuations for the curve once it converges to the power limits
    state = state.copy()
    noise = 0.05  # this is for debug
    ranges = x_max - x_min
    noise = np.random.uniform(-noise, noise, size=state.shape)
    state = state + noise * ranges

    state = np.clip(state, x_min, x_max)

    return state, noise


def apply_power_error(power, error_type, P_min=350, P_max=4400):
    P_min = P_min / 4 * n_gpu
    P_max = P_max / 8 * n_gpu
    if error_type == "offset":
        if np.random.uniform(0, 1) < 0.5:
            return power + np.random.uniform(100, 225) * n_gpu
        else:
            return power + np.random.uniform(-225, -100) * n_gpu
    elif error_type == "mul":
        return np.random.uniform(0, 2) * power
    elif error_type == "stuck_min":
        return P_min
    elif error_type == "stuck_max":
        return P_max
    elif error_type == "out_of_range":
        if np.random.uniform(0, 1) < 0.5:
            fake_power = power + np.random.uniform(1000, 5000)
        else:
            fake_power = power + np.random.uniform(-5000, -1000)
        return np.clip(fake_power, P_min, P_max)
    elif error_type == "none":
        return power
    else:
        raise ValueError(f"Unknown error type: {error_type}")


def mpc_with_pac(current_state, current_GPU_freqs, current_CPU_freqs,
                 power_meter, pred_period, gpu_coefs, cpu_coefs,
                 target, delta_period, EPSILON_STAR):
    u = None
    if pred_period >= delta_period:
        u = conservative_control(current_GPU_freqs, current_CPU_freqs)
        return u, 0, "CON"

    power_pred_gpu, pac_piece = predict(current_state)
    piece_error = float(pac_piece["epsilon"])

    if piece_error > EPSILON_STAR:
        u = conservative_control(current_GPU_freqs, current_CPU_freqs)
        return u, 0, "CON"
    if np.abs(power_meter - power_pred_gpu) > EPSILON_STAR:
        u = mpc(power_pred_gpu, target, current_GPU_freqs, current_CPU_freqs, gpu_coefs, cpu_coefs)
        return u, pred_period + 1, "PAC"
    else:
        u = mpc(power_meter, target, current_GPU_freqs, current_CPU_freqs, gpu_coefs, cpu_coefs)
        return u, 0, "MPC"


def just_con(current_state, current_GPU_freqs, current_CPU_freqs,
             target, gpu_coefs, cpu_coefs, EPSILON_STAR, power_meter):
    u = None

    power_pred_gpu, pac_piece = predict(current_state)
    piece_error = float(pac_piece["epsilon"])
    if piece_error > EPSILON_STAR:
        u = conservative_control(current_GPU_freqs, current_CPU_freqs)
        return u, "CON"
    if np.abs(power_meter - power_pred_gpu) > EPSILON_STAR:
        u = conservative_control(current_GPU_freqs, current_CPU_freqs)
        return u, "CON"
    else:
        u = mpc(power_meter, target, current_GPU_freqs, current_CPU_freqs, gpu_coefs, cpu_coefs)
        return u, "MPC"


class RandomWalkKalman:
    def __init__(self, Q=0.2, R=4.0, initial_power=None, initial_P=1.0):
        self.Q = Q  # Process noise variance
        self.R = R  # Measurement noise variance
        self.initial_P = initial_P
        self.P = initial_P  # Estimation covariance
        self.x = initial_power  # Current estimate

    def reset(self):
        # Keep Q/R (per-GPU config); clear filter state between error/run
        self.x = None
        self.P = self.initial_P

    def update(self, measured_power):
        if self.x is None:
            self.x = measured_power
            return self.x

        x_pred = self.x
        P_pred = self.P + self.Q

        K = P_pred / (P_pred + self.R)

        self.x = x_pred + K * (measured_power - x_pred)
        self.P = (1 - K) * P_pred

        return self.x


def to_python(obj):
    if isinstance(obj, dict):
        return {k: to_python(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_python(v) for v in obj]
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


PAC_PATTERN = re.compile(
    r"pieces_NGPU(?P<ngpu>\d+)_a(?P<alpha>[\d.eE+-]+)_b(?P<beta>[\d.eE+-]+)"
    r"_k(?P<k>\d+)_e(?P<eps>[\d.eE+-]+)_m(?P<m>\d+)_s(?P<split>.+?)_NPAC(?P<npac>\d+)\.json$"
)

def parse_pac_filename(path):
    fname = os.path.basename(path)
    m = PAC_PATTERN.match(fname)
    if not m:
        return None
    info = m.groupdict()
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


# Zhuo: steps represents how many steps the system would run for each control period
#       error is the percentage of the possibility that the power meter would have wrong readings
steps = 100
runs = 1
ControlMethod = ["pac", "mpc", "con", "kalman", "mov_avg"]

np.random.seed(0)  # this is just the random seed set for the error above

# zhuo: replace the states below to one of the actual system state

error_types = ["offset", "mul", "stuck_min", "stuck_max", "out_of_range"]
# error_types = ["offset", "mul", "stuck_min", "stuck_max", "out_of_range", "failure", "none"]
GPU_LIST = [1, 2, 4]

all_pac_files = glob.glob("PAC_PIECE/*.json")

pac_db = []
for path in all_pac_files:
    info = parse_pac_filename(path)
    if info is not None:
        pac_db.append(info)

all_results = []

stdout_orig = sys.stdout
stderr_orig = sys.stderr

for n_gpu in GPU_LIST:
    global PROCESSOR_NUM
    PROCESSOR_NUM = n_gpu + 1
    gpu_coefs = COEFFICIENTS[n_gpu]["gpu"]
    cpu_coefs = COEFFICIENTS[n_gpu]["cpu"]

    log_file = open(f"rt_{n_gpu}gpu_PAC_updated.log", "w")
    sys.stdout = log_file
    sys.stderr = log_file

    try:
        # One Kalman filter per GPU count; Q/R can differ by n_gpu later
        if n_gpu == 1:
            kf = RandomWalkKalman(Q=0, R=0)
        elif n_gpu == 2:
            kf = RandomWalkKalman(Q=0, R=0)
        elif n_gpu == 4:
            kf = RandomWalkKalman(Q=0, R=0)

        P_target = get_power_target(n_gpu)
        print(f"\n===== Running GPU = {n_gpu} =====")

        for method in ControlMethod:
            if method == "pac" or method == "con":
                model_list = [
                    m for m in pac_db
                    if m["ngpu"] == n_gpu
                ]
            else:
                model_list = [None]

            for pac_m in model_list:
                if pac_m is not None:
                    constrained_pac(pac_m["path"])
                    if len(piece_lo) != n_gpu + 1:
                        raise ValueError(
                            f"[FATAL] PAC dimension mismatch: "
                            f"expected {n_gpu + 1}, got {len(piece_lo)}, file={pac_m['path']}"
                        )
                    init_state = piece_lo.copy()
                    EPSILON_STAR = pac_m["eps"]  
                    pac_tag = os.path.basename(pac_m["path"])
                else:
                    init_state = np.concatenate([
                        np.ones(n_gpu) * min_GPU_freq,
                        [min_CPU_freq]
                    ])
                    x_min = np.concatenate([np.full(n_gpu, min_GPU_freq), [min_CPU_freq]])
                    x_max = np.concatenate([np.full(n_gpu, max_GPU_freq), [max_CPU_freq]])
                    pac_tag = "none"

                for error_rate in [0.1, 0.5]:
                    for error in error_types:
                        for run in range(runs):
                            error_mask = np.random.rand(steps) < error_rate
                            num_pac = num_mpc = num_unsafe = num_con = 0
                            print(f"\n===== Error Rate {error_rate}, Type: {error} =====")
                            powercontrol = []  # actual power series
                            power_meter_series = []  # controlled power series
                            mode_series = []

                            current_GPU_states = init_state[:-1].copy()
                            current_CPU_state = init_state[-1]

                            current_GPU_freqs = current_GPU_states.copy()
                            current_CPU_freq = current_CPU_state

                            pred_period = 0
                            if method == "kalman":
                                kf.reset()  # same Q/R, fresh state for this error/run

                            # Simulate multiple control periods
                            for step in range(steps):
                                # Zhuo: Calculate current power -- replace to actual power meter reading
                                power = (sum(f * c for f, c in zip(current_GPU_freqs, gpu_coefs)) +
                                         current_CPU_freq * cpu_coefs)
                                # Zhuo: below here simulates the power meter error
                                hasError = error_mask[step]
                                # print(
                                #     f"===prediction is {power:.3f}, current state is {np.concatenate([
                                #     current_GPU_states,
                                #     [current_CPU_state]
                                # ])}, hasError? {hasError}")

                                if hasError and error != "none":
                                    power_meter = apply_power_error(power, error)
                                else:
                                    power_meter = power
                                if method == "noerror":
                                    power_meter = power

                                powercontrol.append(power)
                                power_meter_series.append(power_meter)

                                current_state = np.concatenate([
                                    current_GPU_states,
                                    [current_CPU_state]
                                ])
                                if power > get_safe_power(n_gpu):
                                    num_unsafe += 1
                                if method == "pac":
                                    u, pred_period, mode = mpc_with_pac(
                                        current_state=current_state,
                                        current_GPU_freqs=current_GPU_freqs,
                                        current_CPU_freqs=current_CPU_freq,
                                        power_meter=power_meter,
                                        pred_period=pred_period,
                                        gpu_coefs=gpu_coefs,
                                        cpu_coefs=cpu_coefs,
                                        target=P_target,
                                        delta_period=10,
                                        EPSILON_STAR=EPSILON_STAR
                                    )

                                    if mode == "PAC":
                                        num_pac += 1
                                    elif mode == "MPC":
                                        num_mpc += 1
                                    else:
                                        num_con += 1
                                    mode_series.append(mode)

                                elif method == "mpc" or method == "noerror":
                                    mode = "MPC"

                                    # current_GPU_freqs = float(current_GPU_freqs)

                                    u = mpc(power_meter, P_target, current_GPU_freqs, current_CPU_freq,
                                            gpu_coefs, cpu_coefs)

                                elif method == "con":
                                    u, mode = just_con(current_state=current_state,
                                                       current_GPU_freqs=current_GPU_freqs,
                                                       current_CPU_freqs=current_CPU_freq,
                                                       target=P_target,
                                                       gpu_coefs=gpu_coefs,
                                                       cpu_coefs=cpu_coefs,
                                                       EPSILON_STAR=EPSILON_STAR,
                                                       power_meter=power_meter)
                                    if mode == "CON":
                                        num_con += 1
                                    mode_series.append(mode)
                                elif method == "kalman":
                                    filtered_power = kf.update(power_meter)
                                    u = mpc(filtered_power, P_target, current_GPU_freqs,
                                            current_CPU_freq, gpu_coefs, cpu_coefs)

                                elif method == "mov_avg":
                                    if step == 0:
                                        power_avg = power_meter
                                    else:
                                        power_avg = np.mean(power_meter_series[-3:])
                                    u = mpc(power_avg, P_target, current_GPU_freqs,
                                            current_CPU_freq, gpu_coefs, cpu_coefs)


                                current_GPU_freqs = np.clip(
                                    current_GPU_freqs + u[:-1],
                                    min_GPU_freq,
                                    max_GPU_freq
                                )

                                current_CPU_freq = np.clip(
                                    current_CPU_freq + u[-1],
                                    min_CPU_freq,
                                    max_CPU_freq
                                )

                                current_GPU_states = current_GPU_freqs.copy()
                                current_CPU_state = current_CPU_freq

                                full_state = np.concatenate([
                                    current_GPU_states,
                                    [current_CPU_state]
                                ])

                                full_state, noise = random_state_update(full_state, x_min, x_max)

                                current_GPU_states = full_state[:-1]
                                current_CPU_state = full_state[-1]

                                if method == "pac":
                                    full_state = np.minimum(
                                        np.maximum(full_state, piece_lo),
                                        piece_hi
                                    )
                                    current_GPU_states = full_state[:-1]
                                    current_CPU_state = full_state[-1]

                            power_arr = np.array(powercontrol)
                            power_mean = float(np.mean(power_arr))
                            power_std = float(np.std(power_arr))
                            max_overshoot = np.max(powercontrol) - P_target
                            counts = {"PAC": num_pac, "MPC": num_mpc, "CON": num_con}

                            mode_per = {k: (counts[k] / steps) for k in counts}

                            result = {
                                "gpu_num": n_gpu, "method": method,
                                "pac_model": pac_tag, "error_rate": error_rate,
                                "error_type": error, "run": int(run), "steps": int(steps),

                                "power_series": powercontrol,
                                "power_meter_series": power_meter_series,
                                "EPSILON_STAR": EPSILON_STAR if method == "pac" or method == "con" else None,

                                "power_mean": power_mean, "power_std": power_std,
                                "max_overshoot": max_overshoot,
                                "r_unsafe": num_unsafe / steps,
                                "mode_counts": counts
                            }
                            result = to_python(result)
                            all_results.append(result)

                            # One JSON file per setting (gpu / method / pac / error / run)
                            pac_stem = os.path.splitext(pac_tag)[0]
                            out_path = (
                                f"control_results_NGPU{n_gpu}"
                                f"_method-{method}"
                                f"_err-{error}"
                                f"_rate-{error_rate}"
                                f"_run{run}"
                                f"_{pac_stem}.json"
                            )
                            with open(out_path, "w", encoding="utf-8") as f:
                                json.dump(result, f, ensure_ascii=False, indent=2)
                            print(f"Saved {out_path}")

    finally:
        sys.stdout = stdout_orig
        sys.stderr = stderr_orig
        log_file.close()

print(f"Done. Total records: {len(all_results)}")
