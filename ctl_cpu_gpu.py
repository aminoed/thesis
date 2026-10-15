import glob
import json
import os
import re
from pynvml import *
import csv
import multiprocessing
import subprocess
import time
from datetime import datetime
from collections import defaultdict
import psutil
from dataclasses import dataclass
import atexit
import threading
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from pac_features import (
    build_feature_column,
    cpu_freq_index,
    features_per_gpu,
    gpu_freq_indices,
    N_FANS,
)

SENSOR_MAP = {
    "CPU0_Temp": "ipmi_cpu0_temp_c",
    "CPU1_Temp": "ipmi_cpu1_temp_c",
    "PCH_Temp": "ipmi_pch_temp_c",
    "SYS_Air_Inlet": "ipmi_sys_air_inlet_temp_c",
    "MB_Air_Inlet": "ipmi_mb_air_inlet_temp_c",
    "SYS_Air_Outlet": "ipmi_sys_air_outlet_temp_c",
}

IPMI_KEYS = [
    "ipmi_cpu0_temp_c",
    "ipmi_cpu1_temp_c",
    "ipmi_pch_temp_c",
    "ipmi_sys_air_inlet_temp_c",
    "ipmi_mb_air_inlet_temp_c",
    "ipmi_sys_air_outlet_temp_c",
]

FAN_KEYS = [f"SYS_FAN{i}" for i in range(1, 7)]

COEFFICIENTS = {
    1: {"gpu": [0.180046],
        "cpu": 0.333423},
    2: {"gpu": [0.179403, 0.215758],
        "cpu": 0.233980},
    4: {"gpu": [0.13472, 0.13472, 0.13472, 0.13472],
        "cpu": 0.333423},
    6: {"gpu": [0.026031, 0.020563, -0.003276, 0.065450, 0.117049, 0.120461],
        "cpu": 0.365159},
}

PAC_PATTERN = re.compile(
    r"pieces_NGPU(?P<ngpu>\d+)_a(?P<alpha>[\d.eE+-]+)_b(?P<beta>[\d.eE+-]+)"
    r"_k(?P<k>\d+)_e(?P<eps>[\d.eE+-]+)_NPAC(?P<npac>\d+)\.json$"
)

CONTROL_HORIZON = 4
PREDICTION_HORIZON = 2
FEATURES_PER_GPU = features_per_gpu()

min_GPU_freq, max_GPU_freq = 495, 1380
min_CPU_freq, max_CPU_freq = 1100, 2100
GPU_freq_levels = np.arange(min_GPU_freq, max_GPU_freq + 15, 15)
Q = 500
R = 0.001

LATENCY_LOG_DIR = '/home/pacs/Yuan/mpc/logs'


def clear_latency_logs(gpu_ids):
    """Truncate per-GPU latency log files so the next period starts fresh."""
    for gid in gpu_ids:
        path = os.path.join(LATENCY_LOG_DIR, f'latency_log_{gid}.csv')
        with open(path, 'w') as f:
            pass


def read_latency_logs(gpu_ids):
    """
    Read all lines written by InferenceThread since the last clear_latency_logs()
    call.  Returns a list of dicts:
        {"gpu_id": int, "timestamp": float, "latency_ms": float}
    Malformed / incomplete lines are silently skipped.
    """
    records = []
    for gid in gpu_ids:
        path = os.path.join(LATENCY_LOG_DIR, f'latency_log_{gid}.csv')
        try:
            with open(path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split(',')
                    if len(parts) < 2:
                        continue
                    try:
                        records.append({
                            'gpu_id': gid,
                            'timestamp': float(parts[0]),
                            'latency_ms': float(parts[1]),
                        })
                    except ValueError:
                        pass
        except FileNotFoundError:
            pass
    return records


def clean_record():
    with open('/home/pacs/Yuan/mpc/logs/tmp_5_2', 'w') as file:
        file.write('')
    with open('/home/pacs/Yuan/mpc/logs/tmp_wm', 'w') as file:
        file.write('')


def delta_sigma_GPU(next_GPU_freq1, next_GPU_freq2, gpu_ratio, gpu):
    try:
        # print('marker')
        # print(time.time())
        nvmlInit()
        handle_freq = nvmlDeviceGetHandleByIndex(gpu)
        start_time = time.time()
        while True:
            elapsed_time = time.time() - start_time
            if elapsed_time >= 4:
                break
            time_s = datetime.now()
            nvmlDeviceSetApplicationsClocks(handle_freq, 877, next_GPU_freq1)
            # print(next_GPU_freq1)
            time_n = datetime.now()
            timeslice = float(gpu_ratio) * 0.5
            time.sleep(timeslice)
            if gpu_ratio != 1:
                nvmlDeviceSetApplicationsClocks(handle_freq, 877, next_GPU_freq2)
                timeslice = float(4 - gpu_ratio) * 0.5
                time.sleep(timeslice)
    finally:
        nvmlShutdown()


def avg(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        contents = file.read()
    # print(contents)
    powermeterlst = []
    contents = contents.split()
    for content in contents:
        powermeterlst.append(float(content))
    return sum(powermeterlst) / len(powermeterlst)


def snap_gpu_freq_delta_sigma(freq):
    freq = np.clip(freq, min_GPU_freq, max_GPU_freq)
    idx = np.argmin(np.abs(GPU_freq_levels - freq))
    if freq < GPU_freq_levels[idx]:
        ratio = (freq - GPU_freq_levels[idx - 1]) / 15
        return GPU_freq_levels[idx - 1], GPU_freq_levels[idx], ratio
    elif freq > GPU_freq_levels[idx]:
        ratio = (freq - GPU_freq_levels[idx]) / 15
        return GPU_freq_levels[idx], GPU_freq_levels[idx + 1], ratio
    else:
        return GPU_freq_levels[idx], GPU_freq_levels[idx], 1


def get_power_target(n_gpu):
    return 600 + n_gpu * 150


@dataclass(frozen=True)
class SensorSnapshot:
    ipmi_state: list
    fan_state: list
    timestamp: float
    error: Optional[str] = None


class IpmiFanFeatureCache:
    """
    Background cache for the slow `ipmitool sensor` data.

    GPU and CPU values are still sampled live because they are comparatively
    cheap and change quickly. IPMI temperature and fan values are refreshed in
    one background subprocess call and reused by update_server_features().
    """

    def __init__(self, refresh_interval_s=2.0, command=None, timeout_s=10.0):
        self.refresh_interval_s = refresh_interval_s
        self.command = command or ["ipmitool", "sensor"]
        self.timeout_s = timeout_s
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._ready_event = threading.Event()
        self._thread = None
        self._snapshot = None

    def start(self, wait_initial=True, initial_timeout_s=None):
        if self._thread is not None and self._thread.is_alive():
            if wait_initial:
                self._ready_event.wait(initial_timeout_s)
            return self

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._refresh_loop,
            name="ipmi-fan-feature-cache",
            daemon=True,
        )
        self._thread.start()

        if wait_initial:
            self._ready_event.wait(initial_timeout_s)
        return self

    def stop(self, timeout_s=5.0):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout_s)

    def latest(self, max_age_s=None):
        with self._lock:
            snapshot = self._snapshot

        if snapshot is None:
            raise RuntimeError(
                "IPMI/fan feature cache has no sample yet. "
                "Call start_server_feature_cache(wait_initial=True) at program startup."
            )

        if max_age_s is not None and time.time() - snapshot.timestamp > max_age_s:
            raise RuntimeError(
                f"IPMI/fan feature cache is stale: "
                f"{time.time() - snapshot.timestamp:.2f}s old."
            )

        return snapshot

    def _refresh_loop(self):
        while not self._stop_event.is_set():
            snapshot = self._collect_once()
            with self._lock:
                self._snapshot = snapshot
            self._ready_event.set()
            self._stop_event.wait(self.refresh_interval_s)

    def _collect_once(self):
        try:
            output = subprocess.check_output(
                self.command,
                text=True,
                timeout=self.timeout_s,
            )
            ipmi_state, fan_state = parse_ipmitool_sensor_output(output)
            return SensorSnapshot(
                ipmi_state=ipmi_state,
                fan_state=fan_state,
                timestamp=time.time(),
            )
        except Exception as exc:
            previous = self._snapshot
            if previous is not None:
                return SensorSnapshot(
                    ipmi_state=previous.ipmi_state,
                    fan_state=previous.fan_state,
                    timestamp=previous.timestamp,
                    error=str(exc),
                )
            return SensorSnapshot(
                ipmi_state=[np.nan] * len(IPMI_KEYS),
                fan_state=[np.nan] * len(FAN_KEYS),
                timestamp=time.time(),
                error=str(exc),
            )


def parse_ipmitool_sensor_output(output):
    ipmi_values = {key: np.nan for key in IPMI_KEYS}
    fan_values = {key: np.nan for key in FAN_KEYS}

    for line in output.splitlines():
        fields = [x.strip() for x in line.split("|")]
        if len(fields) < 2:
            continue

        name = fields[0]
        try:
            value = float(fields[1])
        except ValueError:
            value = np.nan

        if name in SENSOR_MAP:
            ipmi_values[SENSOR_MAP[name]] = value
        elif name in fan_values:
            fan_values[name] = value

    return (
        [ipmi_values[key] for key in IPMI_KEYS],
        [fan_values[key] for key in FAN_KEYS],
    )


def get_gpu_features(handle):
    util = nvmlDeviceGetUtilizationRates(handle)

    return [
        nvmlDeviceGetClockInfo(handle, NVML_CLOCK_SM),
        util.gpu,
        util.memory,
    ]


def get_cpu_features():
    freq = psutil.cpu_freq()
    # Non-blocking since last call; seed once at import/startup.
    util = psutil.cpu_percent(interval=None)
    return [freq.current, float(util)]


_DEFAULT_CACHE = IpmiFanFeatureCache()
# Prime cpu_percent so the first control-loop sample is meaningful.
psutil.cpu_percent(interval=None)


def start_server_feature_cache(
        refresh_interval_s=2.0,
        wait_initial=True,
        initial_timeout_s=15.0,
):
    global _DEFAULT_CACHE

    _DEFAULT_CACHE.refresh_interval_s = refresh_interval_s
    return _DEFAULT_CACHE.start(
        wait_initial=wait_initial,
        initial_timeout_s=initial_timeout_s,
    )


def stop_server_feature_cache():
    _DEFAULT_CACHE.stop()


def get_cached_ipmi_features(max_age_s=None):
    return list(_DEFAULT_CACHE.latest(max_age_s=max_age_s).ipmi_state)


def get_cached_fan_speeds(max_age_s=None):
    return list(_DEFAULT_CACHE.latest(max_age_s=max_age_s).fan_state)


def update_server_features(n_gpu, max_sensor_age_s=None):
    nvmlInit()
    try:
        gpu_state = []
        for gpu_id in range(n_gpu):
            handle = nvmlDeviceGetHandleByIndex(gpu_id)
            gpu_state.extend(get_gpu_features(handle))
    finally:
        nvmlShutdown()

    cpu_state = get_cpu_features()
    # Feature order must match pac_features.build_feature_column(..., include_ipmi_temps=False):
    # GPU feats + cpu_freq/cpu_util + optional fans (IPMI temps excluded from PAC state).
    state = gpu_state + cpu_state
    if N_FANS > 0:
        snapshot = _DEFAULT_CACHE.latest(max_age_s=max_sensor_age_s)
        state = state + list(snapshot.fan_state)[:N_FANS]
    return state


atexit.register(stop_server_feature_cache)


def compute_power(delta_gpu_freqs, delta_cpu_freqs, current_power,
                  gpu_coefs, cpu_coef):
    return (sum(d * c for d, c in zip(delta_gpu_freqs, gpu_coefs))
            + delta_cpu_freqs * cpu_coef + current_power)


def get_dynamic_bounds(GPU_freqs, CPU_freq):
    n_gpu = len(GPU_freqs)
    lower_bounds = []
    upper_bounds = []

    for _ in range(CONTROL_HORIZON):

        for i in range(n_gpu):
            lower_bounds.append(min_GPU_freq - GPU_freqs[i])
            upper_bounds.append(max_GPU_freq - GPU_freqs[i])

        lower_bounds.append(min_CPU_freq - CPU_freq)
        upper_bounds.append(max_CPU_freq - CPU_freq)

    return np.array(lower_bounds), np.array(upper_bounds)


def conservative_control(current_GPU_freqs, current_CPU_freqs):
    cpu_ctl = min_CPU_freq - current_CPU_freqs
    return np.array([min_GPU_freq - f for f in current_GPU_freqs] + [cpu_ctl])


def mpc(current_power, target, current_GPU_freqs, current_CPU_freq,
        gpu_coefs, cpu_coef):
    n_gpu = len(current_GPU_freqs)
    n_cpu = 1
    PROCESSOR_NUM = n_gpu + n_cpu

    current_GPU_freqs = np.asarray(current_GPU_freqs, dtype=float)
    current_CPU_freq = float(current_CPU_freq)

    def objective(x):
        control_sequence = x.reshape((CONTROL_HORIZON, PROCESSOR_NUM))
        predicted_GPU_freqs = current_GPU_freqs.copy()
        predicted_CPU_freq = current_CPU_freq
        cost = 0
        for k in range(PREDICTION_HORIZON):
            delta = control_sequence[k] if k < CONTROL_HORIZON else np.zeros(PROCESSOR_NUM)
            delta_gpu = delta[:n_gpu]
            delta_cpu = delta[n_gpu]

            predicted_GPU_freqs = np.clip(
                predicted_GPU_freqs + delta_gpu,
                min_GPU_freq,
                max_GPU_freq
            )

            predicted_CPU_freq = np.clip(
                predicted_CPU_freq + delta_cpu,
                min_CPU_freq,
                max_CPU_freq
            )
            gpu_delta = predicted_GPU_freqs - current_GPU_freqs
            cpu_delta = predicted_CPU_freq - current_CPU_freq

            mpc_power = compute_power(
                gpu_delta, cpu_delta, current_power, gpu_coefs, cpu_coef)
            cost += Q * (mpc_power - target) ** 2
            if k < CONTROL_HORIZON:
                cost += R * (np.sum(delta_gpu ** 2) + delta_cpu ** 2)
        return cost

    x0 = np.zeros(CONTROL_HORIZON * PROCESSOR_NUM)
    lower_bounds, upper_bounds = get_dynamic_bounds(current_GPU_freqs,
                                                    current_CPU_freq)
    bounds = list(zip(lower_bounds, upper_bounds))
    result = minimize(objective, x0, bounds=bounds, method='SLSQP',
                      options={'maxiter': 3000})
    if result.success:
        return result.x[:PROCESSOR_NUM]
    else:
        # print("Optimization failed:", result.message)
        return np.zeros(PROCESSOR_NUM)


def mpc_with_pac(current_state, current_GPU_freqs, current_CPU_freqs,
                 power_meter, pred_period, gpu_coefs, cpu_coefs,
                 target, delta_period, EPSILON_STAR, cold_start=False):
    if cold_start:
        print(
            f"====cold start: trust power meter (skip PAC), "
            f"power_meter={power_meter:.2f}"
        )
        u = mpc(power_meter, target, current_GPU_freqs,
                current_CPU_freqs, gpu_coefs, cpu_coefs)
        return u, 0, "MPC"

    u = None
    if pred_period >= delta_period:
        print("====Switch to CON: prediction used too long")
        u = conservative_control(current_GPU_freqs, current_CPU_freqs)
        return u, 0, "CON"

    # power_pred_gpu, pac_piece = predict(current_state)
    PAC_POWER_OFFSET = -11.0

    power_pred_gpu, pac_piece = predict(current_state)
    if power_pred_gpu < 0:
        print(f"====Switch to CON: power_pred_gpus{power_pred_gpu:.2f} < 0")
        u = conservative_control(current_GPU_freqs, current_CPU_freqs)
        return u, 0, "CON"
    power_pred_gpu += PAC_POWER_OFFSET
    piece_error = float(pac_piece["epsilon"]) if pac_piece is not None else 1000
    if piece_error > EPSILON_STAR:
        print(f"====Switch to CON: piece_error={piece_error:.3f} > {EPSILON_STAR}")
        u = conservative_control(current_GPU_freqs, current_CPU_freqs)
        return u, 0, "CON"
    if np.abs(power_meter - power_pred_gpu) > EPSILON_STAR:
        u = mpc(power_pred_gpu, target, current_GPU_freqs,
                current_CPU_freqs, gpu_coefs, cpu_coefs)
        print(f"====PAC: power_meter={power_meter:.2f}, pred={power_pred_gpu:.2f}")
        return u, pred_period + 1, "PAC"
    else:
        print(f"====trust the power meter: power_meter={power_meter:.2f}, pred={power_pred_gpu:.2f}")
        u = mpc(power_meter, target, current_GPU_freqs,
                current_CPU_freqs, gpu_coefs, cpu_coefs)
        return u, 0, "MPC"


class RandomWalkKalman:
    def __init__(self, Q=0.2, R=4.0, initial_power=None, initial_P=1.0):
        self.Q = Q  # Process noise variance
        self.R = R  # Measurement noise variance
        self.P = initial_P  # Estimation covariance
        self.x = initial_power  # Current estimate

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


def load_model(model_path):
    with open(model_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    pieces = []
    for p in data:
        pieces.append({
            "bounds": [tuple(map(float, b)) for b in p["bounds"]],
            "w": np.asarray(p["w"], dtype=float),
            "epsilon": float(p.get("epsilon", 0.0)),
            "n_sample": int(p.get("n_sample", len(p.get("idx", [])))),
            "idx": np.asarray(p.get("idx", []), dtype=int),
        })
    return pieces


def select_piece(xs, pieces):
    xs = np.asarray(xs, dtype=float).ravel()
    tol = np.maximum(0.1 * (x_max - x_min), 1e-6)
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
    if not models:
        print(f"[Warning] No matched model: {xs}")
        return None
        # if not models:
    #     # print("\n=== DEBUG: no matched piece ===")
    #     # print("xs:", xs)
    #     for p_idx, p in enumerate(pieces[:5]):
    #         # print(f"\n-- piece {p_idx} --")
    #         for i in range(d):
    #             lo, hi = p["bounds"][i]
    #             tol = tol_vec[i]
    #             if not (lo - tol <= xs[i] <= hi + tol):
    #                 # print(f"[FAIL] dim={i}, xs={xs[i]:.6f}, "
    #                 #   f"range=[{lo:.6f},{hi:.6f}], tol={tol:.6f}")
    #                 break
    #     raise ValueError(f"not able to find a matched model: {xs}")
    return min(models, key=lambda p: p["epsilon"])


def constrained_pac(pac_path, n_gpu):
    global pieces, piece_lo, piece_hi, x_min, x_max
    pieces = load_model(pac_path)
    piece_lo = np.array([min(p["bounds"][i][0] for p in pieces)
                         for i in range(len(pieces[0]["bounds"]))], dtype=float)
    piece_hi = np.array([max(p["bounds"][i][1] for p in pieces)
                         for i in range(len(pieces[0]["bounds"]))], dtype=float)
    x_min = piece_lo.copy()
    x_max = piece_hi.copy()
    for idx in gpu_freq_indices(n_gpu, N_FANS):
        x_min[idx] = min_GPU_freq
        x_max[idx] = max_GPU_freq
    cpu_idx = cpu_freq_index(n_gpu, N_FANS)
    x_min[cpu_idx] = min_CPU_freq
    x_max[cpu_idx] = max_CPU_freq


def predict(xs):
    xs = np.asarray(xs, dtype=float).ravel()
    p = select_piece(xs, pieces)
    if p is None:
        # print("Not able to select!")
        return -10000, None
    w = p["w"]
    # print(f'model is w={w}, epsilon is {p["epsilon"]:.3f}')
    if w.size != xs.size + 1:
        # raise ValueError(f"w size mismatch: {w.size} vs {xs.size + 1}")
        print(f"[Warning] w size mismatch: {w.size} vs {xs.size + 1}")
        return -10000, None
    y_pred = float(w[0] + np.dot(w[1:], xs))
    return y_pred, p


def apply_power_error(power, error_type, n_gpu, P_min=350, P_max=2000):
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
            fake_power = power - np.random.uniform(1000, 5000)
        return np.clip(fake_power, P_min, P_max)
    elif error_type == "none":
        return power
    else:
        print(f"[Warning] Unknown error type: {error_type}, use original power.")
        return power


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
        "npac": int(info["npac"]),
        "path": path,
    }


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
    return obj


start_server_feature_cache()
INIT_FREQ = 705

steps = 49
runs = 6
ControlMethod = ["mov_avg"]
# ControlMethod = ["standalone"]
# ControlMethod = ["kalman", "mov_avg"]
# ControlMethod = ["pac"]
np.random.seed(0)

# error_types = ["offset", "mul", "stuck_min", "stuck_max", "out_of_range"]
error_types = ["offset"]
GPU_LIST = [1]

all_pac_files = glob.glob("PAC_PIECE/*.json")
pac_db = []
for p in all_pac_files:
    info = parse_pac_filename(p)
    if info is not None:
        pac_db.append(info)

all_results = []
true_all_results = []

for n_gpu in GPU_LIST:
    log_file = open(f"rt_{n_gpu}gpu_PAC_updated.log", "w")

    sys.stdout = log_file

    sys.stderr = log_file

    global PROCESSOR_NUM
    PROCESSOR_NUM = n_gpu

    GFLOPS_PER_BATCH = 163.56737024
    GFLOPS_PER_IMAGE = 8.178368512
    e_imax = 21.495299657282743
    f_max = 1380
    alpha = 0.9319116344447579

    if n_gpu == 1:
        kf = RandomWalkKalman(Q=2.44, R=4.28)
    elif n_gpu == 2:
        kf = RandomWalkKalman(Q=13.03, R=6.45)
    elif n_gpu == 4:
        kf = RandomWalkKalman(Q=51.53, R=59.92)

    process_measure = subprocess.Popen(
        ['python3', '/home/pacs/Yuan/mpc/wattsup_unm_acpi_5_2.py', 'tmp_5_2'])

    gpu_coefs = COEFFICIENTS[n_gpu]["gpu"]
    cpu_coefs = COEFFICIENTS[n_gpu]["cpu"]

    P_target = get_power_target(n_gpu)
    print(f"\n===== Running GPU = {n_gpu} =====")

    for method in ControlMethod:
        if method == "pac":
            model_list = [m for m in pac_db if m["ngpu"] == n_gpu]
        else:
            model_list = [None]

        for pac_m in model_list:
            if pac_m is not None:
                constrained_pac(pac_m["path"], n_gpu)
                EPSILON_STAR = pac_m["eps"]
                pac_tag = os.path.basename(pac_m["path"])
            else:
                pac_tag = "none"

            for error_rate in [0.1, 0.5]:
                print(f"\n=== Error_rate={error_rate} ===")
                for error in error_types:
                    for run in range(runs):
                        nvmlInit()
                        gpu_state = []
                        for gpu_id in range(n_gpu):
                            handle = nvmlDeviceGetHandleByIndex(gpu_id)
                            nvmlDeviceSetApplicationsClocks(handle, 877, INIT_FREQ)
                        nvmlShutdown()

                        time.sleep(5)

                        init_state = update_server_features(n_gpu)

                        print("Initial state:", [round(x, 2) for x in init_state])
                        csv_dir = "./CSV_LOGS"
                        os.makedirs(csv_dir, exist_ok=True)

                        # --- existing per-step power/freq CSV ---
                        csv_name = (
                            f"et_gpu_power_trace_"
                            f"gpu_{n_gpu}"
                            f"error_rate_{error_rate}_"
                            f"run_{run}_"
                            f"error_{error}_"
                            f"method_{method}.csv"
                        )
                        csv_path = os.path.join(csv_dir, csv_name)

                        # --- NEW: per-inference latency CSV ---
                        lat_csv_name = (
                            f"et_latency_per_period_"
                            f"{n_gpu}gpu_"
                            f"error_rate_{error_rate}_"
                            f"method_{method}_"
                            f"error_{error}_"
                            f"run_{run}.csv"
                        )
                        lat_csv_path = os.path.join(csv_dir, lat_csv_name)

                        freq_log_rows = []
                        latency_log_rows = []  # <-- accumulates per-inference records

                        print(f"\n=======this is #{run + 1} run:==========")
                        error_mask = np.random.rand(steps) < error_rate
                        num_pac = num_mpc = num_unsafe = num_con = 0
                        powercontrol = []  # actual power series
                        power_meter_series = []  # power meter reading series
                        mode_series = []

                        current_GPU_freqs = [
                            init_state[FEATURES_PER_GPU * i] for i in range(n_gpu)
                        ]
                        current_CPU_freq = init_state[FEATURES_PER_GPU * n_gpu]
                        pred_period = 0

                        clean_record()
                        time.sleep(5)
                        # Simulate multiple control periods
                        for step in range(steps):
                            power = avg('/home/pacs/Yuan/mpc/logs/tmp_5_2')
                            clean_record()
                            full_state = update_server_features(n_gpu)
                            # Zhuo: below here simulates the power meter error
                            hasError = False if step == 0 else error_mask[step]
                            print(f"===current power is {power:.3f},hasError? {hasError}")
                            print("Current state:", [round(x, 2) for x in full_state])

                            if step == 0:
                                power_meter = power
                            elif hasError and error != "none":
                                power_meter = apply_power_error(power, error, n_gpu)
                            else:
                                power_meter = power

                            powercontrol.append(power)
                            power_meter_series.append(power_meter)

                            if power > P_target + 25 * n_gpu:
                                num_unsafe += 1
                            print(
                                f"power meter reading: {power_meter:.2f}w,"
                                f" actual power {power:.2f}w, target {P_target}w, "
                                f"safe {P_target + 25 * n_gpu}w, {num_unsafe}/{step}")
                            if method == "pac":
                                u, pred_period, mode = mpc_with_pac(
                                    current_state=full_state,
                                    current_GPU_freqs=current_GPU_freqs,
                                    current_CPU_freqs=current_CPU_freq,
                                    power_meter=power_meter,
                                    pred_period=pred_period,
                                    gpu_coefs=gpu_coefs,
                                    cpu_coefs=cpu_coefs,
                                    target=P_target,
                                    delta_period=10,
                                    EPSILON_STAR=EPSILON_STAR,
                                    cold_start=(step == 0),
                                )

                                if mode == "PAC":
                                    num_pac += 1
                                elif mode == "MPC":
                                    num_mpc += 1
                                else:
                                    num_con += 1
                                mode_series.append(mode)

                            elif method == "standalone":
                                mode = "MPC"
                                u = mpc(power_meter, P_target, current_GPU_freqs,
                                        current_CPU_freq, gpu_coefs, cpu_coefs)

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

                            gpu_ids_active = list(range(n_gpu))
                            clear_latency_logs(gpu_ids_active)

                            processes = []
                            for gpu_id in range(n_gpu):
                                f1, f2, ratio = snap_gpu_freq_delta_sigma(
                                    current_GPU_freqs[gpu_id])
                                p = multiprocessing.Process(
                                    target=delta_sigma_GPU,
                                    args=(f1, f2, ratio, gpu_id),
                                )
                                processes.append(p)
                            for p in processes:
                                p.start()
                            for p in processes:
                                p.join()

                            # Collect all latencies observed during this period
                            period_records = read_latency_logs(gpu_ids_active)
                            freq_snapshot = {
                                f"gpu{i}_freq_at_step": float(current_GPU_freqs[i])
                                for i in range(n_gpu)
                            }
                            for rec in period_records:
                                lat_row = {
                                    "step": step,
                                    "n_gpu": n_gpu,
                                    "method": method,
                                    "error_type": error,
                                    "error_rate": error_rate,
                                    "run": run,
                                    "gpu_id": rec["gpu_id"],
                                    "timestamp": rec["timestamp"],
                                    "latency_ms": rec["latency_ms"],
                                    "power": float(power),
                                    "mode": mode if "mode" in locals() else "MPC",
                                }
                                lat_row.update(freq_snapshot)
                                latency_log_rows.append(lat_row)

                            print(f"  step {step}: collected "
                                  f"{len(period_records)} latency samples")
                            print(
                                f"Control Actions: ΔGPU = {[f'{d:.2f}' for d in u[:-1]]} MHz, "
                                f"target freq = {[f'{f:.2f}' for f in current_GPU_freqs]} MHz\n"
                            )
                            if power >= P_target:
                                print(f"Target power {P_target} W reached at step {step + 1}.\n")

                        # Save latency-per-period CSV (new)
                        if latency_log_rows:
                            lat_df = pd.DataFrame(latency_log_rows)
                            lat_df.to_csv(lat_csv_path, index=False)
                            print(f"Saved {len(latency_log_rows)} latency "
                                  f"records to {lat_csv_path}")
                        else:
                            print(f"[WARNING] No latency records collected. "
                                  f"Is workload_runner.py running?")

                        power_arr = np.array(powercontrol)
                        power_mean = float(np.mean(power_arr))
                        power_std = float(np.std(power_arr))
                        max_overshoot = np.max(powercontrol) - P_target
                        counts = {"PAC": num_pac, "MPC": num_mpc, "CON": num_con}

                        mode_per = {k: (counts[k] / steps) for k in counts}

                        result = {
                            "gpu_num": n_gpu, "method": method,
                            "pac_model": pac_tag, "error_rate": error_rate,
                            "error_type": error, "run": int(run),
                            "steps": int(steps),

                            "power_series": powercontrol,
                            "power_meter_series": power_meter_series,
                            "EPSILON_STAR": EPSILON_STAR if method == "pac" else None,

                            "power_mean": power_mean, "power_std": power_std,
                            "max_overshoot": max_overshoot,
                            "r_unsafe": num_unsafe / steps,
                            "mode_counts": counts,
                        }
                        all_results.append(result)
                        true_all_results.append(result)

                all_results = to_python(all_results)
                out_path = f"json_results/control_results_all_{n_gpu}_{method}_{error_rate}_{error}_server.json"
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(all_results, f, ensure_ascii=False, indent=2)
                print(f"Saved {len(all_results)} records to {out_path}")
                all_results = []
    process_measure.kill()

summary_results = []

grouped = defaultdict(list)

for r in true_all_results:
    key = (
        r["gpu_num"],
        r["method"],
        r["pac_model"],
        r["error_rate"],
        r["error_type"]
    )
    grouped[key].append(r)

for key, runs_data in grouped.items():
    gpu_num, method, pac_model, error_rate, error_type = key

    avg_power_mean = np.mean([x["power_mean"] for x in runs_data])
    avg_power_std = np.mean([x["power_std"] for x in runs_data])
    avg_max_overshoot = np.mean([x["max_overshoot"] for x in runs_data])
    avg_r_unsafe = np.mean([x["r_unsafe"] for x in runs_data])

    avg_mode_counts = {
        "PAC": np.mean([x["mode_counts"]["PAC"] for x in runs_data]),
        "MPC": np.mean([x["mode_counts"]["MPC"] for x in runs_data]),
        "CON": np.mean([x["mode_counts"]["CON"] for x in runs_data]),
    }

    summary = {
        "gpu_num": gpu_num,
        "method": method,
        "pac_model": pac_model,
        "error_rate": error_rate,
        "error_type": error_type,

        "num_runs": len(runs_data),

        "avg_power_mean": float(avg_power_mean),
        "avg_power_std": float(avg_power_std),
        "avg_max_overshoot": float(avg_max_overshoot),
        "avg_r_unsafe": float(avg_r_unsafe),

        "avg_mode_counts": {
            k: float(v) for k, v in avg_mode_counts.items()
        }
    }

    summary_results.append(summary)

true_all_results = to_python(true_all_results)

out_path = f"control_results_all_Server.json"

with open(out_path, "w", encoding="utf-8") as f:
    json.dump(true_all_results, f, ensure_ascii=False, indent=2)

print(f"Saved {len(true_all_results)} records to {out_path}")

summary_results = to_python(summary_results)

summary_path = f"control_results_summary_Server.json"

with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(summary_results, f, ensure_ascii=False, indent=2)

print(f"Saved {len(summary_results)} summary records to {summary_path}")

log_file.close()
