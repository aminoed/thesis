import json

import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Define constants for the model
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

min_CPU_freq, max_CPU_freq = 1100, 2100
min_GPU_freq, max_GPU_freq = 495, 1380

FREQUENCY_STEP_MHZ = 15
GPU_FREQUENCY_LEVELS = np.arange(
    min_GPU_freq, max_GPU_freq + 1, FREQUENCY_STEP_MHZ, dtype=float
)
CPU_FREQUENCY_LEVELS = np.arange(
    min_CPU_freq, max_CPU_freq + 1, FREQUENCY_STEP_MHZ, dtype=float
)

# Define LQR parameters
Q = 500  # State cost
R = 0.01  # Control effort cost


def get_power_target(n_gpu):
    return 600 + n_gpu * 150


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

    if result.success:
        # Extract the first control action
        return result.x[:PROCESSOR_NUM]
    else:
        print("Optimization failed:", result.message)
        return np.zeros(PROCESSOR_NUM)


def plant_power(gpu_freqs, cpu_freq, gpu_coefs, cpu_coef):
    return sum(f * c for f, c in zip(gpu_freqs, gpu_coefs)) + cpu_freq * cpu_coef


def get_safe_power(n_gpu):
    return 600 + n_gpu * 175


def sample_initial_state(n_gpu, rng):
    gpu_coefs = COEFFICIENTS[n_gpu]["gpu"]
    cpu_coef = COEFFICIENTS[n_gpu]["cpu"]
    safe_power = get_safe_power(n_gpu)

    while True:
        gpu_freqs = rng.choice(GPU_FREQUENCY_LEVELS, size=n_gpu).tolist()
        cpu_freq = float(rng.choice(CPU_FREQUENCY_LEVELS))
        power = plant_power(gpu_freqs, cpu_freq, gpu_coefs, cpu_coef)
        if power <= safe_power:
            return gpu_freqs, cpu_freq


def snap_to_frequency(freq, levels):
    return float(levels[np.abs(levels - freq).argmin()])


def apply_frequency_update(gpu_freqs, cpu_freq, u):
    n_gpu = len(gpu_freqs)
    next_gpu_freqs = [
        snap_to_frequency(gpu_freqs[i] + u[i], GPU_FREQUENCY_LEVELS)
        for i in range(n_gpu)
    ]
    next_cpu_freq = snap_to_frequency(cpu_freq + u[-1], CPU_FREQUENCY_LEVELS)
    return next_gpu_freqs, next_cpu_freq


def collect_one_end_sample(n_gpu, gpu_coefs, cpu_coef, P_target, rng):
    gpu_freqs, cpu_freq = sample_initial_state(n_gpu, rng)
    power = plant_power(gpu_freqs, cpu_freq, gpu_coefs, cpu_coef)

    u = mpc(power, P_target, gpu_freqs, cpu_freq, gpu_coefs, cpu_coef)
    gpu_freqs, cpu_freq = apply_frequency_update(gpu_freqs, cpu_freq, u)

    end_power = plant_power(gpu_freqs, cpu_freq, gpu_coefs, cpu_coef)
    return gpu_freqs + [cpu_freq, end_power]


def generate_data(target_size, n_gpu, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    gpu_coefs = COEFFICIENTS[n_gpu]["gpu"]
    cpu_coef = COEFFICIENTS[n_gpu]["cpu"]
    P_target = get_power_target(n_gpu)

    dataset = [
        collect_one_end_sample(n_gpu, gpu_coefs, cpu_coef, P_target, rng)
        for _ in range(target_size)
    ]

    columns = [f"gpu{i + 1}_freq" for i in range(n_gpu)] + ["cpu_freq", "total_power"]
    return pd.DataFrame(dataset, columns=columns)


rng = np.random.default_rng(0)
targets = 20000

for n_gpu in [1, 2, 4]:
    n_samples = n_gpu * targets
    df = generate_data(n_samples, n_gpu, rng)

    filename = f"Simulator1_gpu{n_gpu}_{n_samples}.csv"
    df.to_csv(filename, index=False)

    print(f"Completed: {filename}, sample_size: {len(df)}")
