import os
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(SCRIPT_DIR, "Simulator1_gpu4_80000.csv")

df = pd.read_csv(CSV)

segments = df.groupby(["gpu1_freq", "gpu2_freq", "gpu3_freq", "gpu4_freq", "cpu_freq"])

all_delta = []
segment_vars = []

for _, seg in segments:

    power = seg["total_power"].to_numpy()

    if len(power) < 2:
        continue

    all_delta.extend(np.diff(power))

    segment_vars.append(np.var(power, ddof=1))

Q = np.var(all_delta, ddof=1)
R = np.mean(segment_vars)

print(f"Q {Q:.2f}")
print(f"R {R:.2f}")
print(f"Q std {np.sqrt(Q):.2f}")
print(f"R std {np.sqrt(R):.2f}")
