"""Build the two control-result tables from JSON_RES files.

Usage:
    python make_control_tables.py JSON_DIR OUT_PREFIX

Example:
    python make_control_tables.py Simulator1/JSON_RES Simulator1/sim_i

Writes OUT_PREFIX_control_table.csv and OUT_PREFIX_modes_table.csv.
"""

import csv
import glob
import json
import os
import re
import sys

ERRORS = ["offset", "mul", "stuck_min", "stuck_max", "out_of_range"]
# Power table, left to right inside each GPU group.
POWER_METHODS = [
    ("pac", "Our Method"),
    ("mpc", "Original Controller"),
    ("mov_avg", "MA"),
    ("con", "Immediate Conservative"),
]
NGPUS = [1, 2, 4]
RATES = [0.1, 0.5]

NAME = re.compile(
    r"control_results_NGPU(?P<ngpu>\d+)"
    r"_method-(?P<method>[a-z_]+)"
    r"_err-(?P<error>[a-z_]+)"
    r"_rate-(?P<rate>[0-9.]+)"
    r"(?P<rest>.*)\.json$"
)


def power_set(n_gpu):
    return 600 + 150 * n_gpu


def use_this_model(rest):
    """mpc / mov_avg have no model suffix. pac / con keep m=1 residual_range.

    When the filename also carries alpha, keep alpha = 0.05 only.
    """
    if rest == "":
        return True
    if "_m1_" not in rest or "residual_range" not in rest:
        return False
    if "a0." in rest and "a0.05" not in rest:
        return False
    return True


def collect(json_dir):
    """Map (ngpu, method, error, rate) to one JSON path. Newest file wins."""
    chosen = {}
    for path in glob.glob(os.path.join(json_dir, "control_results_NGPU*.json")):
        match = NAME.match(os.path.basename(path))
        if match is None or not use_this_model(match.group("rest")):
            continue
        key = (
            int(match.group("ngpu")),
            match.group("method"),
            match.group("error"),
            float(match.group("rate")),
        )
        if key not in chosen or os.path.getmtime(path) > os.path.getmtime(chosen[key]):
            chosen[key] = path
    return chosen


def load_overall(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data["overall"], data["steps"], data["n_runs"]


def require(chosen, n_gpu, method, error, rate):
    key = (n_gpu, method, error, rate)
    if key not in chosen:
        raise SystemExit(f"missing result for NGPU={n_gpu} {method} {error} rate={rate}")
    return load_overall(chosen[key])


def write_control_table(chosen, path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "error_rate",
                "row",
                "error_type",
                "ngpu",
                "P_set",
                "method",
                "P_bar_W",
                "unsafe_pct",
                "steps",
                "n_runs",
            ]
        )
        for rate in RATES:
            for row, error in enumerate(ERRORS, start=1):
                for n_gpu in NGPUS:
                    for method, label in POWER_METHODS:
                        overall, steps, n_runs = require(chosen, n_gpu, method, error, rate)
                        writer.writerow(
                            [
                                rate,
                                row,
                                error,
                                n_gpu,
                                power_set(n_gpu),
                                label,
                                f"{overall['power_mean']:.1f}",
                                f"{overall['r_unsafe'] * 100:.2f}",
                                steps,
                                n_runs,
                            ]
                        )


def write_modes_table(chosen, path):
    """Columns per GPU: prediction, meter, Our conservative, Immediate Conservative."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "error_rate",
                "row",
                "error_type",
                "ngpu",
                "method",
                "pred_pct",
                "meter_pct",
                "con_pct",
                "steps",
                "n_runs",
            ]
        )
        for rate in RATES:
            for row, error in enumerate(ERRORS, start=1):
                for n_gpu in NGPUS:
                    overall, steps, n_runs = require(chosen, n_gpu, "pac", error, rate)
                    rates = overall["mode_rates"]
                    writer.writerow(
                        [
                            rate,
                            row,
                            error,
                            n_gpu,
                            "Our Method",
                            f"{rates['PAC'] * 100:.2f}",
                            f"{rates['MPC'] * 100:.2f}",
                            f"{rates['CON'] * 100:.2f}",
                            steps,
                            n_runs,
                        ]
                    )
                    overall, steps, n_runs = require(chosen, n_gpu, "con", error, rate)
                    writer.writerow(
                        [
                            rate,
                            row,
                            error,
                            n_gpu,
                            "Immediate Conservative",
                            "",
                            "",
                            f"{overall['mode_rates']['CON'] * 100:.2f}",
                            steps,
                            n_runs,
                        ]
                    )


def main():
    if len(sys.argv) != 3:
        print("Usage: python make_control_tables.py JSON_DIR OUT_PREFIX")
        print("Example: python make_control_tables.py Simulator1/JSON_RES Simulator1/sim_i")
        sys.exit(1)

    json_dir, prefix = sys.argv[1], sys.argv[2]
    chosen = collect(json_dir)
    control_path = prefix + "_control_table.csv"
    modes_path = prefix + "_modes_table.csv"
    write_control_table(chosen, control_path)
    write_modes_table(chosen, modes_path)
    print(f"wrote {control_path}")
    print(f"wrote {modes_path}")


if __name__ == "__main__":
    main()
