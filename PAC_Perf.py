import re
import json
import math
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any

PAC_DIR = Path("PAC_PIECE")

FNAME_RE = re.compile(
    r"pieces_NGPU(?P<ngpu>\d+)_a(?P<alpha>[\d.eE+-]+)_b(?P<beta>[\d.eE+-]+)_k(?P<k>\d+)_e(?P<eps>[\d.eE+-]+)_NPAC(?P<npac>\d+)bnb\.json$"
)


def parse_meta_from_filename(path: Path) -> Dict[str, Any]:
    m = FNAME_RE.search(path.name)
    if not m:
        raise ValueError(f"Unrecognized filename pattern: {path.name}")
    d = m.groupdict()
    return {
        "NGPU": int(d["ngpu"]),
        "alpha": float(d["alpha"]),
        "beta": float(d["beta"]),
        "k_total": int(d["k"]),
        "eps_star": float(d["eps"]),
        "NPAC": int(d["npac"]),
    }


def load_pieces(path):
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def get_n_sample(p):
    if "n_sample" in p and p["n_sample"] is not None:
        return int(p["n_sample"])
    if "idx" in p and p["idx"] is not None:
        try:
            return int(len(p["idx"]))
        except Exception:
            pass
    return None


def get_epsilon(p):
    val = p.get("epsilon", None)
    if val is None:
        raise ValueError("Missing 'epsilon' in a piece.")
    return float(val)


def describe_eps(eps_arr):
    return {
        "eps_min": float(np.min(eps_arr)) if eps_arr.size else math.nan,
        "eps_median": float(np.median(eps_arr)) if eps_arr.size else math.nan,
        "eps_p95": float(np.percentile(eps_arr, 95)) if eps_arr.size else math.nan,
        "eps_max": float(np.max(eps_arr)) if eps_arr.size else math.nan,
    }


def fmt_ratio(n, d):
    return float(n) / d if d else math.nan


def evaluate_one_model(path):
    meta = parse_meta_from_filename(path)
    pieces = load_pieces(path)
    total = len(pieces)

    eps_star = meta["eps_star"]
    npac = meta["NPAC"]

    eps_list = []
    num_lt_eps_star = 0
    num_gt_eps_star = 0

    num_ge_npac = 0
    num_lt_npac = 0

    for p in pieces:
        eps = get_epsilon(p)
        eps_list.append(eps)

        if eps <= eps_star:
            num_lt_eps_star += 1
        else:
            num_gt_eps_star += 1

        n_s = get_n_sample(p)
        if n_s is not None:
            if n_s >= npac:
                num_ge_npac += 1
            else:
                num_lt_npac += 1

        else:
            pass

    eps_arr = np.array(eps_list, dtype=float)

    out = {
        "file": path.name,
        **meta,
        "n_pieces": total,

        "n_eps_le_epsstar": num_lt_eps_star,
        "r_eps_le_epsstar": fmt_ratio(num_lt_eps_star, total),
        "n_eps_gt_epsstar": num_gt_eps_star,
        "r_eps_gt_epsstar": fmt_ratio(num_gt_eps_star, total),

        "n_ge_NPAC": num_ge_npac,
        "r_ge_NPAC": fmt_ratio(num_ge_npac, total),
        "n_lt_NPAC": num_lt_npac,
        "r_lt_NPAC": fmt_ratio(num_lt_npac, total),

    }
    out.update(describe_eps(eps_arr))
    return out


def main():
    rows = []

    if not PAC_DIR.exists():
        raise FileNotFoundError(f"{PAC_DIR} does not exist")

    all_files = list(PAC_DIR.glob("*.json"))

    matched_files = []
    skipped_files = []

    for path in all_files:
        if FNAME_RE.search(path.name):
            matched_files.append(path)
        else:
            skipped_files.append(path.name)

    print(f"Total json files: {len(all_files)}")
    print(f"Matched files: {len(matched_files)}")
    print(f"Skipped files: {len(skipped_files)}")

    if skipped_files:
        print("\nSkipped (unmatched pattern):")
        for s in skipped_files[:10]:
            print("  ", s)

    for path in matched_files:
        try:
            res = evaluate_one_model(path)
            rows.append(res)
        except Exception as e:
            print(f"Error processing {path.name}: {e}")

    if not rows:
        print("No valid data processed.")
        return

    df = pd.DataFrame(rows).sort_values(by=["alpha", "beta", "eps_star"])

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)

    print("\n=== Per-model summary ===")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    df.to_csv("pac_model_summary_bnb.csv", index=False)
    print("\nSaved to pac_model_summary.csv")


if __name__ == "__main__":
    main()
