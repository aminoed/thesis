import json
import re
from pathlib import Path

import pandas as pd

PAC_DIR = Path("PAC_PERF")
# pac_build_NGPU2_a0.05_b1e-08_k40000_e30_m1_maximum_residual_NPAC1377.json
FNAME_RE = re.compile(
    r"pac_build_NGPU(?P<ngpu>\d+)_a(?P<alpha>[\d.eE+-]+)_b(?P<beta>[\d.eE+-]+)"
    r"_k(?P<k>\d+)_e(?P<eps>[\d.eE+-]+)_m(?P<m>\d+)_s?"
    r"(?P<split>residual_range|maximum_residual)"
    r"_NPAC(?P<npac>\d+)\.json$"
)


def parse_meta_from_filename(p: Path):
    m = FNAME_RE.search(p.name)
    if not m:
        return {}
    g = m.groupdict()
    return {
        "NGPU": int(g["ngpu"]),
        "alpha": float(g["alpha"]),
        "beta": float(g["beta"]),
        "k_total": int(g["k"]),
        "eps_star": float(g["eps"]),
        "m": int(g["m"]),
        "split": g["split"],
        "NPAC": int(g["npac"]),
    }


def pick(d, *keys, default=None):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


def load_one(path: Path):
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    t_total = float(pick(data, "t_total", "t_build_total", default=float("nan")))
    lp_count = int(pick(data, "lp_count", "lp_solve_count", default=0))
    lp_time_total = float(pick(data, "lp_time", "lp_time_total", default=0.0))
    lp_time_avg = pick(data, "lp_time_avg", default=None)
    if lp_time_avg is None:
        lp_time_avg = (lp_time_total / lp_count) if lp_count else float("nan")
    else:
        lp_time_avg = float(lp_time_avg)

    split_num = int(pick(data, "split_num", "split_candidates", default=0))
    split_time = float(pick(data, "split_time", "split_scan_time_total", default=0.0))
    n_pieces = int(pick(data, "n_pieces", default=0))
    n_splits = int(pick(data, "n_splits", default=max(0, n_pieces - 1)))

    row = {
        "file": path.name,
        "t_total": t_total,
        "lp_count": lp_count,
        "lp_time_total": lp_time_total,
        "lp_time_avg": lp_time_avg,
        "split_num": split_num,
        "split_time": split_time,
        "n_pieces": n_pieces,
        "n_splits": n_splits,
    }
    row.update(parse_meta_from_filename(path))
    return row


def main():
    rows = []
    json_files = list(PAC_DIR.glob("*.json"))

    if not json_files:
        print(f"[ERROR] No JSON files found in {PAC_DIR.resolve()}")
        return

    matched = 0
    skipped = []
    for p in json_files:
        if not FNAME_RE.search(p.name):
            skipped.append(p.name)
            continue
        row = load_one(p)
        rows.append(row)
        matched += 1

        print(
            "==== PAC Build Data ===\n"
            f"Total time (s):            {row['t_total']:.4f}\n"
            f"Total LP time(s):          {row['lp_time_total']:.4f}\n"
            f"LP solved:                 {row['lp_count']}\n"
            f"Avg LP solve time (s):     {row['lp_time_avg']:.4f}\n"
            f"Split times:               {row['split_num']}\n"
            f"Split time (s):            {row['split_time']:.6f}\n"
            f"Total Pieces:              {row['n_pieces']}\n"
            f"Total Splits:              {row['n_splits']}\n"
            f"File:                      {row['file']}\n"
        )

    print(f"Total json files: {len(json_files)}")
    print(f"Matched files: {matched}")
    print(f"Skipped files: {len(skipped)}")
    if skipped:
        print("Skipped (unmatched pattern):")
        for s in skipped[:10]:
            print("  ", s)

    if not rows:
        print("[ERROR] No model loaded. Check file paths / filename pattern.")
        return

    df = pd.DataFrame(rows)

    sort_cols = [c for c in ["NGPU", "alpha", "beta", "eps_star", "m", "split"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(by=sort_cols)

    out_csv = Path("pac_build_summary.csv")
    df.to_csv(out_csv, index=False)
    print(f"[saved] {out_csv.resolve()}")


if __name__ == "__main__":
    main()
