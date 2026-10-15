import json
from collections import deque
import numpy as np
from matplotlib import pyplot as plt
from scipy import optimize
import pandas as pd
import time


class BuildStates:
    def __init__(self):
        # build time
        self.t_start = 0.0
        self.t_end = 0.0
        self.t_total = 0.0

        # lp solve time and counts
        self.lp_count = 0
        self.lp_time = 0.0

        # split time
        self.split_num = 0
        self.split_time = 0.0

        # pieces
        self.n_pieces = 0
        self.n_splits = 0

    def start_timing(self):
        self.t_start = time.perf_counter()

    def end_timing(self):
        self.t_end = time.perf_counter()
        self.t_total = self.t_end - self.t_start

    def lp_time_avg(self):
        return (self.lp_time / self.lp_count) if self.lp_count else 0.0

    def to_dict(self):
        dic = dict(self.__dict__)
        dic.update({
            "lp_time_avg": self.lp_time_avg(),
        })
        return dic


DATA = BuildStates()


def fit_linear_pac(X_sub, y_sub):
    n_sub, d = X_sub.shape
    if n_sub == 0:
        return None, None

    A = np.zeros((2 * n_sub, d + 2))
    b = np.zeros(2 * n_sub)
    for i in range(n_sub):
        xi, yi = X_sub[i], y_sub[i]

        A[2 * i] = np.hstack(([1.0], xi, [-1.0]))
        b[2 * i] = yi

        A[2 * i + 1] = np.hstack(([-1.0], -xi, [-1.0]))
        b[2 * i + 1] = -yi

    c = np.zeros(d + 2)
    c[-1] = 1.0

    bounds = [(None, None)] * (d + 2 - 1) + [(1e-4, None)]

    # time the lp
    t_s = time.perf_counter()

    res = optimize.linprog(c, A_ub=A, b_ub=b, bounds=bounds, method='highs')
    t_e = time.perf_counter()
    DATA.lp_count += 1
    DATA.lp_time += t_e - t_s

    if res.success:
        z = res.x
        w = z[:-1]
        epsilon = z[-1]
        return w, float(epsilon)
    else:
        # not solvable
        print("lingprog failed: ", res.message)
        # fallback
        X_aug = np.hstack([np.ones((X_sub.shape[0], 1)), X_sub])
        w = np.linalg.lstsq(X_aug, y_sub, rcond=None)[0]

        y_pred = X_aug @ w
        epsilon = np.max(np.abs(y_pred - y_sub))

        return w, float(epsilon)


# compute the errors within the samples
# error = |p(x) - y|
def compute_errors(X_sub, y_sub, w):
    if w is None:
        print("not solvable, w is none")
        return None
    y_pred = w[0] + X_sub.dot(w[1:])
    error = np.abs(y_pred - y_sub)
    return error


def best_split(X_sub, y_sub, par_idx, k, error, fit_idx=None):
    # Split on a sample point (not the midpoint)
    n_sub, d = X_sub.shape
    if error is None or n_sub < k:
        # only do subdivision if at least one side is greater than k
        return None

    best = None
    best_impurity = np.inf
    best_balance = np.inf

    t_s = time.perf_counter()
    temp_split = 0
    # if n_sub < 2 * k, prefer the splits where one side has exactly k samples
    split_one_side = (n_sub < 2 * k)

    if fit_idx is None:
        can_fit = np.ones(n_sub, dtype=bool)
    else:
        fit_set = set(np.asarray(fit_idx).tolist())
        can_fit = np.array([int(par_idx[t]) in fit_set for t in range(n_sub)], dtype=bool)

    for i in range(d):
        # sort by this particular dimension
        order = np.argsort(X_sub[:, i])
        xs_sort = X_sub[order, i]
        errors_sort = error[order]
        can_fit_sort = can_fit[order]

        left_max = np.maximum.accumulate(errors_sort)
        left_min = np.minimum.accumulate(errors_sort)
        right_max = np.maximum.accumulate(errors_sort[::-1])[::-1]
        right_min = np.minimum.accumulate(errors_sort[::-1])[::-1]

        # cut on sample j (threshold = xs_sort[j]); j itself is excluded from
        # left/right fit counts and residual-range
        for j in range(len(xs_sort) - 1):
            if np.isclose(xs_sort[j + 1], xs_sort[j]):  # same coordinate: skip
                continue

            left_n = int(np.sum(can_fit_sort[:j]))
            right_n = int(np.sum(can_fit_sort[j + 1:]))

            if left_n < k and right_n < k:
                continue

            if (not split_one_side) and (left_n < k or right_n < k):
                continue

            # residual-range excludes the cut point j
            if j == 0:
                left_range = 0.0
            else:
                left_range = float(left_max[j - 1] - left_min[j - 1])
            right_range = float(right_max[j + 1] - right_min[j + 1])

            impurity = max(left_range, right_range)
            balance = abs(left_n - right_n)

            better = False
            if impurity < best_impurity:
                better = True
            elif np.isclose(impurity, best_impurity):
                if split_one_side:
                    if balance < best_balance and (left_n >= k or right_n >= k):
                        better = True
                else:
                    if balance < best_balance:
                        better = True

            if better:
                temp_split += 1
                # change the split value to the left sample value
                split_value = float(xs_sort[j])
                # split_value = float(0.5 * (xs_sort[j] + xs_sort[j + 1]))
                left_idx = order[:j + 1]
                right_idx = order[j + 1:]
                cut_idx = int(par_idx[order[j]])
                best_impurity = impurity
                best_balance = balance
                best = (
                    i,
                    split_value,
                    par_idx[left_idx].copy(),
                    par_idx[right_idx].copy(),
                    cut_idx,
                )

    DATA.split_num += temp_split

    t_e = time.perf_counter()
    DATA.split_time += t_e - t_s
    return best



def adaSubDiv(X, y, bounds, eps_star, N_PAC):
    n, dim = X.shape
    pieces = []

    # the queue for the box
    S = deque()
    # the whole domain
    entire_idx = np.arange(n)
    S.append({
        'bounds': bounds,
        'idx': entire_idx,
        'fit_idx': entire_idx,
        'par_w': None,
        'par_eps': None,
    })

    while S:
        node = S.popleft()
        idx = node['idx']
        fit_idx = node.get('fit_idx', idx)
        X_sub = X[idx]
        y_sub = y[idx]

        # cut points are kept in idx / pieces but excluded from fitting
        w_sub, eps_sub = fit_linear_pac(X[fit_idx], y[fit_idx])

        if eps_sub <= eps_star:
            pieces.append({
                'bounds': node['bounds'],
                'w': w_sub,
                'idx': idx,
                'epsilon': eps_sub
            })
        else:
            # try to do a subdivision
            errors = compute_errors(X_sub, y_sub, w_sub)
            split = best_split(X_sub, y_sub, idx, N_PAC, errors, fit_idx=fit_idx)
            if split is not None:
                split_d, split_value, left_idx, right_idx, cut_idx = split
                bounds_l = node['bounds'].copy()
                bounds_r = node['bounds'].copy()
                low, high = node['bounds'][split_d]

                # update the boundary
                bounds_l[split_d] = (low, split_value)
                bounds_r[split_d] = (split_value, high)

                parent_fit = np.asarray(fit_idx)
                left_fit = left_idx[(left_idx != cut_idx) & np.isin(left_idx, parent_fit)]
                right_fit = right_idx[np.isin(right_idx, parent_fit)]

                node_left = {
                    'bounds': bounds_l,
                    'idx': left_idx,
                    'fit_idx': left_fit,
                    'par_w': w_sub,
                    'par_eps': eps_sub
                }
                node_right = {
                    'bounds': bounds_r,
                    'idx': right_idx,
                    'fit_idx': right_fit,
                    'par_w': w_sub,
                    'par_eps': eps_sub
                }
                # PAC sample check uses points that will enter fit_linear_pac
                if node_left['fit_idx'].size < N_PAC and node_left['par_w'] is not None:
                    pieces.append({
                        'bounds': node_left['bounds'],
                        'w': node_left['par_w'],
                        'idx': left_idx,
                        'epsilon': node_left['par_eps']
                    })
                    S.append(node_right)
                    # print(f"right part can be further divided")
                elif node_right['fit_idx'].size < N_PAC and node_right['par_w'] is not None:
                    pieces.append({
                        'bounds': node_right['bounds'],
                        'w': node_right['par_w'],
                        'idx': right_idx,
                        'epsilon': node_right['par_eps']
                    })
                    # print(f"left part can be further divided")
                    S.append(node_left)
                else:
                    # print(f"both parts can be further divided")
                    S.append(node_left)
                    S.append(node_right)
            else:
                # there's no valid subdivision
                pieces.append({
                    'bounds': node['bounds'],
                    'w': w_sub,
                    'idx': idx,
                    'epsilon': eps_sub
                })
    return pieces


np.random.seed(0)

# alpha = 0.05, 0.01
# beta = 10^-8
# eps_stars = [15, 20, 25, 30, 35]
# 30 20 15
# 35 25 20
# 40 30 25

# eps_stars = [40]
# eps_stars = [30]
eps_stars = [25]

# param = [(0.05, 1e-8, 80000), (0.1, 1e-8, 80000)]
# param = [(0.05, 1e-8, 40000), (0.1, 1e-8, 40000)]
param = [(0.05, 1e-8, 20000), (0.1, 1e-8, 20000)]
# param = [(0.01, 1e-8, 80000)]
GPU_FEATURE_TEMP = [
    "freq",
    "memory",
    "gflops_batch",
    "gflops_per_image",
    "latency_avg_ms",
    # "latency_p95_ms"
]


def build_feature_column(n_gpus):
    x_cols = []
    for i in range(1, n_gpus + 1):
        for feat in GPU_FEATURE_TEMP:
            x_cols.append(f"gpu{i}_{feat}")
    x_cols.append("cpu_freq")
    return x_cols


N_GPUs = 1
# N_GPUs = 2
# N_GPUs = 4

for par in param:
    alpha, beta, k = par
    DATA.__init__()
    for eps in eps_stars:
        eps_star = eps
        df = pd.read_csv(f"FNN_{N_GPUs}_{k}_new.csv")
        # x_cols = [
        #     "gpu_freq",
        #     "gpu_memory",
        #     "gpu_gflops_batch",
        #     "gpu_gflops_per_image",
        #     "latency_avg_ms",
        #     "latency_p95_ms",
        #     "cpu_freq"
        # ]
        y_col = "total_power"
        x_cols = build_feature_column(N_GPUs)

        df = df.dropna(subset=x_cols + [y_col])
        # df = df.drop_duplicates(subset=x_cols + [y_col]).reset_index(drop=True)

        X = df[x_cols].to_numpy()
        y = df[y_col].to_numpy()

        # gpu_freq = df["gpu_freq"].to_numpy().reshape(-1, 1)
        # cpu_freq = df["cpu_freq"].to_numpy().reshape(-1, 1)

        # cpu_freq_sq = cpu_freq ** 2
        # gpu_freq_sq = gpu_freq ** 2
        # cpu_gpu_interaction = cpu_freq * gpu_freq

        print(f"data shape: \n X:{X.shape}, y:{y.shape}")
        sample_size = X.shape[0]
        print(f"sample size = {sample_size}")

        features = X.shape[1] + 2  # w + 1
        N_PAC = int(2 / alpha * (np.log(1 / beta) + features))  # N_PAC
        print(f"N_PAC = {N_PAC}")

        init_bounds = [tuple(map(float, pair)) for pair in np.column_stack(
            (X.min(axis=0), X.max(axis=0))
        )]

        DATA.start_timing()

        pieces = adaSubDiv(X, y, init_bounds, eps_star, N_PAC)
        DATA.end_timing()

        DATA.n_pieces = len(pieces)
        DATA.n_splits = max(0, len(pieces) - 1)

        print(f"==== PAC Build Data ===\n"
              f"Total time (s):     {DATA.t_total:.4f}\n"
              f"LP solved:          {DATA.lp_count}\n"
              f"Avg LP solve time (s)   {DATA.lp_time_avg():.4f}\n"
              f"Split times:        {DATA.split_num}\n"
              f"Split time (s):     {DATA.split_time}\n"
              f"Total Pieces:       {DATA.n_pieces}\n"
              f"Total Splits:       {DATA.n_splits}")

        with open(f"PAC_PERF/pac_build_gpu_{N_GPUs}_a{alpha}_b{beta}_k{sample_size}_e{eps_star}_NPAC{N_PAC}.json",
                  "w", ) as f_datas:
            json.dump(DATA.to_dict(), f_datas, ensure_ascii=False, indent=2)

        print(f"there's {len(pieces)} pieces")
        for i, p in enumerate(pieces):
            print(f"piece {i}:"
                  f"bounds = {p['bounds']}, n_sample = {p['idx'].size}, epsilon = {p['epsilon']:.4f}\n"
                  f"model = {p['w']}")

        to_save = [{
            "bounds": [[float(a), float(b)] for (a, b) in p["bounds"]],
            "w": [float(x) for x in np.asarray(p["w"]).ravel()],
            "epsilon": float(p["epsilon"]),
            "n_sample": int(np.asarray(p["idx"]).size),
            "idx": [int(i) for i in np.asarray(p["idx"]).ravel().tolist()],
        } for p in pieces]
        print(
            f" \n======now dump pieces_NGPU{N_GPUs}_a{alpha}_b{beta}_k{sample_size}_e{eps_star}_NPAC{N_PAC}.json\n")
        with open(f"PAC_PIECE/pieces_NGPU{N_GPUs}_a{alpha}_b{beta}_k{sample_size}_e{eps_star}_NPAC{N_PAC}.json", "w",
                  encoding="utf-8") as f:
            json.dump(to_save, f, ensure_ascii=False, indent=2)
