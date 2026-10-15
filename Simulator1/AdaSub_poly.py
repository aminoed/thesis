import json
import math
from collections import deque
from itertools import combinations_with_replacement
import numpy as np
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
        dic["lp_time_avg"] = self.lp_time_avg()
        return dic


DATA = BuildStates()


def n_monomials(d, m):
    return math.comb(d + m, m)


def n_pac_for_degree(alpha, beta, d, m):
    return int(math.ceil(
        (2.0 / alpha) * (math.log(1.0 / beta) + n_monomials(d, m) + 1)
    ))


def monomial_exponents(d, degree):
    exps = []
    # combinations_with_replacement gives nondecreasing multi-indices of length t
    for total in range(degree + 1):
        if total == 0:
            exps.append(tuple([0] * d))
            continue
        for combo in combinations_with_replacement(range(d), total):
            q = [0] * d
            for var in combo:
                q[var] += 1
            exps.append(tuple(q))
    return exps


def build_monomial_matrix(X, degree):
    n, d = X.shape
    exps = monomial_exponents(d, degree)
    Phi = np.ones((n, len(exps)), dtype=float)

    for j, q in enumerate(exps):
        for dim, power in enumerate(q):
            if power > 0:
                Phi[:, j] *= X[:, dim] ** power

    return Phi, exps


def predict_polynomial(X, w, degree):
    Phi, _ = build_monomial_matrix(X, degree)
    return Phi @ w


def compute_errors(X, y, w, degree):
    if w is None:
        print("not solvable, w is none")
        return None
    y_pred = predict_polynomial(X, w, degree)
    return np.abs(y_pred - y)


def fit_polynomial_pac(X_sub, y_sub, degree):
    n_sub, d = X_sub.shape
    if n_sub == 0:
        return None, None

    Phi, _ = build_monomial_matrix(X_sub, degree)
    M = Phi.shape[1]          # number of coefficients

    A = np.zeros((2 * n_sub, M + 1))
    b = np.zeros(2 * n_sub)

    for i in range(n_sub):
        phi_i = Phi[i]
        yi = y_sub[i]

        A[2 * i, :-1] = phi_i
        A[2 * i, -1] = -1.0
        b[2 * i] = yi

        A[2 * i + 1, :-1] = -phi_i
        A[2 * i + 1, -1] = -1.0
        b[2 * i + 1] = -yi

    c = np.zeros(M + 1)
    c[-1] = 1.0

    bounds = [(None, None)] * M + [(1e-4, None)]

    t_s = time.perf_counter()
    res = optimize.linprog(c, A_ub=A, b_ub=b, bounds=bounds, method="highs")
    t_e = time.perf_counter()
    DATA.lp_count += 1
    DATA.lp_time += t_e - t_s

    if res.success:
        w = res.x[:-1]
        eps = float(res.x[-1])
        return w, eps

    print("linprog failed:", res.message)
    w = np.linalg.lstsq(Phi, y_sub, rcond=None)[0]
    eps = float(np.max(np.abs(Phi @ w - y_sub)))
    return w, eps


def fit_adaptive_degree(X_sub, y_sub, eps_min, alpha, beta, m_max):
    n_sub, d = X_sub.shape
    if n_sub == 0:
        return None

    previous_model = None  

    for m in range(1, m_max + 1):
        n_required = n_pac_for_degree(alpha, beta, d, m)
        if n_sub < n_required:
            if m == 1:
                return None
            return previous_model

        w, eps = fit_polynomial_pac(X_sub, y_sub, m)
        if w is None:
            if m == 1:
                return None
            return previous_model

        previous_model = (w, float(eps), m)

        if eps <= eps_min:
            return previous_model

    return previous_model


def best_split(X_sub, y_sub, par_idx, k, error, fit_idx=None):
    n_sub, d = X_sub.shape
    if error is None or n_sub < k:
        return None

    best = None
    best_impurity = np.inf
    best_balance = np.inf

    t_s = time.perf_counter()
    temp_split = 0
    split_one_side = (n_sub < 2 * k)

    if fit_idx is None:
        can_fit = np.ones(n_sub, dtype=bool)
    else:
        fit_set = set(np.asarray(fit_idx).tolist())
        can_fit = np.array(
            [int(par_idx[t]) in fit_set for t in range(n_sub)], dtype=bool
        )

    for dim in range(d):
        order = np.argsort(X_sub[:, dim])
        xs_sort = X_sub[order, dim]
        errors_sort = error[order]
        can_fit_sort = can_fit[order]

        left_max = np.maximum.accumulate(errors_sort)
        left_min = np.minimum.accumulate(errors_sort)
        right_max = np.maximum.accumulate(errors_sort[::-1])[::-1]
        right_min = np.minimum.accumulate(errors_sort[::-1])[::-1]

        for j in range(len(xs_sort) - 1):
            # same coordinate on this axis: cannot separate left / right
            if np.isclose(xs_sort[j + 1], xs_sort[j]):
                continue

            # counts and impurity exclude the cut point j
            left_n = int(np.sum(can_fit_sort[:j]))
            right_n = int(np.sum(can_fit_sort[j + 1:]))

            if left_n < k and right_n < k:
                continue
            if (not split_one_side) and (left_n < k or right_n < k):
                continue

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
                split_value = float(xs_sort[j])
                left_local = order[: j + 1]     # cut point stays on the left
                right_local = order[j + 1:]
                cut_idx = int(par_idx[order[j]])
                best_impurity = impurity
                best_balance = balance
                best = (
                    dim,
                    split_value,
                    par_idx[left_local].copy(),
                    par_idx[right_local].copy(),
                    cut_idx,
                )

    DATA.split_num += temp_split
    DATA.split_time += time.perf_counter() - t_s
    return best


def adaSubDiv(X, y, bounds, eps_star, alpha, beta, m_max):
    n, dim = X.shape
    pieces = []
    N_PAC = n_pac_for_degree(alpha, beta, dim, 1)

    S = deque()
    entire_idx = np.arange(n)
    S.append({
        "bounds": bounds,
        "idx": entire_idx,
        "fit_idx": entire_idx,
        "par_w": None,
        "par_eps": None,
        "par_degree": None,
    })

    while S:
        node = S.popleft()
        idx = node["idx"]
        fit_idx = node.get("fit_idx", idx)
        X_sub = X[idx]
        y_sub = y[idx]

        fitted = fit_adaptive_degree(
            X[fit_idx], y[fit_idx], eps_star, alpha, beta, m_max
        )

        if fitted is None:
            if node["par_w"] is not None:
                pieces.append({
                    "bounds": node["bounds"],
                    "w": node["par_w"],
                    "degree": node["par_degree"],
                    "idx": idx,
                    "epsilon": node["par_eps"],
                })
            continue

        w_sub, eps_sub, degree_sub = fitted

        if eps_sub <= eps_star:
            pieces.append({
                "bounds": node["bounds"],
                "w": w_sub,
                "degree": degree_sub,
                "idx": idx,
                "epsilon": eps_sub,
            })
            continue

        errors = compute_errors(X_sub, y_sub, w_sub, degree_sub)
        split = best_split(X_sub, y_sub, idx, N_PAC, errors, fit_idx=fit_idx)

        if split is None:
            pieces.append({
                "bounds": node["bounds"],
                "w": w_sub,
                "degree": degree_sub,
                "idx": idx,
                "epsilon": eps_sub,
            })
            continue

        split_d, split_value, left_idx, right_idx, cut_idx = split
        bounds_l = node["bounds"].copy()
        bounds_r = node["bounds"].copy()
        low, high = node["bounds"][split_d]
        bounds_l[split_d] = (low, split_value)
        bounds_r[split_d] = (split_value, high)

        parent_fit = np.asarray(fit_idx)

        left_fit = left_idx[(left_idx != cut_idx) & np.isin(left_idx, parent_fit)]
        right_fit = right_idx[np.isin(right_idx, parent_fit)]

        node_left = {
            "bounds": bounds_l,
            "idx": left_idx,
            "fit_idx": left_fit,
            "par_w": w_sub,
            "par_eps": eps_sub,
            "par_degree": degree_sub,
        }
        node_right = {
            "bounds": bounds_r,
            "idx": right_idx,
            "fit_idx": right_fit,
            "par_w": w_sub,
            "par_eps": eps_sub,
            "par_degree": degree_sub,
        }

        if node_left["fit_idx"].size < N_PAC and node_left["par_w"] is not None:
            pieces.append({
                "bounds": node_left["bounds"],
                "w": node_left["par_w"],
                "degree": node_left["par_degree"],
                "idx": left_idx,
                "epsilon": node_left["par_eps"],
            })
            S.append(node_right)
        elif node_right["fit_idx"].size < N_PAC and node_right["par_w"] is not None:
            pieces.append({
                "bounds": node_right["bounds"],
                "w": node_right["par_w"],
                "degree": node_right["par_degree"],
                "idx": right_idx,
                "epsilon": node_right["par_eps"],
            })
            S.append(node_left)
        else:
            S.append(node_left)
            S.append(node_right)

    return pieces


np.random.seed(0)

eps_stars = [25]
param = [(0.05, 1e-8, 20000), (0.1, 1e-8, 20000)]

# maximum polynomial degree for Algorithm 1
M_MAX = 3

GPU_FEATURE_TEMP = [
    "freq",
    "memory",
    "gflops_batch",
    "gflops_per_image",
    "latency_avg_ms",
]


def build_feature_column(n_gpus):
    x_cols = []
    for i in range(1, n_gpus + 1):
        for feat in GPU_FEATURE_TEMP:
            x_cols.append(f"gpu{i}_{feat}")
    x_cols.append("cpu_freq")
    return x_cols


N_GPUs = 1

for par in param:
    alpha, beta, k = par
    DATA.__init__()
    for eps in eps_stars:
        eps_star = eps
        df = pd.read_csv(f"FNN_{N_GPUs}_{k}_new.csv")
        y_col = "total_power"
        x_cols = build_feature_column(N_GPUs)

        df = df.dropna(subset=x_cols + [y_col])
        X = df[x_cols].to_numpy()
        y = df[y_col].to_numpy()

        print(f"data shape:\n X:{X.shape}, y:{y.shape}")
        sample_size = X.shape[0]
        print(f"sample size = {sample_size}")

        d = X.shape[1]
        N_PAC = n_pac_for_degree(alpha, beta, d, 1)
        print(f"N_PAC (m=1) = {N_PAC}")
        print(f"N_PAC (m={M_MAX}) = {n_pac_for_degree(alpha, beta, d, M_MAX)}")
        print(f"m_max = {M_MAX}")

        init_bounds = [
            tuple(map(float, pair))
            for pair in np.column_stack((X.min(axis=0), X.max(axis=0)))
        ]

        DATA.start_timing()
        pieces = adaSubDiv(X, y, init_bounds, eps_star, alpha, beta, M_MAX)
        DATA.end_timing()

        DATA.n_pieces = len(pieces)
        DATA.n_splits = max(0, len(pieces) - 1)

        print(
            f"==== PAC Build Data ===\n"
            f"Total time (s):     {DATA.t_total:.4f}\n"
            f"LP solved:          {DATA.lp_count}\n"
            f"Avg LP solve time (s)   {DATA.lp_time_avg():.4f}\n"
            f"Split times:        {DATA.split_num}\n"
            f"Split time (s):     {DATA.split_time}\n"
            f"Total Pieces:       {DATA.n_pieces}\n"
            f"Total Splits:       {DATA.n_splits}"
        )

        with open(
            f"PAC_PERF/pac_build_poly_gpu_{N_GPUs}_a{alpha}_b{beta}_k{sample_size}"
            f"_e{eps_star}_m{M_MAX}_NPAC{N_PAC}.json",
            "w",
        ) as f_datas:
            json.dump(DATA.to_dict(), f_datas, ensure_ascii=False, indent=2)

        print(f"there's {len(pieces)} pieces")
        for i, p in enumerate(pieces):
            print(
                f"piece {i}:"
                f"bounds = {p['bounds']}, n_sample = {p['idx'].size}, "
                f"degree = {p['degree']}, epsilon = {p['epsilon']:.4f}\n"
                f"model = {p['w']}"
            )

        to_save = [{
            "bounds": [[float(a), float(b)] for (a, b) in p["bounds"]],
            "w": [float(x) for x in np.asarray(p["w"]).ravel()],
            "degree": int(p["degree"]) if p["degree"] is not None else None,
            "epsilon": float(p["epsilon"]),
            "n_sample": int(np.asarray(p["idx"]).size),
            "idx": [int(i) for i in np.asarray(p["idx"]).ravel().tolist()],
        } for p in pieces]

        out_name = (
            f"PAC_PIECE/pieces_poly_NGPU{N_GPUs}_a{alpha}_b{beta}_k{sample_size}"
            f"_e{eps_star}_m{M_MAX}_NPAC{N_PAC}.json"
        )
        print(f"\n====== dump {out_name}\n")
        with open(out_name, "w", encoding="utf-8") as f:
            json.dump(to_save, f, ensure_ascii=False, indent=2)
