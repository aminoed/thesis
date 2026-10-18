import json
import math
import os
from collections import deque
from itertools import combinations_with_replacement
import numpy as np
from scipy import optimize
import pandas as pd
import time


ALPHA = 0.05
BETA = 1e-8
M_MAX = 3
SPLIT_STRATEGY = "residual_range" 
# SPLIT_STRATEGY = "maximum_residual"
EPS_STAR = 0.5


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


def fit_adaptive_degree(X_sub, y_sub, eps_star, alpha, beta, m_max):
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

        if eps <= eps_star:
            return previous_model

    return previous_model


def best_split(X_sub, y_sub, par_idx, k, error, fit_idx=None, split_strategy="residual_range"):
    n_sub, d = X_sub.shape
    if error is None or n_sub < k:
        return None

    best = None
    best_impurity = np.inf
    best_balance = np.inf
    temp_split = 0
    t_s = time.perf_counter()

    if fit_idx is None:
        can_fit = np.ones(n_sub, dtype=bool)
    else:
        can_fit = np.isin(par_idx, fit_idx)

    for dim in range(d):
        order = np.argsort(X_sub[:, dim], kind="stable")
        xs_sort = X_sub[order, dim]
        errors_sort = error[order]
        can_fit_sort = can_fit[order]

        left_max = np.maximum.accumulate(errors_sort)
        left_min = np.minimum.accumulate(errors_sort)
        right_max = np.maximum.accumulate(errors_sort[::-1])[::-1]
        right_min = np.minimum.accumulate(errors_sort[::-1])[::-1]

        for j in range(n_sub - 1):
            if np.isclose(xs_sort[j + 1], xs_sort[j]):
                continue

            left_n = int(np.sum(can_fit_sort[:j]))
            right_n = int(np.sum(can_fit_sort[j + 1:]))
            remaining_fit = left_n + right_n

            if left_n < k and right_n < k:
                continue

            split_one_side = remaining_fit < 2 * k
            if not split_one_side:
                if left_n < k or right_n < k:
                    continue

            if split_strategy == "residual_range":
                if j == 0:
                    left_score = 0.0
                else:
                    left_score = float(left_max[j - 1] - left_min[j - 1])

                right_score = float(right_max[j + 1] - right_min[j + 1])

            else:
                if j == 0:
                    left_score = 0.0
                else:
                    left_score = float(left_max[j - 1])

                right_score = float(right_max[j + 1])

            impurity = max(left_score, right_score)
            balance = abs(left_n - right_n)
            temp_split += 1

            better = False
            if impurity < best_impurity:
                better = True
            elif np.isclose(impurity, best_impurity):
                if balance < best_balance:
                    better = True

            if better:
                split_value = float(xs_sort[j])
                left_local = order[:j + 1]
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


def adaSubDiv(X, y, bounds, eps_star, alpha, beta, m_max, split_strategy="residual_range"):

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
        "par_fit_idx": None,
    })

    while S:
        node = S.popleft()
        idx = node["idx"]
        fit_idx = node["fit_idx"]
        X_sub = X[idx]
        y_sub = y[idx]

        fitted = fit_adaptive_degree(
            X[fit_idx], y[fit_idx], eps_star, alpha, beta, m_max
        )

        if fitted is None:
            if node["par_w"] is None:
                raise RuntimeError("Could not fit the root region")

            pieces.append({
                "bounds": node["bounds"],
                "w": node["par_w"],
                "degree": node["par_degree"],
                "epsilon": node["par_eps"],
                "idx": idx,
                "fit_idx": fit_idx,
                "model_fit_idx": node["par_fit_idx"],
                "inherited": True,
                "stop_reason": "insufficient_fitting_samples",
            })
            continue

        w_sub, eps_sub, degree_sub = fitted

        if eps_sub <= eps_star:
            pieces.append({
                "bounds": node["bounds"],
                "w": w_sub,
                "degree": degree_sub,
                "epsilon": eps_sub,
                "idx": idx,
                "fit_idx": fit_idx,
                "model_fit_idx": fit_idx,
                "inherited": False,
                "stop_reason": "threshold_met",
            })
            continue

        errors = compute_errors(X_sub, y_sub, w_sub, degree_sub)
        split = best_split(X_sub, y_sub, idx, N_PAC, errors, fit_idx=fit_idx, split_strategy=split_strategy)

        if split is None:
            pieces.append({
                "bounds": node["bounds"],
                "w": w_sub,
                "degree": degree_sub,
                "epsilon": eps_sub,
                "idx": idx,
                "fit_idx": fit_idx,
                "model_fit_idx": fit_idx,
                "inherited": False,
                "stop_reason": "no_valid_split",
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
            "par_fit_idx": fit_idx,
        }
        node_right = {
            "bounds": bounds_r,
            "idx": right_idx,
            "fit_idx": right_fit,
            "par_w": w_sub,
            "par_eps": eps_sub,
            "par_degree": degree_sub,
            "par_fit_idx": fit_idx,
        }
        DATA.n_splits += 1

        if left_fit.size < N_PAC:
            pieces.append({
                "bounds": node_left["bounds"],
                "w": w_sub,
                "degree": degree_sub,
                "epsilon": eps_sub,
                "idx": left_idx,
                "fit_idx": left_fit,
                "model_fit_idx": fit_idx,
                "inherited": True,
                "stop_reason": "insufficient_fitting_samples",
            })
            S.append(node_right)

        elif right_fit.size < N_PAC:
            pieces.append({
                "bounds": node_right["bounds"],
                "w": w_sub,
                "degree": degree_sub,
                "epsilon": eps_sub,
                "idx": right_idx,
                "fit_idx": right_fit,
                "model_fit_idx": fit_idx,
                "inherited": True,
                "stop_reason": "insufficient_fitting_samples",
            })
            S.append(node_left)

        else:
            S.append(node_left)
            S.append(node_right)

    DATA.n_pieces = len(pieces)
    return pieces


def serialize_piece(piece, X, y):
    degree = int(piece["degree"])
    idx = piece["idx"]
    fit_idx = piece["fit_idx"]
    model_fit_idx = piece["model_fit_idx"]

    saved_bounds = []
    for low, high in piece["bounds"]:
        saved_bounds.append([float(low), float(high)])

    saved_exponents = []
    for exponent in monomial_exponents(X.shape[1], degree):
        saved_exponents.append(list(exponent))

    predictions = predict_polynomial(X[idx], piece["w"], degree)
    errors = np.abs(predictions - y[idx])

    saved_piece = {
        "bounds": saved_bounds,
        "w": np.asarray(piece["w"]).ravel().tolist(),
        "degree": degree,
        "exponents": saved_exponents,
        "epsilon": float(piece["epsilon"]),
        "n_sample": int(idx.size),
        "idx": idx.tolist(),
        "n_region_fit_samples": int(fit_idx.size),
        "fit_idx": fit_idx.tolist(),
        "n_fit_samples": int(model_fit_idx.size),
        "model_fit_idx": model_fit_idx.tolist(),
        "inherited": bool(piece["inherited"]),
        "stop_reason": piece["stop_reason"],
        "max_error": float(np.max(errors)),
    }
    return saved_piece


# data_file = f"bounce_ball_X.csv" 
data_file = f"bounce_ball_Y.csv"
X = pd.read_csv(data_file)[["x0", "x1"]].to_numpy(dtype=float)
# y = pd.read_csv(data_file)["y0"].to_numpy(dtype=float)    
y = pd.read_csv(data_file)["y1"].to_numpy(dtype=float)
sample_size, d = X.shape
N_PAC = n_pac_for_degree(ALPHA, BETA, d, 1)
init_bounds = [(X[:, 0].min(), X[:, 0].max()), (X[:, 1].min(), X[:, 1].max())]

DATA.__init__()
DATA.start_timing()
pieces = adaSubDiv(X, y, init_bounds, EPS_STAR, ALPHA, BETA, M_MAX, split_strategy=SPLIT_STRATEGY)
DATA.end_timing()

saved_pieces = []
for piece in pieces:
    saved_piece = serialize_piece(piece, X, y)
    saved_pieces.append(saved_piece)

performance = DATA.to_dict()
performance["alpha"] = ALPHA
performance["beta"] = BETA
performance["m_max"] = M_MAX
performance["split_strategy"] = SPLIT_STRATEGY
performance["epsilon_threshold"] = EPS_STAR
performance["sample_size"] = sample_size
performance["n_pac_linear"] = N_PAC
performance["epsilon_low"] = min(p["epsilon"] for p in pieces)
performance["epsilon_high"] = max(p["epsilon"] for p in pieces)

for degree in (1, 2, 3):
    count = 0
    for piece in pieces:
        if piece["degree"] == degree:
            count += 1
    performance[f"degree_{degree}_pieces"] = count

tag = (
    f"a{ALPHA:g}_b{BETA:g}_k{sample_size}"
    f"_e{EPS_STAR}_m{M_MAX}_s{SPLIT_STRATEGY}_NPAC{N_PAC}_Y"   
)

os.makedirs("PAC_PIECE", exist_ok=True)
os.makedirs("PAC_PERF", exist_ok=True)

model_file = f"PAC_PIECE/pieces_{tag}.json"
performance_file = f"PAC_PERF/pac_build_{tag}.json"

with open(model_file, "w", encoding="utf-8") as model_output:
    json.dump(saved_pieces, model_output, ensure_ascii=False, indent=2)
    model_output.write("\n")

with open(performance_file, "w", encoding="utf-8") as performance_output:
    json.dump(performance, performance_output, ensure_ascii=False, indent=2)
    performance_output.write("\n")

print(f"alpha={ALPHA}, m_max={M_MAX}, split={SPLIT_STRATEGY}")
print(f"Total pieces: {DATA.n_pieces}")
print(f"Total splits: {DATA.n_splits}")
print(f"LP solved: {DATA.lp_count}")
print(f"Build time: {DATA.t_total:.4f} seconds")
print(f"Saved model: {model_file}")
print(f"Saved performance: {performance_file}")


