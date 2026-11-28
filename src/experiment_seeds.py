import csv
import os
import time

import numpy as np

from data_gen import make_data
from scalers import ALL_SCALERS
from gd import Problem, gradient_descent, solve_optimum


SEEDS = list(range(10))
TASKS = ["linear", "logistic"]

N = 2000
D = 10
LAM = 0.01
MAX_ITER = 20000

E1_SCALE_RATIOS = [1, 10, 100, 1000, 10000]
E1_RHO = 0.5
E2_RHOS = [0, 0.5, 0.9, 0.99]
E2_SCALE_RATIO = 100


def run_setting(task, scale_ratio, rho, seed):
    X, y, _ = make_data(
        n=N, d=D, scale_ratio=scale_ratio, rho=rho,
        task=task, offset=5.0, seed=seed,
    )
    rows = []
    for S in ALL_SCALERS:
        sc = S()
        Xs = sc.fit_transform(X)
        prob = Problem(Xs, y, task, lam=LAM, P=sc.penalty_matrix())
        w_star, f_star = solve_optimum(prob)
        cond = np.linalg.cond(prob.hessian(w_star))
        res = gradient_descent(prob, f_star=f_star, max_iter=MAX_ITER)
        rows.append({
            "task": task,
            "scale_ratio": scale_ratio,
            "rho": rho,
            "seed": seed,
            "scaler": sc.name,
            "condition_number": cond,
            "iterations": res["n_iter"],
            "converged": res["converged"],
        })
    return rows


def run_all():
    settings = []
    for task in TASKS:
        for ratio in E1_SCALE_RATIOS:
            settings.append(("E1", task, ratio, E1_RHO))
        for rho in E2_RHOS:
            settings.append(("E2", task, E2_SCALE_RATIO, rho))

    results = []
    start = time.perf_counter()
    for i, (exp, task, ratio, rho) in enumerate(settings, 1):
        for seed in SEEDS:
            for row in run_setting(task, ratio, rho, seed):
                row["experiment"] = exp
                results.append(row)
        elapsed = time.perf_counter() - start
        print(f"[{i}/{len(settings)}] {exp} {task} scale_ratio={ratio} rho={rho} "
              f"done ({elapsed:.0f}s)")

    os.makedirs("results", exist_ok=True)
    output_file = "results/e1_e2_seeds.csv"
    with open(output_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    run_all()
