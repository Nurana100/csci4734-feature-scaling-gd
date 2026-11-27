import time
import csv
import os

import numpy as np

from data_gen import make_data
from scalers import ALL_SCALERS
from gd import Problem, gradient_descent, solve_optimum


# E2: Feature correlation vs gradient descent convergence
RHOS = [0, 0.5, 0.9, 0.99]
TASKS = ["linear", "logistic"]

N = 2000
D = 10
SCALE_RATIO = 100
LAM = 0.01
MAX_ITER = 20000


def run_e2():
    results = []

    for task in TASKS:
        for rho in RHOS:

            X, y, _ = make_data(
                n=N,
                d=D,
                scale_ratio=SCALE_RATIO,
                rho=rho,
                task=task,
                offset=5.0,
                seed=0,
            )

            print(f"\n{task}: scale_ratio={SCALE_RATIO}, rho={rho}")
            print(
                f"{'scaler':12s} {'scale time':>12s} "
                f"{'cond(H)':>12s} {'iters':>10s} {'f*':>12s}"
            )

            for S in ALL_SCALERS:
                t0 = time.perf_counter()
                sc = S()
                Xs = sc.fit_transform(X)
                t_scale = time.perf_counter() - t0

                prob = Problem(
                    Xs, y, task, lam=LAM,
                    P=sc.penalty_matrix()
                )

                w_star, f_star = solve_optimum(prob)
                cond = np.linalg.cond(prob.hessian(w_star))

                res = gradient_descent(
                    prob,
                    f_star=f_star,
                    max_iter=MAX_ITER
                )

                iterations = res["n_iter"]
                converged = res["converged"]

                iter_display = (
                    str(iterations) if converged
                    else f">{iterations}"
                )

                print(
                    f"{sc.name:12s} {t_scale:12.5f} "
                    f"{cond:12.1f} {iter_display:>10s} "
                    f"{f_star:12.6f}"
                )

                results.append({
                    "task": task,
                    "scale_ratio": SCALE_RATIO,
                    "rho": rho,
                    "scaler": sc.name,
                    "scale_time": t_scale,
                    "condition_number": cond,
                    "iterations": iterations,
                    "converged": converged,
                    "f_star": f_star,
                })

    os.makedirs("results", exist_ok=True)

    output_file = "results/e2_correlation_iterations.csv"

    with open(output_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    run_e2()
