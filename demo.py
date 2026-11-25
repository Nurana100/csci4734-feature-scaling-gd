import time
import numpy as np
from data_gen import make_data
from scalers import ALL_SCALERS
from gd import Problem, gradient_descent, solve_optimum


def run(task, scale_ratio, rho, lam=0.01, offset=5.0):
    X, y, _ = make_data(n=2000, d=10, scale_ratio=scale_ratio, rho=rho,
                        task=task, offset=offset, seed=0)
    print(f"\n{task}: scale_ratio={scale_ratio}, rho={rho}, lam={lam}")
    print(f"{'scaler':12s} {'scale time(s)':>13s} {'cond(H)':>10s} {'iters':>8s} {'f*':>10s}")
    for S in ALL_SCALERS:
        t0 = time.perf_counter()
        sc = S()
        Xs = sc.fit_transform(X)
        t_scale = time.perf_counter() - t0
        prob = Problem(Xs, y, task, lam=lam, P=sc.penalty_matrix())
        w_star, f_star = solve_optimum(prob)
        cond = np.linalg.cond(prob.hessian(w_star))
        res = gradient_descent(prob, f_star=f_star, max_iter=20000)
        it = res["n_iter"] if res["converged"] else f">{res['n_iter']}"
        print(f"{sc.name:12s} {t_scale:13.5f} {cond:10.1f} {str(it):>8s} {f_star:10.6f}")


if __name__ == "__main__":
    run("linear", scale_ratio=100, rho=0.5)
    run("logistic", scale_ratio=100, rho=0.5)
