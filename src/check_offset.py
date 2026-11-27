import numpy as np
from data_gen import make_data
from scalers import Identity
from gd import Problem, gradient_descent, solve_optimum

for offset in (5.0, 0.0):
    for ratio in (1, 100):
        X, y, _ = make_data(n=2000, d=10, scale_ratio=ratio, rho=0.5, task="linear", offset=offset, seed=0)
        sc = Identity()
        Xs = sc.fit_transform(X)
        prob = Problem(Xs, y, "linear", lam=0.01, P=sc.penalty_matrix())
        w, f = solve_optimum(prob)
        r = gradient_descent(prob, f_star=f, max_iter=20000)
        print("offset", offset, "ratio", ratio, "cond", round(np.linalg.cond(prob.hessian(w)), 1), "iters", r["n_iter"], r["converged"])