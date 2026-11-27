import numpy as np
from data_gen import make_data
from scalers import MinMax
from gd import Problem, solve_optimum

X, y, _ = make_data(n=2000, d=10, scale_ratio=100, rho=0.5, task="linear", offset=5.0, seed=0)
Xs = MinMax().fit_transform(X)
print(Xs.mean(axis=0).round(3))
print(Xs.std(axis=0).round(3))
for name, Z in [("min-max", Xs), ("min-max centered", Xs - Xs.mean(axis=0))]:
    prob = Problem(Z, y, "linear", lam=0.01)
    w, _ = solve_optimum(prob)
    print(name, round(np.linalg.cond(prob.hessian(w)), 1))