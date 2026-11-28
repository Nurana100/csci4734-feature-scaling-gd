from data_gen import make_data
from scalers import ALL_SCALERS
from gd import Problem, solve_optimum

# E5: a naive L2 penalty on the scaled weights (P = None, i.e. P = I) is a
# different problem for every scaler. The equivalent penalty P = A^T A
# (sc.penalty_matrix()) gives the same objective as raw-space ridge.
lam = 0.1
X, y, _ = make_data(n=2000, d=10, scale_ratio=100, rho=0.5,
                    task="linear", offset=5.0, seed=0)

print(f"{'scaler':12s} {'f_naive':>10s} {'f_equiv':>10s}")
for S in ALL_SCALERS:
    sc = S(); Xs = sc.fit_transform(X)
    _, f_naive = solve_optimum(Problem(Xs, y, "linear", lam=lam, P=None))
    _, f_equiv = solve_optimum(Problem(Xs, y, "linear", lam=lam,
                                       P=sc.penalty_matrix()))
    print(f"{sc.name:12s} {f_naive:10.4f} {f_equiv:10.4f}")
