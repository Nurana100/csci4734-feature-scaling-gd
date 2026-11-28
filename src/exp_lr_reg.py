import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from data_gen import make_data
from scalers import ALL_SCALERS
from gd import Problem, gradient_descent, solve_optimum

MAX_ITER = 5000
task = "linear"
X, y, _ = make_data(n=2000, d=10, scale_ratio=100, rho=0.5,
                    task=task, offset=5.0, seed=0)
rows = []
for S in ALL_SCALERS:
    sc = S(); Xs = sc.fit_transform(X)
    prob = Problem(Xs, y, task, lam=0.01, P=sc.penalty_matrix())
    _, f_star = solve_optimum(prob)
    L = prob.smoothness()
    for lr in np.logspace(-6, 1, 29):
        r = gradient_descent(prob, lr=lr, f_star=f_star, max_iter=MAX_ITER)
        rows.append(dict(scaler=sc.name, lr=lr, lr_times_L=lr * L,
                         iters=r["n_iter"], converged=r["converged"]))
    print(sc.name, "L =", L, "-> stable lr must be < 2/L =", 2 / L)

df = pd.DataFrame(rows)
df.to_csv("../results/e3.csv", index=False)

plt.figure(figsize=(6, 4))
for name, g in df.groupby("scaler"):
    ok = g[g.converged]
    plt.plot(ok.lr, ok.iters, marker="o", label=name)
plt.xscale("log"); plt.yscale("log")
plt.xlabel("learning rate"); plt.ylabel(f"iterations to converge (cap {MAX_ITER})")
plt.title("Only converged runs shown"); plt.legend()
plt.tight_layout(); plt.savefig("../results/e3.png", dpi=150)
