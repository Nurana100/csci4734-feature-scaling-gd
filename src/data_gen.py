import numpy as np


def corr_matrix(d, rho, kind="equi"):
    if kind == "equi":
        C = np.full((d, d), rho)
        np.fill_diagonal(C, 1.0)
    else:
        idx = np.arange(d)
        C = rho ** np.abs(idx[:, None] - idx[None, :])
    return C


def make_data(n=2000, d=10, scale_ratio=1.0, rho=0.0, corr_kind="equi",
              task="linear", noise=0.5, offset=0.0, seed=0):
    rng = np.random.default_rng(seed)
    L = np.linalg.cholesky(corr_matrix(d, rho, corr_kind))
    Z = rng.standard_normal((n, d)) @ L.T
    scales = np.logspace(0, np.log10(scale_ratio), d)
    X = Z * scales + offset * rng.uniform(0.5, 1.5, size=d)
    w = rng.standard_normal(d) / scales
    s = Z @ (w * scales)
    if task == "linear":
        y = s + noise * rng.standard_normal(n)
    else:
        p = 1 / (1 + np.exp(-s))
        y = (rng.uniform(size=n) < p).astype(float)
    return X, y, w


def train_test_split(X, y, test_frac=0.25, seed=0):
    rng = np.random.default_rng(seed + 12345)
    idx = rng.permutation(len(y))
    k = int(len(y) * (1 - test_frac))
    tr, te = idx[:k], idx[k:]
    return X[tr], y[tr], X[te], y[te]
