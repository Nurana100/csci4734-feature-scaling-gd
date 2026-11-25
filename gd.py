import numpy as np


def _sigmoid(z):
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1 / (1 + np.exp(-z[pos]))
    e = np.exp(z[~pos])
    out[~pos] = e / (1 + e)
    return out


class Problem:
    def __init__(self, X, y, task="linear", lam=0.0, P=None):
        n, d = X.shape
        self.Xb = np.hstack([X, np.ones((n, 1))])
        self.y, self.task, self.lam, self.n, self.d = y, task, lam, n, d
        self.Pb = np.zeros((d + 1, d + 1))
        self.Pb[:d, :d] = np.eye(d) if P is None else P

    def loss(self, w):
        z = self.Xb @ w
        if self.task == "linear":
            base = 0.5 * np.mean((z - self.y) ** 2)
        else:
            base = np.mean(np.logaddexp(0, z) - self.y * z)
        return base + 0.5 * self.lam * w @ self.Pb @ w

    def grad(self, w):
        z = self.Xb @ w
        r = (z - self.y) if self.task == "linear" else (_sigmoid(z) - self.y)
        return self.Xb.T @ r / self.n + self.lam * self.Pb @ w

    def hessian(self, w):
        if self.task == "linear":
            s = np.ones(self.n)
        else:
            p = _sigmoid(self.Xb @ w)
            s = p * (1 - p)
        return self.Xb.T @ (self.Xb * s[:, None]) / self.n + self.lam * self.Pb

    def smoothness(self):
        c = 1.0 if self.task == "linear" else 0.25
        H = c * self.Xb.T @ self.Xb / self.n + self.lam * self.Pb
        return np.linalg.eigvalsh(H).max()

    def cost_per_iter(self):
        return 4 * self.n * (self.d + 1)


def solve_optimum(prob, iters=100):
    w = np.zeros(prob.d + 1)
    for _ in range(iters):
        g = prob.grad(w)
        if np.linalg.norm(g) < 1e-12:
            break
        step = np.linalg.lstsq(prob.hessian(w), g, rcond=None)[0]
        t, f0 = 1.0, prob.loss(w)
        while prob.loss(w - t * step) > f0 and t > 1e-10:
            t /= 2
        w = w - t * step
    return w, prob.loss(w)


def gradient_descent(prob, lr=None, f_star=None, tol=1e-8, max_iter=100000):
    if f_star is None:
        _, f_star = solve_optimum(prob)
    if lr is None:
        lr = 1.0 / prob.smoothness()
    w = np.zeros(prob.d + 1)
    gaps = []
    for k in range(max_iter):
        gap = prob.loss(w) - f_star
        gaps.append(gap)
        if not np.isfinite(gap) or gap > 1e12:
            return dict(w=w, n_iter=k, converged=False, lr=lr, gaps=gaps)
        if gap < tol:
            return dict(w=w, n_iter=k, converged=True, lr=lr, gaps=gaps)
        w = w - lr * prob.grad(w)
    return dict(w=w, n_iter=max_iter, converged=False, lr=lr, gaps=gaps)
