import numpy as np


class Scaler:
    name = "scaler"

    def fit(self, X):
        raise NotImplementedError

    def transform(self, X):
        return (X - self.mu) @ self.A

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def coef_to_raw(self, w_b):
        d = len(w_b) - 1
        w_raw = self.A @ w_b[:d]
        return np.append(w_raw, w_b[d] - self.mu @ w_raw)

    def penalty_matrix(self):
        return self.A.T @ self.A


class Identity(Scaler):
    name = "none"

    def fit(self, X):
        d = X.shape[1]
        self.mu, self.A = np.zeros(d), np.eye(d)
        return self


class Standardizer(Scaler):
    name = "standardize"

    def fit(self, X):
        self.mu = X.mean(axis=0)
        self.A = np.diag(1.0 / X.std(axis=0))
        return self


class MinMax(Scaler):
    name = "min-max"

    def fit(self, X):
        self.mu = X.min(axis=0)
        self.A = np.diag(1.0 / (X.max(axis=0) - X.min(axis=0)))
        return self


class Whitener(Scaler):
    name = "whiten"

    def fit(self, X, eps=1e-12):
        self.mu = X.mean(axis=0)
        Xc = X - self.mu
        cov = Xc.T @ Xc / X.shape[0]
        lam, V = np.linalg.eigh(cov)
        self.A = V @ np.diag(1.0 / np.sqrt(lam + eps)) @ V.T
        return self


class PCAWhitener(Scaler):
    name = "pca-whiten"

    def fit(self, X, eps=1e-12):
        self.mu = X.mean(axis=0)
        Xc = X - self.mu
        lam, V = np.linalg.eigh(Xc.T @ Xc / X.shape[0])
        self.A = V @ np.diag(1.0 / np.sqrt(lam + eps))
        return self


ALL_SCALERS = [Identity, Standardizer, MinMax, Whitener, PCAWhitener]
