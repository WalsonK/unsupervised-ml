# full_scratch_pca.py

import numpy as np

class FullScratchPCA:
    def __init__(self, n_components, max_iter=1000, tol=1e-6):
        self.n_components = n_components
        self.mean = None
        self.components = None
        self.max_iter = max_iter
        self.tol = tol

    @staticmethod
    def center_data(X):
        mean = np.mean(X, axis=0)
        X_centered = X - mean
        return X_centered, mean

    @staticmethod
    def compute_covariance_matrix(X_centered):
        n_samples = X_centered.shape[0]
        return (1 / (n_samples - 1)) * (X_centered.T @ X_centered)

    def power_iteration(self, A):
        n = A.shape[0]
        v = np.random.randn(n)
        v /= np.linalg.norm(v)
        for _ in range(self.max_iter):
            Av = A @ v
            v_new = Av / np.linalg.norm(Av)
            if np.linalg.norm(v - v_new) < self.tol:
                break
            v = v_new
        return v

    def deflate_matrix(self, A, v):
        lambda_ = v.T @ A @ v
        return A - lambda_ * np.outer(v, v)

    def fit(self, X):
        X_centered, self.mean = self.center_data(X)
        cov_matrix = self.compute_covariance_matrix(X_centered)

        A = cov_matrix.copy()
        components = []
        for _ in range(self.n_components):
            v = self.power_iteration(A)
            v /= np.linalg.norm(v)
            components.append(v)
            A = self.deflate_matrix(A, v)

        self.components = np.array(components).T

    def transform(self, X):
        X_centered = X - self.mean
        return X_centered @ self.components

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
