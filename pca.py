import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits

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
        n = A.shape[1]
        v = np.random.rand(n)
        v /= np.linalg.norm(v)
        for _ in range(self.max_iter):
            Av = A @ v
            v_new = Av / np.linalg.norm(Av)
            if np.linalg.norm(v - v_new) < self.tol:
                break
            v = v_new
        return v

    @staticmethod
    def deflate_matrix(A, v):
        return A - (A @ v).reshape(-1, 1) @ v.reshape(1, -1)

    def fit(self, X):
        # Centrage
        X_centered, self.mean = self.center_data(X)
        # Matrice de covariance
        cov_matrix = self.compute_covariance_matrix(X_centered)

        A = cov_matrix.copy()
        components = []
        for _ in range(self.n_components):
            v = self.power_iteration(A)
            components.append(v)
            A = self.deflate_matrix(A, v)

        self.components = np.array(components).T  # shape (features, n_components)

    def transform(self, X):
        X_centered = X - self.mean
        return X_centered @ self.components

# ======== Utilisation ========
digits = load_digits()
X = digits.data
y = digits.target

pca = FullScratchPCA(n_components=2)
pca.fit(X)
X_proj = pca.transform(X)

plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_proj[:, 0], X_proj[:, 1], c=y, cmap='tab10', s=15, alpha=0.7)
plt.xlabel("Composante Principale 1")
plt.ylabel("Composante Principale 2")
plt.title("PCA from scratch - Projection 2D des chiffres")
plt.colorbar(scatter, label="Classe (chiffre)")
plt.grid(True)
plt.tight_layout()
plt.show()