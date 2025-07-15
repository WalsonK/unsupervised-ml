import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits

import numpy as np

class FullScratchPCA:
    def __init__(self, n_components, max_iter=1000, tol=1e-6):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.mean = None
        self.components = None
        self.eigenvalues = None

    def center_data(self, X):
        self.mean = np.mean(X, axis=0)
        return X - self.mean

    def compute_covariance_matrix(self, X_centered):
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

    def deflate_matrix(self, A, v, eigenvalue):
        return A - eigenvalue * np.outer(v, v)

    def fit(self, X):
        X_centered = self.center_data(X)
        cov_matrix = self.compute_covariance_matrix(X_centered)

        A = cov_matrix.copy()
        components = []
        eigenvalues = []

        for _ in range(self.n_components):
            v = self.power_iteration(A)
            eigenvalue = v.T @ A @ v
            components.append(v)
            eigenvalues.append(eigenvalue)
            A = self.deflate_matrix(A, v, eigenvalue)

        self.components = np.array(components).T  # shape (features, n_components)
        self.eigenvalues = np.array(eigenvalues)

        # Affichage des valeurs propres
        print("Valeurs propres (approx. via power iteration) :")
        for i, val in enumerate(self.eigenvalues):
            print(f"λ{i+1} = {val:.4f}")

    def compress(self, X):
        """Compression : projection dans l’espace réduit"""
        X_centered = X - self.mean
        return X_centered @ self.components

    def decompress(self, X_compressed):
        """Décompression : retour dans l’espace original"""
        return X_compressed @ self.components.T + self.mean

    def transform(self, X):
        """Alias de compress"""
        return self.compress(X)

    def inverse_transform(self, X_compressed):
        """Alias de decompress"""
        return self.decompress(X_compressed)

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