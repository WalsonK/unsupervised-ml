import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits


class FullScratchPCA:
    def __init__(self, n_components, max_iter=1000, tol=1e-6):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.mean = None
        self.components = None
        self.eigenvalues = None

    def center_data(self, X):
        """Centre les données"""
        self.mean = np.mean(X, axis=0)
        return X - self.mean

    def compute_covariance_matrix(self, X_centered):
        """Calcule la matrice de covariance"""
        n_samples = X_centered.shape[0]
        return (1 / (n_samples - 1)) * (X_centered.T @ X_centered)

    def power_iteration(self, A):
        """Power iteration pour trouver le vecteur propre dominant"""
        n = A.shape[1]
        v = np.random.rand(n)
        v /= np.linalg.norm(v)

        eigenvalue = 0
        for _ in range(self.max_iter):
            Av = A @ v
            eigenvalue = np.dot(v, Av)  # Quotient de Rayleigh
            v_new = Av / np.linalg.norm(Av)
            if np.linalg.norm(v - v_new) < self.tol:
                break
            v = v_new

        return v, eigenvalue

    def deflate_matrix(self, A, v, eigenvalue):
        """Déflation de la matrice"""
        return A - eigenvalue * np.outer(v, v)

    def fit(self, X):
        """Ajuste la PCA aux données"""
        X_centered = self.center_data(X)
        cov_matrix = self.compute_covariance_matrix(X_centered)

        A = cov_matrix.copy()
        components = []
        eigenvalues = []

        for _ in range(self.n_components):
            v, eigenvalue = self.power_iteration(A)
            components.append(v)
            eigenvalues.append(eigenvalue)
            A = self.deflate_matrix(A, v, eigenvalue)

        self.components = np.array(components).T  # shape (features, n_components)
        self.eigenvalues = np.array(eigenvalues)

        # Calcul variance expliquée
        total_variance = np.trace(cov_matrix)
        self.explained_variance_ratio = self.eigenvalues / total_variance

        return self

    def compress(self, X):
        """Compression : projection dans l'espace réduit"""
        if self.components is None:
            raise ValueError("PCA not fitted. Call fit() first.")
        X_centered = X - self.mean
        return X_centered @ self.components

    def decompress(self, X_compressed):
        """Décompression : retour dans l'espace original"""
        if self.components is None:
            raise ValueError("PCA not fitted. Call fit() first.")
        return X_compressed @ self.components.T + self.mean

    def transform(self, X):
        """Alias de compress"""
        return self.compress(X)

    def inverse_transform(self, X_compressed):
        """Alias de decompress"""
        return self.decompress(X_compressed)

    def reconstruction_error(self, X):
        """Calcule l'erreur de reconstruction"""
        X_compressed = self.compress(X)
        X_reconstructed = self.decompress(X_compressed)
        return np.mean((X - X_reconstructed) ** 2)


# ======== Test automatique ========
if __name__ == "__main__":
    digits = load_digits()
    X = digits.data / 16.0  # Normalisation
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
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()