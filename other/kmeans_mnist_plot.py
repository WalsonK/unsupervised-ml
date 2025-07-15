#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Cluster the MNIST digits with a GPU K-Means, then visualise the result with our
home-made PCA implemented in `my_pca.py`.
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# ---- our own PCA ----------------------------------------------------
from my_pca import PCA


# ──────────────────────────────────────────────────────────────────────
# 1)  CUDA K-Means
# ──────────────────────────────────────────────────────────────────────
class KMeansCUDA:
    def __init__(self, n_clusters: int, max_iter: int = 300,
                 tol: float = 1e-4, device: str | torch.device = 'cuda'):
        self.n_clusters = n_clusters
        self.max_iter   = max_iter
        self.tol        = tol
        self.device     = torch.device(device)

        self.centroids: torch.Tensor | None = None
        self.labels:    torch.Tensor | None = None

    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray):
        X = torch.as_tensor(X, dtype=torch.float32, device=self.device)

        # random initial centres ---------------------------------------
        perm = torch.randperm(X.shape[0], device=self.device)
        self.centroids = X[perm[:self.n_clusters]].clone()

        # main loop -----------------------------------------------------
        for it in tqdm(range(self.max_iter), desc="Training K-Means"):
            self.labels = self._assign_clusters(X)
            new_centroids = self._update_centroids(X)

            # convergence? ---------------------------------------------
            if torch.allclose(new_centroids, self.centroids, atol=self.tol):
                print(f"Converged at iteration {it}")
                break
            self.centroids = new_centroids

        return self

    # ------------------------------------------------------------------
    def _assign_clusters(self, X: torch.Tensor) -> torch.Tensor:
        dists = torch.cdist(X, self.centroids)
        return torch.argmin(dists, dim=1)

    # ------------------------------------------------------------------
    def _update_centroids(self, X: torch.Tensor) -> torch.Tensor:
        centroids = []
        for k in range(self.n_clusters):
            pts = X[self.labels == k]
            if len(pts) > 0:
                centroids.append(pts.mean(0))
            else:                       # rare empty cluster
                centroids.append(self.centroids[k])
        return torch.stack(centroids)

    # ------------------------------------------------------------------
    @torch.inference_mode()
    def predict(self, X: np.ndarray) -> np.ndarray:
        X = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        return self._assign_clusters(X).cpu().numpy()


# ──────────────────────────────────────────────────────────────────────
# 2)  Load + scale MNIST
# ──────────────────────────────────────────────────────────────────────
print("Downloading / loading MNIST …")
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X_raw = mnist.data.astype(np.float32)   # (70 000 × 784)
y      = mnist.target.astype(int)       # not used

print("Standardising features …")
scaler = StandardScaler(copy=False)
X = scaler.fit_transform(X_raw)         # keeps it as float32 numpy array


# ──────────────────────────────────────────────────────────────────────
# 3)  Run CUDA K-Means
# ──────────────────────────────────────────────────────────────────────
kmeans = KMeansCUDA(n_clusters=10, max_iter=100, tol=1e-4)
kmeans.fit(X)


# ──────────────────────────────────────────────────────────────────────
# 4)  Project to 2-D with OUR PCA
# ──────────────────────────────────────────────────────────────────────
print("Computing PCA projection …")
pca = PCA(n_components=2)
X_2d = pca.fit_transform(X)                             # (70 000 × 2)

centroids_cpu = kmeans.centroids.cpu().numpy()
centroids_2d  = pca.transform(centroids_cpu)            # (10 × 2)

labels = kmeans.labels.cpu().numpy()                    # (70 000,)


# ──────────────────────────────────────────────────────────────────────
# 5)  Plot
# ──────────────────────────────────────────────────────────────────────
N   = 10_000                                # subset for readability
idx = np.random.choice(len(X_2d), N, replace=False)

plt.figure(figsize=(8, 8))
plt.scatter(X_2d[idx, 0], X_2d[idx, 1],
            c=labels[idx], cmap='tab10',
            s=3, alpha=0.6, linewidths=0)

plt.scatter(centroids_2d[:, 0], centroids_2d[:, 1],
            c='k', marker='X', s=200, edgecolors='white', linewidths=2,
            label='Cluster centres')

plt.title('K-Means clusters on MNIST (PCA projection — from scratch)')
plt.xlabel('Principal component 1')
plt.ylabel('Principal component 2')
plt.legend()
plt.tight_layout()
plt.show()