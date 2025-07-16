#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Cluster MNIST digits using custom GPU K-Means and visualize with FullScratchPCA.
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from full_scratch_pca import FullScratchPCA  # <- Your custom PCA

# ──────────────────────────────────────────────────────────────────────
# Custom GPU-based KMeans
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

    def fit(self, X: np.ndarray):
        X = torch.as_tensor(X, dtype=torch.float32, device=self.device)

        perm = torch.randperm(X.shape[0], device=self.device)
        self.centroids = X[perm[:self.n_clusters]].clone()

        for it in tqdm(range(self.max_iter), desc="Training K-Means"):
            self.labels = self._assign_clusters(X)
            new_centroids = self._update_centroids(X)

            if torch.allclose(new_centroids, self.centroids, atol=self.tol):
                print(f"Converged at iteration {it}")
                break
            self.centroids = new_centroids

        return self

    def _assign_clusters(self, X: torch.Tensor) -> torch.Tensor:
        dists = torch.cdist(X, self.centroids)
        return torch.argmin(dists, dim=1)

    def _update_centroids(self, X: torch.Tensor) -> torch.Tensor:
        centroids = []
        for k in range(self.n_clusters):
            pts = X[self.labels == k]
            if len(pts) > 0:
                centroids.append(pts.mean(0))
            else:
                centroids.append(self.centroids[k])
        return torch.stack(centroids)

    @torch.inference_mode()
    def predict(self, X: np.ndarray) -> np.ndarray:
        X = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        return self._assign_clusters(X).cpu().numpy()

# ──────────────────────────────────────────────────────────────────────
# 1) Load and standardize MNIST
# ──────────────────────────────────────────────────────────────────────
print("Downloading / loading MNIST …")
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X_raw = mnist.data.astype(np.float32)
y     = mnist.target.astype(int)

print("Standardising features …")
scaler = StandardScaler(copy=False)
X = scaler.fit_transform(X_raw)

# ──────────────────────────────────────────────────────────────────────
# 2) Run custom KMeans on GPU
# ──────────────────────────────────────────────────────────────────────
kmeans = KMeansCUDA(n_clusters=10, max_iter=100, tol=1e-4)
kmeans.fit(X)

# ──────────────────────────────────────────────────────────────────────
# 3) PCA (2D projection)
# ──────────────────────────────────────────────────────────────────────
print("Computing PCA projection …")
pca = FullScratchPCA(n_components=2)
pca.fit(X)
X_2d = pca.transform(X)

centroids_cpu = kmeans.centroids.cpu().numpy()
centroids_2d  = pca.transform(centroids_cpu)

labels = kmeans.labels.cpu().numpy()

# ──────────────────────────────────────────────────────────────────────
# 4) Plot result
# ──────────────────────────────────────────────────────────────────────
N   = 70_000
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
