#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from tqdm import tqdm


# ----------------------------------------------------------------------
# 1)  K-Means implementation that lives on the GPU
# ----------------------------------------------------------------------
class KMeansCUDA:
    def __init__(self, n_clusters: int, max_iter: int = 300, tol: float = 1e-4,
                 device: torch.device | str = 'cuda'):
        self.n_clusters = n_clusters
        self.max_iter   = max_iter
        self.tol        = tol
        self.device     = torch.device(device)

        self.centroids: torch.Tensor | None = None
        self.labels:    torch.Tensor | None = None

    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray):
        """
        X : numpy array of shape (n_samples, n_features)
        """
        X = torch.as_tensor(X, dtype=torch.float32, device=self.device)

        # ------- initialise centroids with K distinct random samples ----
        rnd_idx = torch.randperm(X.shape[0], device=self.device)[: self.n_clusters]
        self.centroids = X[rnd_idx].clone()

        # ------------------------ main loop ----------------------------
        for it in tqdm(range(self.max_iter), desc="Training KMeans"):
            self.labels = self._assign_clusters(X)
            new_centroids = self._update_centroids(X)

            # convergence test -----------------------------------------
            if torch.allclose(new_centroids, self.centroids, atol=self.tol):
                print(f"Converged at iteration {it}")
                break
            self.centroids = new_centroids

        return self

    # ------------------------------------------------------------------
    def _assign_clusters(self, X: torch.Tensor) -> torch.Tensor:
        # pairwise distances sample-to-centroid (shape n_samples × K)
        dists = torch.cdist(X, self.centroids)       # uses GPU BLAS
        return torch.argmin(dists, dim=1)

    # ------------------------------------------------------------------
    def _update_centroids(self, X: torch.Tensor) -> torch.Tensor:
        centroids = []
        for k in range(self.n_clusters):
            cluster_pts = X[self.labels == k]
            if len(cluster_pts) > 0:
                centroids.append(cluster_pts.mean(0))
            else:                           # empty cluster → keep old
                centroids.append(self.centroids[k])
        return torch.stack(centroids)

    # ------------------------------------------------------------------
    @torch.inference_mode()
    def predict(self, X: np.ndarray) -> np.ndarray:
        X = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        labels = self._assign_clusters(X)
        return labels.cpu().numpy()


# ----------------------------------------------------------------------
# 2)  Load and preprocess the MNIST data
# ----------------------------------------------------------------------
print("Downloading / loading MNIST …")
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X     = mnist.data.astype(np.float32)       # (70 000 × 784)
y     = mnist.target.astype(int)            # not used for clustering

print("Standardising features …")
scaler = StandardScaler(copy=False)
X_scaled = scaler.fit_transform(X)          # stays numpy


# ----------------------------------------------------------------------
# 3)  Fit the CUDA K-Means
# ----------------------------------------------------------------------
kmeans = KMeansCUDA(n_clusters=10, max_iter=100, tol=1e-4)
kmeans.fit(X_scaled)


# ----------------------------------------------------------------------
# 4)  2-D PCA projection of both samples and centroids
# ----------------------------------------------------------------------
print("Computing PCA projection …")
pca = PCA(n_components=2, random_state=0)

X_2d = pca.fit_transform(X_scaled)                       # (70 000 × 2)

centroids_cpu = kmeans.centroids.cpu().numpy()
centroids_2d  = pca.transform(centroids_cpu)             # (10 × 2)

labels_cpu = kmeans.labels.cpu().numpy()                 # (70 000,)


# ----------------------------------------------------------------------
# 5)  Plot (use a subset of points for clarity)
# ----------------------------------------------------------------------
N   = 70_000                                 # how many samples to display
idx = np.random.choice(len(X_2d), N, replace=False)

plt.figure(figsize=(8, 8))
plt.scatter(X_2d[idx, 0], X_2d[idx, 1],
            c=labels_cpu[idx], cmap='tab10',
            s=3, alpha=0.6, linewidths=0)

plt.scatter(centroids_2d[:, 0], centroids_2d[:, 1],
            c='k', marker='X', s=200, edgecolors='white',
            linewidths=2, label='Cluster centres')

plt.title('K-Means clusters on MNIST (PCA projection)')
plt.xlabel('PCA component 1')
plt.ylabel('PCA component 2')
plt.legend(loc='best')
plt.tight_layout()
plt.show()