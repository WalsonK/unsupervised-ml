# kmeans_cuda.py

import numpy as np
import torch
from tqdm import tqdm

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
