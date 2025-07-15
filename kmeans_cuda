import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm  # For progress bar

class KMeansCUDA:
    def __init__(self, n_clusters, max_iter=300, tol=1e-4):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X):
        # Move data to GPU
        X = torch.tensor(X, device='cuda')
        np.random.seed(42)
        random_indices = np.random.choice(X.shape[0], self.n_clusters, replace=False)
        self.centroids = X[random_indices]

        for iteration in tqdm(range(self.max_iter), desc="Training KMeans"):
            # Assign clusters
            self.labels = self._assign_clusters(X)

            # Update centroids
            new_centroids = self._update_centroids(X)

            # Check for convergence
            if torch.all(torch.abs(new_centroids - self.centroids) < self.tol):
                print(f"Converged at iteration {iteration}")
                break

            self.centroids = new_centroids

    def _assign_clusters(self, X):
        distances = torch.cdist(X, self.centroids)
        return torch.argmin(distances, axis=1)

    def _update_centroids(self, X):
        centroids = []
        for i in range(self.n_clusters):
            cluster_points = X[self.labels == i]
            if cluster_points.shape[0] > 0:
                centroids.append(cluster_points.mean(axis=0))
            else:
                centroids.append(torch.zeros(X.shape[1], device='cuda'))
        return torch.stack(centroids)

    def predict(self, X):
        X = torch.tensor(X, device='cuda')
        distances = torch.cdist(X, self.centroids)
        return torch.argmin(distances, axis=1)

# Load MNIST dataset
mnist = fetch_openml('mnist_784', version=1)
X = mnist.data
y = mnist.target

# Preprocess data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train KMeansCUDA
kmeans = KMeansCUDA(n_clusters=10)
kmeans.fit(X_scaled)

# Plot results
def plot_clusters(X, labels):
    X = X.cpu().numpy()  # Move data to CPU for plotting
    labels = labels.cpu().numpy()  # Move labels to CPU for plotting
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=1)
    plt.title('KMeans Clustering on MNIST (CUDA)')
    plt.show()

# Use only the first two features for visualization
X_visual = X_scaled[:, :2]
plot_clusters(torch.tensor(X_visual, device='cuda'), kmeans.predict(X_scaled))
