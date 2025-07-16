# main.py

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
import torch

from kmeans.kmeans_visu.kmeans_cuda import KMeansCUDA
from kmeans.kmeans_visu.plot_kmeans import FullScratchPCA


def kmeans_pipeline(X, use_cuda=True):
    print("Standardizing features …")
    scaler = StandardScaler(copy=False)
    X = scaler.fit_transform(X.astype(np.float32))

    device = 'cuda' if use_cuda and torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Run KMeans
    print("Running KMeans …")
    kmeans = KMeansCUDA(n_clusters=10, max_iter=100, tol=1e-4, device=device)
    kmeans.fit(X)

    # PCA
    print("Computing PCA projection …")
    pca = FullScratchPCA(n_components=2)
    pca.fit(X)
    X_2d = pca.transform(X)

    centroids_2d = pca.transform(kmeans.centroids.cpu().numpy())
    labels = kmeans.labels.cpu().numpy()

    # Plot
    N = 10_000
    idx = np.random.choice(len(X), N, replace=False)

    plt.figure(figsize=(8, 8))
    plt.scatter(X_2d[idx, 0], X_2d[idx, 1], c=labels[idx], cmap='tab10', s=3, alpha=0.6, linewidths=0)
    plt.scatter(centroids_2d[:, 0], centroids_2d[:, 1], c='k', marker='X', s=200,
                edgecolors='white', linewidths=2, label='Cluster centres')

    plt.title('K-Means clusters on MNIST (PCA projection — from scratch)')
    plt.xlabel('Principal component 1')
    plt.ylabel('Principal component 2')
    plt.legend()
    plt.tight_layout()
    plt.show()


# ================================
# Main control logic
# ================================
if __name__ == '__main__':
    print("Loading MNIST dataset …")
    mnist = fetch_openml('mnist_784', version=1)
    X, y = mnist.data, mnist.target.astype(int)

    print("MNIST dataset loaded successfully.")
    algo = input("Which algorithm would you like to use? (kmeans/pca/autoencoder): ").strip().lower()

    if algo not in ['kmeans', 'pca', 'autoencoder']:
        print("Invalid algorithm choice. Please choose from kmeans, pca, or autoencoder.")
    else:
        if algo == 'kmeans':
            use_cuda = input("Use CUDA for KMeans? (y/n): ").strip().lower() == 'y'
            kmeans_pipeline(X, use_cuda)

        elif algo == 'pca':
            print("You chose PCA for dimensionality reduction.")
            # You can insert standalone PCA logic here if desired

        elif algo == 'autoencoder':
            print("You chose Autoencoder for unsupervised learning.")
            # You can insert your autoencoder logic here
