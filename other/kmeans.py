import numpy as np
from tqdm import tqdm

def kmeans(X, n_clusters=10, max_iter=300):
    # Randomly initialize cluster centers
    cluster_centers = np.random.rand(n_clusters, X.shape[1])
    labels = np.zeros(X.shape[0], dtype=int)

    # Loop until convergence
    for _ in tqdm(range(max_iter), desc="K-Means Iterations"):
        # Assign each point to the nearest cluster center
        for i, point in enumerate(X):
            distances = np.linalg.norm(cluster_centers - point, axis=1)
            labels[i] = np.argmin(distances)

        # Calculate new cluster centers
        new_cluster_centers = np.array([X[labels == k].mean(axis=0) for k in range(n_clusters)])

        # If the cluster centers do not change, break the loop
        if np.allclose(cluster_centers, new_cluster_centers):
            break

        cluster_centers = new_cluster_centers

    # Return the cluster centers and labels
    return cluster_centers, labels