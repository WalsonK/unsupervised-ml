import numpy as np
from tqdm import tqdm
import json

class Kmeans:
    def __init__(self):
        cluster_centers = np.ndarray([])
        labels = np.ndarray([])

    def train(self, X, n_clusters=10, max_iter=300):
        self.cluster_centers = np.random.rand(n_clusters, X.shape[1])
        self.labels = np.zeros(X.shape[0], dtype=int)

        # Loop until convergence
        for _ in tqdm(range(max_iter), desc="K-Means Iterations"):
            # Assign each point to the nearest cluster center
            for i, point in enumerate(X):
                distances = np.linalg.norm(self.cluster_centers - point, axis=1)
                self.labels[i] = np.argmin(distances)

            # Calculate new cluster centers
            new_cluster_centers = np.array([X[self.labels == k].mean(axis=0) for k in range(n_clusters)])

            # If the cluster centers do not change, break the loop
            if np.allclose(self.cluster_centers, new_cluster_centers):
                break

            self.cluster_centers = new_cluster_centers

        # Return the cluster centers and labels
        return self.cluster_centers, self.labels
    
    def save_info(self, path="kmeans_info.json"):
        res = {
            "cluster_centers": self.cluster_centers.tolist(),
            "labels": self.labels.tolist()  
        }
        with open(path, 'w') as file:
            json.dump(res, file)
    
    def load_info(self, path="kmeans_info.json"):
        with open(path, 'r') as file:
            res = json.load(file)
        self.cluster_centers = np.array(res["cluster_centers"])
        self.labels = np.array(res["labels"])

    def kmeans_generation():
        # Calc zone of influence of each cluster center (echantillon de l'espace latent)
        # randomly generate points within the zone of influence
        ...

    def kmeans_compression():
        # Calc euclidean distance between point and cluster center
        # Send the vector of distances and the cluster center
        ...

    def kmeans_decompression():
        # Receive the vector of distances and the cluster center
        # Reconstruct the point by adding the distance to the cluster center
        ...