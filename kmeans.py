import numpy as np
from numba import cuda
from tqdm import tqdm
import json
import matplotlib.pyplot as plt

@cuda.jit
def _assign_clusters(X, cluster_centers, labels):
    """
    Kernel CUDA pour assigner chaque point au cluster le plus proche.
    """
    idx = cuda.grid(1)  # Identifiant global du thread

    if idx < X.shape[0]:
        min_dist = float('inf')
        min_cluster = -1

        for k in range(cluster_centers.shape[0]):
            dist = 0.0
            for d in range(X.shape[1]):
                diff = X[idx, d] - cluster_centers[k, d]
                dist += diff * diff

            if dist < min_dist:
                min_dist = dist
                min_cluster = k

        labels[idx] = min_cluster

@cuda.jit
def _update_centroids(X, labels, new_centroids, n_clusters):
    """
    Kernel CUDA pour calculer les nouveaux centroïdes.
    """
    idx = cuda.grid(1)  # Identifiant global du thread

    if idx < n_clusters:
        count = 0
        centroid_sum = cuda.local.array(256, dtype=np.float32)

        for d in range(X.shape[1]):
            centroid_sum[d] = 0.0

        for i in range(X.shape[0]):
            if labels[i] == idx:
                for d in range(X.shape[1]):
                    centroid_sum[d] += X[i, d]
                count += 1

        if count > 0:
            for d in range(X.shape[1]):
                new_centroids[idx, d] = centroid_sum[d] / count
        else:
            for d in range(X.shape[1]):
                new_centroids[idx, d] = 0.0

class Kmeans:
    def __init__(self):
        self.cluster_centers = np.ndarray([])
        self.labels = np.ndarray([])
        self.cluster_std = None
        if cuda.is_available():
            self.device = "gpu"
            print("GPU available for Numba!")
        else:
            self.device = "cpu"
            print("No GPU available, using CPU.")

    def train(self, X, n_clusters=10, max_iter=300, tol=1e-4):
        self.cluster_centers = np.random.rand(n_clusters, X.shape[1])
        self.labels = np.zeros(X.shape[0], dtype=int)
        self.cluster_std = np.zeros((n_clusters, X.shape[1]))

        if self.device == "gpu":
            # Copier les données sur le GPU
            d_X = cuda.to_device(X.astype(np.float32))
            d_cluster_centers = cuda.to_device(self.cluster_centers.astype(np.float32))
            d_labels = cuda.to_device(self.labels)

            threads_per_block = 256
            blocks_per_grid = (X.shape[0] + threads_per_block - 1) // threads_per_block

            for _ in range(max_iter):
                # Appel du kernel optimisé pour assigner les clusters
                _assign_clusters[blocks_per_grid, threads_per_block](d_X, d_cluster_centers, d_labels)

                # Calcul des nouveaux centroïdes sur le GPU
                new_cluster_centers = cuda.device_array_like(d_cluster_centers)
                _update_centroids[blocks_per_grid, threads_per_block](d_X, d_labels, new_cluster_centers, n_clusters)

                # Copier les centroïdes sur le CPU pour vérifier la convergence
                new_cluster_centers_host = new_cluster_centers.copy_to_host()
                if np.allclose(self.cluster_centers, new_cluster_centers_host, atol=tol):
                    break

                # Mise à jour des centroïdes sur le GPU
                d_cluster_centers = cuda.to_device(new_cluster_centers_host)
                self.cluster_centers = new_cluster_centers_host

        else:
            for _ in tqdm(range(max_iter), desc="K-Means Iterations"):
                # Assign each point to the nearest cluster center
                distances = np.linalg.norm(X[:, np.newaxis] - self.cluster_centers, axis=2)
                self.labels = np.argmin(distances, axis=1)

                # Calculate new cluster centers
                new_cluster_centers = np.array([X[self.labels == k].mean(axis=0) for k in range(n_clusters)])

                # If the cluster centers do not change, break the loop
                if np.allclose(self.cluster_centers, new_cluster_centers, atol=tol):
                    # Calculate standard deviation for each cluster
                    for k in range(n_clusters):
                        points_in_cluster = X[self.labels == k]
                        if len(points_in_cluster) > 0:
                            self.cluster_std[k] = np.std(points_in_cluster, axis=0)
                    break

                self.cluster_centers = new_cluster_centers

        # Return the cluster centers and labels
        return self.cluster_centers, self.labels
    
    def save_info(self, path="kmeans_info.json"):
        res = {
            "cluster_centers": self.cluster_centers.tolist(),
            "labels": self.labels.tolist(),
            "cluster_std": self.cluster_std.tolist()  
        }
        with open(path, 'w') as file:
            json.dump(res, file)
    
    def load_info(self, path="kmeans_info.json"):
        with open(path, 'r') as file:
            res = json.load(file)
        self.cluster_centers = np.array(res["cluster_centers"])
        self.labels = np.array(res["labels"])
        self.cluster_std = np.array(res["cluster_std"])

    def inference(self, point):
        distances = np.linalg.norm(self.cluster_centers - point, axis=1)
        return np.argmin(distances)

    def display(self, point, titre="Original", largeur=28, hauteur=28):
        if point.ndim == 1:
            plt.imshow(point.reshape(largeur, hauteur), cmap='gray')
            plt.title(titre)
            plt.axis('off')
            plt.show()
        else:
            # Afficher deux points côte à côte
            fig, axes = plt.subplots(1, 2, figsize=(8, 4))
            for i, ax in enumerate(axes):
                ax.imshow(point[i].reshape(hauteur, largeur), cmap='gray')
                if i==0:
                    ax.set_title("Original")
                else:   
                    ax.set_title(f"{titre} {i+1}")
                ax.axis('off')
            plt.show()


    def kmeans_generation(self, index):
        cluster = self.cluster_centers[index]
        std = self.cluster_std[index]

         # Générer une direction aléatoire fixe
        random_direction = np.random.randn(*cluster.shape)
        normalized_random_direction = random_direction / np.linalg.norm(random_direction)

        reconstructed_point = np.where(std == 0, cluster, cluster + (normalized_random_direction  * std))
        return reconstructed_point
    
    def kmeans_compression(self, point):
        index = self.inference(point)
        cluster = self.cluster_centers[index]
        std = self.cluster_std[index]

        # Calculer la distance pondérée par l'écart type
        weighted_distance = np.linalg.norm((point - cluster) / (std + 1e-8))  # Éviter la division par zéro

        return index, weighted_distance

    def kmeans_decompression(self, index, distance):
        cluster = self.cluster_centers[index]
        std = self.cluster_std[index]

        # Générer une direction aléatoire fixe
        random_direction = np.random.randn(*cluster.shape)
        normalized_random_direction = random_direction / np.linalg.norm(random_direction)

        # Reconstituer le point en tenant compte de l'écart type
        reconstructed_point = np.where(std == 0, cluster, cluster + (normalized_random_direction * distance * std))
        return reconstructed_point