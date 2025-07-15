import numpy as np
from tqdm import tqdm
import json
import matplotlib.pyplot as plt

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
        # Calc zone of influence of each cluster center (echantillon de l'espace latent)
        # randomly generate points within the zone of influence
        ...

    def kmeans_compression(self, point):
        index = self.inference(point)
        cluster = self.cluster_centers[index]

        # Calculer la distance euclidienne
        distance = np.linalg.norm(point - cluster)

        return index, distance

    def kmeans_decompression(self, index, distance):
        cluster = self.cluster_centers[index]
        
        # Générer une direction aléatoire
        random_direction = np.random.randn(*cluster.shape)
        normalized_random_direction = random_direction / np.linalg.norm(random_direction)

        # Reconstituer le point à partir du centre du cluster et de la direction aléatoire
        reconstructed_point = cluster + normalized_random_direction * distance
        return reconstructed_point