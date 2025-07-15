import numpy as np
from sklearn.datasets import fetch_openml
from kmeans import Kmeans
import matplotlib.pyplot as plt


# Charger et préparer les données
mnist = fetch_openml('mnist_784', version=1)
X, y = mnist.data, mnist.target

if __name__ == '__main__':
    print("MNIST dataset loaded successfully.")
    algo = input("Which algorithm would you like to use? (kmeans/pca/autoencoder): ").strip().lower()

    if algo not in ['kmeans', 'pca', 'autoencoder']:
        print("Invalid algorithm choice. Please choose from kmeans, pca, or autoencoder.")

    else:
        if algo == 'kmeans':
            # traitement des données
            X = X.to_numpy()
            # algorithm 
            model = Kmeans()
            cluster_center, labels = model.train(X, n_clusters=10, max_iter=100)
        elif algo == 'pca':
            print("You chose PCA for dimensionality reduction.")
        elif algo == 'autoencoder':
            print("You chose Autoencoder for unsupervised learning.")
