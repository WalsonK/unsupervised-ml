import numpy as np
from sklearn.datasets import fetch_openml


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
            print("You chose KMeans clustering.")
        elif algo == 'pca':
            print("You chose PCA for dimensionality reduction.")
        elif algo == 'autoencoder':
            print("You chose Autoencoder for unsupervised learning.")
