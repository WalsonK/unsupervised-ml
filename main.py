import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import fetch_openml

from pca import FullScratchPCA

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
            print("You chose PCA. What do you want to do?")
            print("1 - Visualisation 2D")
            print("2 - Compression / Décompression")
            print("3 - Synthèse de données (nuages générés)")
            sous_choix = input("Ton choix (1/2/3) : ").strip()

            if sous_choix == "1":
                print(">> Visualisation 2D")
                pca2d = FullScratchPCA(n_components=2)
                pca2d.fit(X)
                X_proj = pca2d.transform(X)

                plt.figure(figsize=(8, 6))
                plt.scatter(X_proj[:, 0], X_proj[:, 1], c=y, cmap='tab10', s=5)
                plt.title("PCA - Projection 2D des chiffres")
                plt.xlabel("Composante 1")
                plt.ylabel("Composante 2")
                plt.colorbar(label="Classe")
                plt.grid(True)
                plt.tight_layout()
                plt.show()

            elif sous_choix == "2":
                print(">> Compression et reconstruction")
                n_components = 50
                pca = FullScratchPCA(n_components=n_components)
                pca.fit(X)
                X_compressed = pca.compress(X)
                X_reconstructed = pca.decompress(X_compressed)

                mse = np.mean((X - X_reconstructed) ** 2)
                print(f"Erreur de reconstruction (MSE) : {mse:.5f}")

                plt.figure(figsize=(10, 4))
                for i in range(5):
                    plt.subplot(2, 5, i + 1)
                    plt.imshow(X[i].reshape(28, 28), cmap='gray')
                    plt.title("Original")
                    plt.axis('off')

                    plt.subplot(2, 5, i + 6)
                    plt.imshow(X_reconstructed[i].reshape(28, 28), cmap='gray')
                    plt.title("Reconstruit")
                    plt.axis('off')

                plt.suptitle(f"PCA - Reconstruction à {n_components} composantes")
                plt.tight_layout()
                plt.show()

            elif sous_choix == "3":
                print(">> Synthèse aléatoire via PCA")
                n_components = 50
                pca = FullScratchPCA(n_components=n_components)
                pca.fit(X)

                n_samples = 10
                X_synth_compressed = np.random.randn(n_samples, n_components)
                X_synth = pca.decompress(X_synth_compressed)

                plt.figure(figsize=(10, 2))
                for i in range(n_samples):
                    plt.subplot(1, n_samples, i + 1)
                    plt.imshow(X_synth[i].reshape(28, 28), cmap='gray')
                    plt.axis('off')
                plt.suptitle("Images synthétiques générées via PCA")
                plt.tight_layout()
                plt.show()

            else:
                print("Choix PCA invalide.")
        elif algo == 'autoencoder':
            print("You chose Autoencoder for unsupervised learning.")
