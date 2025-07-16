import numpy as np
import matplotlib

matplotlib.use('Agg')  # Backend qui sauvegarde au lieu d'afficher
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from pca import FullScratchPCA


def simple_generation_example():
    """
    Exemple simple : prendre une image et générer des variations
    """
    # Charger MNIST
    print("Loading data...")
    mnist = fetch_openml('mnist_784', version=1)
    X, y = mnist.data.values[:10000] / 255.0, mnist.target.values[:5000].astype(int)

    # Choisir une image du chiffre "7"
    digit_7_indices = np.where(y == 8)[0]
    original_image = X[digit_7_indices[0]]  # Première image du chiffre 7

    print(f"Image choisie: chiffre {y[digit_7_indices[0]]}")

    # Entraîner PCA
    pca = FullScratchPCA(n_components=50)
    pca.fit(X)

    # Comprimer l'image originale
    compressed = pca.compress(original_image.reshape(1, -1))[0]

    # Générer 5 variations en ajoutant du bruit
    variations = []

    for i in range(5):
        # Ajouter un peu de bruit aléatoire
        noise = np.random.normal(0, 0.15, compressed.shape)
        noisy_compressed = compressed + noise

        # Décomprimer pour obtenir une nouvelle image
        new_image = pca.decompress(noisy_compressed.reshape(1, -1))[0]
        new_image = np.clip(new_image, 0, 1)  # Garder valeurs entre 0 et 1
        variations.append(new_image)

    # Afficher les résultats
    plt.figure(figsize=(12, 4))

    # Image originale
    plt.subplot(1, 6, 1)
    plt.imshow(original_image.reshape(28, 28), cmap='gray')
    plt.title("Original")
    plt.axis('off')

    # 5 variations générées
    for i, variation in enumerate(variations):
        plt.subplot(1, 6, i + 2)
        plt.imshow(variation.reshape(28, 28), cmap='gray')
        plt.title(f"Variation {i + 1}")
        plt.axis('off')

    plt.suptitle("Génération PCA : 1 image → 5 variations")
    plt.tight_layout()

    # Sauvegarder au lieu d'afficher
    plt.savefig('pca_generation_example.png', dpi=150, bbox_inches='tight')
    print("📸 Image sauvegardée: pca_generation_example.png")
    plt.close()  # Fermer la figure

    print("✅ Génération terminée!")


# Exécuter l'exemple
simple_generation_example()