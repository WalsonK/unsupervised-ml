import numpy as np
import matplotlib

matplotlib.use('TkAgg')  # Force l'utilisation du backend TkAgg
from matplotlib import pyplot as plt
from sklearn.datasets import fetch_openml

from pca import FullScratchPCA

# Charger et préparer les données
print("Loading MNIST dataset...")
mnist = fetch_openml('mnist_784', version=1)
X, y = mnist.data.values, mnist.target.values.astype(int)  # Convert to numpy arrays

# Échantillonnage pour accélérer
n_samples = 10000
indices = np.random.choice(X.shape[0], n_samples, replace=False)
X = X[indices] / 255.0  # Normalisation
y = y[indices]

if __name__ == '__main__':
    print("MNIST dataset loaded successfully.")

    while True:
        print("\n" + "=" * 50)
        print("🎯 MENU PRINCIPAL")
        print("=" * 50)
        algo = input("Which algorithm would you like to use? (kmeans/pca/autoencoder/quit): ").strip().lower()

        if algo == 'quit':
            print("👋 Au revoir!")
            break

        if algo not in ['kmeans', 'pca', 'autoencoder']:
            print("❌ Invalid algorithm choice. Please choose from kmeans, pca, autoencoder or quit.")
            continue

        if algo == 'kmeans':
            print("🔴 You chose KMeans clustering.")
            print("⚠️ K-Means implementation not yet available.")

        elif algo == 'pca':
            while True:
                print("\n🔵 You chose PCA. What do you want to do?")
                print("1 - Visualisation 2D")
                print("2 - Compression / Décompression")
                print("3 - Synthèse de données (nuages générés)")
                print("4 - Retour au menu principal")
                sous_choix = input("Ton choix (1/2/3/4) : ").strip()

                if sous_choix == "4":
                    print("🔄 Retour au menu principal...")
                    break

                elif sous_choix == "1":
                    print(">> Visualisation 2D")
                    pca2d = FullScratchPCA(n_components=2)
                    pca2d.fit(X)
                    X_proj = pca2d.transform(X)

                    plt.figure(figsize=(10, 8))
                    scatter = plt.scatter(X_proj[:, 0], X_proj[:, 1], c=y, cmap='tab10', s=5, alpha=0.7)
                    plt.title("PCA - Projection 2D des chiffres")
                    plt.xlabel("Composante 1")
                    plt.ylabel("Composante 2")
                    plt.colorbar(scatter, label="Classe")
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()
                    plt.show()

                    print("✅ Visualisation terminée!")

                elif sous_choix == "2":
                    print(">> Comparaison multi-composantes")

                    # Liste des nombres de composantes à tester
                    components_list = [10, 20, 50, 70, 100, 130]

                    # Calcul des métriques pour chaque nombre de composantes
                    results = []
                    reconstructions = []

                    print("📊 Calcul en cours pour différents nombres de composantes...")

                    for n_comp in components_list:
                        print(f"   • {n_comp} composantes...")
                        pca = FullScratchPCA(n_components=n_comp)
                        pca.fit(X)
                        X_compressed = pca.compress(X)
                        X_reconstructed = pca.decompress(X_compressed)

                        mse = np.mean((X - X_reconstructed) ** 2)
                        compression_ratio = X.shape[1] / n_comp

                        results.append({
                            'n_components': n_comp,
                            'mse': mse,
                            'compression_ratio': compression_ratio
                        })

                        # Garder quelques reconstructions pour visualisation
                        reconstructions.append(X_reconstructed[:3])  # 3 premiers échantillons

                    # Affichage des résultats
                    print("\n📈 RÉSULTATS DE LA COMPARAISON:")
                    print("-" * 60)
                    print(f"{'Composantes':<12} {'Ratio':<10} {'MSE':<12} {'Qualité'}")
                    print("-" * 60)

                    for result in results:
                        quality = "Excellent" if result['mse'] < 0.01 else "Bon" if result['mse'] < 0.02 else "Moyen" if \
                        result['mse'] < 0.05 else "Faible"
                        print(
                            f"{result['n_components']:<12} {result['compression_ratio']:<10.1f} {result['mse']:<12.5f} {quality}")

                    # Graphique des métriques
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

                    # MSE en fonction du nombre de composantes
                    ax1.plot([r['n_components'] for r in results],
                             [r['mse'] for r in results],
                             'bo-', linewidth=2, markersize=8)
                    ax1.set_xlabel('Nombre de composantes')
                    ax1.set_ylabel('MSE (Erreur de reconstruction)')
                    ax1.set_title('Erreur vs Nombre de composantes')
                    ax1.grid(True, alpha=0.3)

                    # Ratio de compression
                    ax2.plot([r['n_components'] for r in results],
                             [r['compression_ratio'] for r in results],
                             'ro-', linewidth=2, markersize=8)
                    ax2.set_xlabel('Nombre de composantes')
                    ax2.set_ylabel('Ratio de compression')
                    ax2.set_title('Compression vs Nombre de composantes')
                    ax2.grid(True, alpha=0.3)

                    plt.tight_layout()
                    plt.show()

                    # Visualisation des reconstructions
                    fig, axes = plt.subplots(len(components_list) + 1, 3, figsize=(8, 2 * (len(components_list) + 1)))

                    # Images originales
                    for j in range(3):
                        axes[0, j].imshow(X[j].reshape(28, 28), cmap='gray')
                        axes[0, j].set_title(f'Original {j + 1}')
                        axes[0, j].axis('off')

                    # Images reconstruites pour chaque nombre de composantes
                    for i, (n_comp, reconstruction) in enumerate(zip(components_list, reconstructions)):
                        for j in range(3):
                            axes[i + 1, j].imshow(reconstruction[j].reshape(28, 28), cmap='gray')
                            axes[i + 1, j].set_title(f'{n_comp} comp.')
                            axes[i + 1, j].axis('off')

                    plt.suptitle('Comparaison visuelle des reconstructions PCA')
                    plt.tight_layout()
                    plt.show()

                    # Recommandations
                    print("\n💡 RECOMMANDATIONS:")
                    best_balance = min(results, key=lambda x: x['mse'] + 0.001 * x['compression_ratio'])
                    best_compression = max(results, key=lambda x: x['compression_ratio'])
                    best_quality = min(results, key=lambda x: x['mse'])

                    print(f"   • Meilleur équilibre qualité/compression: {best_balance['n_components']} composantes")
                    print(f"   • Meilleure compression: {best_compression['n_components']} composantes")
                    print(f"   • Meilleure qualité: {best_quality['n_components']} composantes")

                    print("✅ Comparaison multi-composantes terminée!")

                elif sous_choix == "3":
                    print(">> Génération de variations via PCA")

                    # Trois exemples avec différents chiffres
                    target_digits = [7, 8, 3]  # Chiffres à utiliser comme base

                    for idx, target_digit in enumerate(target_digits):
                        print(f"\n📊 Exemple {idx + 1}: Génération de variations du chiffre {target_digit}")

                        # Trouver une image du chiffre cible
                        digit_indices = np.where(y == target_digit)[0]
                        if len(digit_indices) == 0:
                            print(f"   ❌ Aucune image du chiffre {target_digit} trouvée")
                            continue

                        original_image = X[digit_indices[0]]
                        print(f"   • Image choisie: chiffre {target_digit}")

                        # Entraîner PCA avec 50 composantes
                        n_components = 50
                        pca = FullScratchPCA(n_components=n_components)
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

                        # Visualisation
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

                        plt.suptitle(f"Exemple {idx + 1}: Génération PCA - Chiffre {target_digit} → 5 variations")
                        plt.tight_layout()
                        plt.show()

                        # Calcul de métriques
                        reconstruction_error = np.mean(
                            (original_image - pca.decompress(pca.compress(original_image.reshape(1, -1)))[0]) ** 2)

                        print(f"   • Erreur de reconstruction: {reconstruction_error:.5f}")
                        print(f"   • Variance expliquée: {np.sum(pca.explained_variance_ratio):.3f}")
                        print(f"   • Ratio de compression: {X.shape[1] / n_components:.1f}x")

                        # Attendre l'utilisateur avant le prochain exemple
                        if idx < len(target_digits) - 1:
                            input("\nAppuyez sur Entrée pour voir l'exemple suivant...")

                    # Comparaison finale avec différents niveaux de bruit
                    print("\n" + "=" * 60)
                    print("📊 COMPARAISON FINALE - EFFET DU NIVEAU DE BRUIT")
                    print("=" * 60)

                    # Choisir une image de référence
                    ref_digit = 5
                    ref_indices = np.where(y == ref_digit)[0]
                    if len(ref_indices) > 0:
                        ref_image = X[ref_indices[0]]

                        # Entraîner PCA
                        pca = FullScratchPCA(n_components=50)
                        pca.fit(X)
                        compressed_ref = pca.compress(ref_image.reshape(1, -1))[0]

                        # Différents niveaux de bruit
                        noise_levels = [0.05, 0.1, 0.2, 0.3]

                        fig, axes = plt.subplots(len(noise_levels), 6, figsize=(15, 8))

                        for row, noise_level in enumerate(noise_levels):
                            # Image originale
                            axes[row, 0].imshow(ref_image.reshape(28, 28), cmap='gray')
                            axes[row, 0].set_title("Original")
                            axes[row, 0].axis('off')

                            # 5 variations avec ce niveau de bruit
                            for col in range(5):
                                noise = np.random.normal(0, noise_level, compressed_ref.shape)
                                noisy_compressed = compressed_ref + noise
                                variation = pca.decompress(noisy_compressed.reshape(1, -1))[0]
                                variation = np.clip(variation, 0, 1)

                                axes[row, col + 1].imshow(variation.reshape(28, 28), cmap='gray')
                                axes[row, col + 1].set_title(f"Var {col + 1}")
                                axes[row, col + 1].axis('off')

                        # Labels pour les lignes
                        for i, noise_level in enumerate(noise_levels):
                            axes[i, 0].set_ylabel(f'Bruit:\n{noise_level}', rotation=0, labelpad=40, ha='center')

                        plt.suptitle(
                            f"Comparaison finale: Effet du niveau de bruit sur la génération (chiffre {ref_digit})")
                        plt.tight_layout()
                        plt.show()

                        print("\n💡 OBSERVATIONS:")
                        print("   • Bruit 0.05: Variations subtiles, très similaires à l'original")
                        print("   • Bruit 0.1: Variations modérées, encore reconnaissables")
                        print("   • Bruit 0.2: Variations importantes, début de déformation")
                        print("   • Bruit 0.3: Variations extrêmes, possibles artefacts")

                    print("✅ Génération synthétique terminée!")

                else:
                    print("❌ Choix PCA invalide. Veuillez choisir 1, 2, 3 ou 4.")

        elif algo == 'autoencoder':
            print("🟡 You chose Autoencoder for unsupervised learning.")
            print("⚠️ AutoEncoder implementation not yet available.")

    print("🏁 Programme terminé.")