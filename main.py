import numpy as np
from sklearn.datasets import fetch_openml
from kmeans.kmeans import Kmeans
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
            print("What do you want to do with Kmeans? \n1.train\n2.load: ")
            choice = int(input("> ").strip())

            if choice == 1:
                cluster_center, labels = model.train(X, n_clusters=10, max_iter=100)
                print("Training completed.")
                is_saving = input("Do you want to save the model info? (y/n): ").strip().lower()
                if is_saving == 'y':
                    model.save_info("kmeans_info.json")

            elif choice == 2:
                model.load_info("kmeans/kmeans_info.json")
                mode = input("Do you want to use the model for inference or compression? ([I]nference/[C]ompression/[G]eneration): ").strip().lower()
                if mode == 'i':
                    point_index = int(input("Enter the index of the point to infer: "))
                    point = X[point_index]
                    label = model.inference(point)
                    print(f"The inferred label for the point at index {point_index} is: {label}")
                elif mode == 'c':
                    index, distance = model.kmeans_compression(X[0])
                    print((index, distance))
                    new_point = model.kmeans_decompression(index, distance)
                    model.display(np.array([X[0], new_point]), titre="Decompressed Image")
                elif mode == 'g':
                    index = int(input("Enter the cluster index to generate a point: "))
                    new_point = model.kmeans_generation(index)
                    model.display(new_point, titre="Generated Image")

        elif algo == 'pca':
            print("You chose PCA for dimensionality reduction.")
        elif algo == 'autoencoder':
            print("You chose Autoencoder for unsupervised learning.")
