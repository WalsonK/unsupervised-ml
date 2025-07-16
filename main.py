# main.py

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
import torch

from kmeans.kmeans_visu.kmeans_cuda import KMeansCUDA
from kmeans.kmeans_visu.plot_kmeans import run_mnist_clustering


# ================================
# Main control logic
# ================================
if __name__ == '__main__':
    print("Loading MNIST dataset …")
    mnist = fetch_openml('mnist_784', version=1)
    X, y = mnist.data.to_numpy(), mnist.target.astype(int).to_numpy()

    print("MNIST dataset loaded successfully.")
    
    # Algorithm selection
    algo = input("Which algorithm would you like to use? (kmeans/pca/autoencoder): ").strip().lower()
    
    if algo not in ['kmeans', 'pca', 'autoencoder']:
        print("Invalid algorithm choice. Please choose from kmeans, pca, or autoencoder.")
    else:
        # Task selection
        task = input("What task would you like to perform? (train/visualization/compression/generation): ").strip().lower()
        
        if task not in ['train', 'visualization', 'compression', 'generation']:
            print("Invalid task choice. Please choose from train, visualization, compression, or generation.")
        else:
            print(f"\nRunning {algo} for {task}...")
            
            if algo == 'kmeans':
                if task == 'train':
                    print("Training KMeans model...")
                    # Add your kmeans training code here
                elif task == 'visualization':
                    print("Running KMeans visualization...")
                    run_mnist_clustering()
                elif task == 'compression':
                    print("Running KMeans compression...")
                    # Add your kmeans compression code here
                elif task == 'generation':
                    print("Running KMeans generation...")
                    # Add your kmeans generation code here
                    
            elif algo == 'pca':
                if task == 'train':
                    print("Training PCA model...")
                    # Add your PCA training code here
                elif task == 'visualization':
                    print("Running PCA visualization...")
                    # Add your PCA visualization code here
                elif task == 'compression':
                    print("Running PCA compression...")
                    # Add your PCA compression code here
                elif task == 'generation':
                    print("Running PCA generation...")
                    # Add your PCA generation code here
                    
            elif algo == 'autoencoder':
                if task == 'train':
                    print("Training Autoencoder model...")
                    # Add your autoencoder training code here
                elif task == 'visualization':
                    print("Running Autoencoder visualization...")
                    # Add your autoencoder visualization code here
                elif task == 'compression':
                    print("Running Autoencoder compression...")
                    # Add your autoencoder compression code here
                elif task == 'generation':
                    print("Running Autoencoder generation...")
                    # Add your autoencoder generation code here
            
            print(f"\n{algo.upper()} {task} completed!")