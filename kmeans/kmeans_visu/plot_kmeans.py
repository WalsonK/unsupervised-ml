# mnist_clustering.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
import os
import pandas as pd
from kmeans_cuda import KMeansCUDA

def run_mnist_clustering():
    """
    Run the complete MNIST clustering analysis and generate all plots.
    """
    # 0. Create image folder in the same directory as the script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    image_dir = os.path.join(script_dir, "image")
    os.makedirs(image_dir, exist_ok=True)
    
    # 1. Load MNIST
    mnist = fetch_openml('mnist_784', version=1)
    X = mnist.data.to_numpy()
    y = mnist.target.astype(int).to_numpy()
    
    # 2. Normalize data
    X = X / 255.0
    
    # 3. Run KMeans clustering with CUDA implementation
    kmeans = KMeansCUDA(n_clusters=50, max_iter=300, tol=1e-4, device='cuda')
    kmeans.fit(X)
    labels = kmeans.predict(X)
    
    # Get centroids for plotting (convert from torch tensor to numpy)
    cluster_centers = kmeans.centroids.cpu().numpy()
    
    # 4. Plot and save all 50 cluster centroids as images
    plt.figure(figsize=(12, 10))
    for i in range(50):
        plt.subplot(10, 5, i + 1)
        plt.imshow(cluster_centers[i].reshape(28, 28), cmap='gray')
        plt.title(f'Cluster {i}')
        plt.axis('off')
    plt.suptitle("KMeans Cluster Centroids", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space for suptitle
    plt.savefig(os.path.join(image_dir, "centroids.png"))
    plt.close()
    
    # 5. Create stacked bar chart: distribution of true labels in each cluster
    num_clusters = 50
    num_classes = 10
    
    # Build a contingency table: cluster x true class
    confusion_matrix = np.zeros((num_clusters, num_classes), dtype=int)
    for cluster_id in range(num_clusters):
        idxs = np.where(labels == cluster_id)[0]
        true_labels = y[idxs]
        for digit in range(num_classes):
            confusion_matrix[cluster_id, digit] = np.sum(true_labels == digit)
    
    # Plot stacked bar chart
    cluster_ids = np.arange(num_clusters)
    bottom = np.zeros(num_clusters)
    plt.figure(figsize=(12, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, num_classes))
    bar_width = 0.8  # Adjust this value to control spacing (0.8 means 20% spacing)
    
    for digit in range(num_classes):
        plt.bar(cluster_ids,
                confusion_matrix[:, digit],
                bottom=bottom,
                color=colors[digit],
                label=f'Digit {digit}',
                width=bar_width)  # Add width parameter for spacing
        bottom += confusion_matrix[:, digit]
    
    plt.xlabel('Cluster ID')
    plt.ylabel('Number of Samples')
    plt.title('True Label Distribution in Each Cluster')
    plt.legend(title='True Label', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(image_dir, "cluster_class_distribution.png"))
    plt.close()
    
    # 6. Save example digits per cluster (updated for 50 clusters)
    def plot_cluster_examples(X, labels, cluster_id, num_examples=10):
        idxs = np.where(labels == cluster_id)[0][:num_examples]
        if len(idxs) == 0:
            return  # skip empty clusters
        
        plt.figure(figsize=(10, 1))
        for i, idx in enumerate(idxs):
            plt.subplot(1, num_examples, i + 1)
            plt.imshow(X[idx].reshape(28, 28), cmap='gray')
            plt.axis('off')
        plt.suptitle(f'Examples from Cluster {cluster_id}')
        plt.tight_layout()
        plt.savefig(os.path.join(image_dir, f"cluster_{cluster_id}.png"))
        plt.close()
    
    for i in range(50):
        plot_cluster_examples(X, labels, cluster_id=i)

# Allow running directly
if __name__ == "__main__":
    run_mnist_clustering()