import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.cluster import KMeans
import os

# 0. Create image folder
os.makedirs("image", exist_ok=True)

# 1. Load MNIST
mnist = fetch_openml('mnist_784', version=1)
X = mnist.data.to_numpy()
y = mnist.target.astype(int).to_numpy()

# 2. Normalize data
X = X / 255.0

# 3. Run KMeans clustering
kmeans = KMeans(n_clusters=300, random_state=0)
labels = kmeans.fit_predict(X)

# 4. Plot and save all 20 cluster centroids as images
plt.figure(figsize=(12, 10))
for i in range(300):
    plt.subplot(60, 5, i + 1)
    plt.imshow(kmeans.cluster_centers_[i].reshape(28, 28), cmap='gray')
    plt.title(f'Cluster {i}')
    plt.axis('off')
plt.suptitle("KMeans Cluster Centroids", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space for suptitle
plt.savefig("image/centroids.png")
plt.close()

import pandas as pd

# 5. Create stacked bar chart: distribution of true labels in each cluster
num_clusters = 300
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

for digit in range(num_classes):
    plt.bar(cluster_ids,
            confusion_matrix[:, digit],
            bottom=bottom,
            color=colors[digit],
            label=f'Digit {digit}')
    bottom += confusion_matrix[:, digit]

plt.xlabel('Cluster ID')
plt.ylabel('Number of Samples')
plt.title('True Label Distribution in Each Cluster')
plt.legend(title='True Label', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig("image/cluster_class_distribution.png")
plt.close()

# 6. Save example digits per cluster (updated for 300 clusters)
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
    plt.savefig(f"image/cluster_{cluster_id}.png")
    plt.close()

for i in range(300):
    plot_cluster_examples(X, labels, cluster_id=i)