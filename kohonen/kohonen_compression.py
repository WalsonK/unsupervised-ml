# som_compression.py

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from tqdm import tqdm
import math
import os

# ==============================================================================
# 1. Self-Organizing Map (SOM) Class with Save/Load Functionality
# ==============================================================================

class SOMCuda:
    """
    SOM class with added save() and load() methods for the weights.
    """
    def __init__(self, grid_size=(20, 20), input_dim=784, learning_rate=0.5, 
                 sigma=None, decay_function='exponential', device='cuda'):
        self.grid_height, self.grid_width = grid_size
        self.input_dim = input_dim
        self.initial_learning_rate = learning_rate
        self.learning_rate = learning_rate
        
        if sigma is None:
            self.initial_sigma = max(self.grid_height, self.grid_width) / 2.0
        else:
            self.initial_sigma = float(sigma)
        self.sigma = self.initial_sigma
        
        self.decay_function = decay_function
        
        if device == 'cuda' and not torch.cuda.is_available():
            print("WARNING: CUDA not available, falling back to CPU.")
            device = 'cpu'
        self.device = torch.device(device)
        
        self.weights = torch.rand(
            self.grid_height, self.grid_width, self.input_dim,
            device=self.device, dtype=torch.float32
        )
        self._create_coordinate_grid()

    def save(self, filepath):
        """Saves the SOM's weights to a file."""
        torch.save(self.weights, filepath)
        print(f"SOM weights saved to {filepath}")

    def load(self, filepath):
        """Loads the SOM's weights from a file."""
        self.weights = torch.load(filepath, map_location=self.device)
        print(f"SOM weights loaded from {filepath}")
        self.grid_height, self.grid_width, self.input_dim = self.weights.shape
        self._create_coordinate_grid()
        
    def _create_coordinate_grid(self):
        y_coords, x_coords = torch.meshgrid(
            torch.arange(self.grid_height, device=self.device),
            torch.arange(self.grid_width, device=self.device),
            indexing='ij'
        )
        self.neuron_coords = torch.stack([y_coords, x_coords], dim=-1).float()
        
    def _find_bmu(self, input_vector):
        distances_sq = torch.sum((self.weights - input_vector) ** 2, dim=2)
        bmu_idx = torch.argmin(distances_sq)
        bmu_y = bmu_idx // self.grid_width
        bmu_x = bmu_idx % self.grid_width
        return bmu_y, bmu_x
    
    def _update_weights(self, input_vector, bmu_y, bmu_x):
        bmu_coord = torch.tensor([bmu_y, bmu_x], device=self.device, dtype=torch.float32)
        distances_sq_to_bmu = torch.sum((self.neuron_coords - bmu_coord) ** 2, dim=2)
        neighborhood_influence = torch.exp(-distances_sq_to_bmu / (2 * self.sigma ** 2))
        influence = neighborhood_influence.unsqueeze(2)
        weight_update = influence * self.learning_rate * (input_vector - self.weights)
        self.weights += weight_update
    
    def _decay_parameters(self, iteration, total_iterations):
        fraction_done = iteration / total_iterations
        self.learning_rate = self.initial_learning_rate * (1 - fraction_done)
        self.sigma = self.initial_sigma * (1 - fraction_done)
    
    def train(self, data, num_iterations):
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data).float().to(self.device)
        n_samples = data.shape[0]
        
        for iteration in tqdm(range(num_iterations), desc=f"Training ({num_iterations} iters)"):
            sample_idx = torch.randint(0, n_samples, (1,)).item()
            input_vector = data[sample_idx]
            self._decay_parameters(iteration, num_iterations)
            bmu_y, bmu_x = self._find_bmu(input_vector)
            self._update_weights(input_vector, bmu_y, bmu_x)
    
    def get_bmu_coordinates(self, data):
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data).float().to(self.device)
        bmu_coords = []
        for sample in tqdm(data, desc="Finding BMUs (Compressing)"):
            bmu_y, bmu_x = self._find_bmu(sample)
            bmu_coords.append([bmu_y.cpu().item(), bmu_x.cpu().item()])
        return np.array(bmu_coords, dtype=np.uint8)

    def get_weights(self):
        return self.weights.cpu().numpy()

# ==============================================================================
# 2. Main Compression & Decompression Pipeline
# ==============================================================================

def run_compression_pipeline():
    # --- Parameters ---
    grid_size = (20, 20)
    num_training_iterations = 50000
    model_filename = "som_codebook.pt"
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, model_filename)

    # --- Step 1: Train the Model (or load if it exists) ---
    print("--- Step 1: Training Phase ---")
    if not os.path.exists(model_path):
        print("No pre-trained model found. Training a new SOM...")
        mnist = fetch_openml('mnist_784', version=1)
        X_train = mnist.data.to_numpy()[:60000] / 255.0
        
        som_trainer = SOMCuda(grid_size=grid_size, input_dim=X_train.shape[1], device='cuda')
        som_trainer.train(X_train, num_iterations=num_training_iterations)
        som_trainer.save(model_path)
    else:
        print(f"Found pre-trained model at {model_path}")

    # --- Step 2: Load the Model and Data for Compression ---
    print("\n--- Step 2: Loading Phase ---")
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    X_test = mnist.data[60000:] / 255.0
    y_test = mnist.target[60000:].astype(int)

    som_compressor = SOMCuda(grid_size=grid_size, input_dim=X_test.shape[1], device='cuda')
    som_compressor.load(model_path)
    codebook = som_compressor.get_weights()

    # --- Step 3: Compress the Data ---
    print("\n--- Step 3: Compression Phase ---")
    compressed_indices = som_compressor.get_bmu_coordinates(X_test)

    # --- Step 4: Decompress the Data ---
    print("\n--- Step 4: Decompression Phase ---")
    decompressed_data = codebook[compressed_indices[:, 0], compressed_indices[:, 1]]

    # --- Step 5: Analysis and Visualization ---
    print("\n--- Step 5: Analysis ---")
    
    original_size = X_test.nbytes
    codebook_size = codebook.nbytes
    indices_size = compressed_indices.nbytes
    total_compressed_size = codebook_size + indices_size
    
    print(f"Original test data size:      {original_size / 1e6:.2f} MB")
    print(f"Compressed indices size:    {indices_size / 1e6:.2f} MB")
    print(f"Codebook (model) size:      {codebook_size / 1e6:.2f} MB")
    print(f"Total compressed size:      {total_compressed_size / 1e6:.2f} MB")
    print(f"Compression Ratio:          {original_size / total_compressed_size:.2f}x")
    
    mse = np.mean((X_test - decompressed_data)**2)
    print(f"Reconstruction MSE:         {mse:.6f}")
    
    # Visualize the results
    fig, axes = plt.subplots(2, 10, figsize=(15, 4))
    for i in range(10):
        axes[0, i].imshow(X_test[i].reshape(28, 28), cmap='gray')
        axes[0, i].set_title("Original")
        axes[0, i].axis('off')
        
        axes[1, i].imshow(decompressed_data[i].reshape(28, 28), cmap='gray')
        axes[1, i].set_title("Decompressed")
        axes[1, i].axis('off')
        
    fig.suptitle(f"SOM Compression | {grid_size[0]}x{grid_size[1]} Codebook | Reconstruction MSE: {mse:.4f}", fontsize=16)
    
    # Save the figure to a file
    plot_filename = "som_compression_comparison.png"
    plot_path = os.path.join(script_dir, plot_filename)
    fig.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\nComparison plot saved to: {plot_path}")
    
    plt.show()

if __name__ == "__main__":
    run_compression_pipeline()