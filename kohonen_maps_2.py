# full_som_analysis_with_weights.py

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from tqdm import tqdm
import math
import os

# ==============================================================================
# 1. Self-Organizing Map (SOM) Class with CUDA Optimization
# ==============================================================================

class SOMCuda:
    """
    A from-scratch implementation of a Self-Organizing Map (Kohonen Map)
    optimized to run on a CUDA-enabled GPU using PyTorch.
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
            print("WARNING: CUDA not available, falling back to CPU. This will be slow.")
            device = 'cpu'
        self.device = torch.device(device)
        print(f"Using device: {self.device}")
        
        self.weights = torch.rand(
            self.grid_height, self.grid_width, self.input_dim,
            device=self.device, dtype=torch.float32
        )
        
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
        if self.decay_function == 'exponential':
            time_constant = total_iterations / math.log(self.initial_sigma)
            self.learning_rate = self.initial_learning_rate * math.exp(-iteration / total_iterations)
            self.sigma = self.initial_sigma * math.exp(-iteration / time_constant)
        elif self.decay_function == 'linear':
            fraction_done = iteration / total_iterations
            self.learning_rate = self.initial_learning_rate * (1 - fraction_done)
            self.sigma = self.initial_sigma * (1 - fraction_done)
    
    def train(self, data, num_iterations):
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data).float()
        data = data.to(self.device)
        n_samples = data.shape[0]
        
        for iteration in tqdm(range(num_iterations), desc=f"Training ({num_iterations} iters)"):
            sample_idx = torch.randint(0, n_samples, (1,)).item()
            input_vector = data[sample_idx]
            self._decay_parameters(iteration, num_iterations)
            bmu_y, bmu_x = self._find_bmu(input_vector)
            self._update_weights(input_vector, bmu_y, bmu_x)
    
    def get_bmu_coordinates(self, data):
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data).float()
        data = data.to(self.device)
        bmu_coords = []
        for sample in data:
            bmu_y, bmu_x = self._find_bmu(sample)
            bmu_coords.append([bmu_y.cpu().item(), bmu_x.cpu().item()])
        return np.array(bmu_coords)

    def get_weights(self):
        """Returns the weights as a NumPy array on the CPU."""
        return self.weights.cpu().numpy()

# ==============================================================================
# 2. Visualization Functions
# ==============================================================================

def plot_projection(som, X, y, ax, iteration_count):
    """Plots the 'cloud point' projection of data onto the SOM grid."""
    print(f"  Plotting data projection for {iteration_count:,} iterations...")
    bmu_coords = som.get_bmu_coordinates(X)
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    for i in range(10):
        class_bmus = bmu_coords[y == i]
        if len(class_bmus) > 0:
            jitter_x = class_bmus[:, 1] + np.random.rand(len(class_bmus)) * 0.8 - 0.4
            jitter_y = class_bmus[:, 0] + np.random.rand(len(class_bmus)) * 0.8 - 0.4
            ax.scatter(jitter_x, jitter_y, c=[colors[i]], label=f'Digit {i}', s=5, alpha=0.7)

    ax.set_title(f'{iteration_count:,} Iterations')
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(-1, som.grid_width)
    ax.set_ylim(-1, som.grid_height)
    ax.invert_yaxis()

def plot_weight_grid(som, ax, iteration_count):
    """Plots the SOM's weights as a grid of images."""
    print(f"  Plotting weight grid for {iteration_count:,} iterations...")
    weights = som.get_weights()
    grid_h, grid_w, _ = weights.shape
    
    # Create a large canvas to tile the weight images
    canvas = np.ones((grid_h * 28, grid_w * 28))
    
    for i in range(grid_h):
        for j in range(grid_w):
            # Get the weight vector for the neuron and reshape it to a 28x28 image
            neuron_image = weights[i, j].reshape(28, 28)
            # Place it on the canvas
            canvas[i*28:(i+1)*28, j*28:(j+1)*28] = neuron_image
            
    ax.imshow(canvas, cmap='gray')
    ax.set_title(f'{iteration_count:,} Iterations')
    ax.axis('off')

# ==============================================================================
# 3. Main Analysis Driver
# ==============================================================================

def run_evolution_analysis(save_dir="som_evolution_combined"):
    """
    Trains a SOM in stages and, for each stage, plots both the data projection
    and the grid of learned neuron weights.
    """
    os.makedirs(save_dir, exist_ok=True)

    print("Loading MNIST dataset...")
    mnist = fetch_openml('mnist_784', version=1)
    X = mnist.data.to_numpy() / 255.0
    y = mnist.target.astype(int).to_numpy()
    
    n_samples = 10000
    indices = np.random.choice(len(X), n_samples, replace=False)
    X_subset = X[indices]
    y_subset = y[indices]

    grid_size = (20, 20)
    checkpoints = [100, 500, 1000, 5000, 10000, 25000, 50000, 100000]
    
    # --- Set up two figures: one for projections, one for weight grids ---
    fig_proj, axes_proj = plt.subplots(2, 4, figsize=(20, 10), constrained_layout=True)
    axes_proj = axes_proj.flatten()

    fig_weights, axes_weights = plt.subplots(2, 4, figsize=(16, 16), constrained_layout=True)
    axes_weights = axes_weights.flatten()

    for i, n_iter in enumerate(checkpoints):
        print(f"\n--- STAGE {i+1}/{len(checkpoints)}: Training a new SOM for {n_iter} iterations ---")
        
        som = SOMCuda(
            grid_size=grid_size,
            input_dim=784,
            learning_rate=0.6,
            sigma=max(grid_size) / 2,
            device='cuda'
        )
        
        som.train(X_subset, num_iterations=n_iter)
        
        # Generate both plots for the current stage
        plot_projection(som, X_subset, y_subset, axes_proj[i], n_iter)
        plot_weight_grid(som, axes_weights[i], n_iter)

    # --- Finalize and save the PROJECTION plot ---
    handles, labels = axes_proj[0].get_legend_handles_labels()
    fig_proj.legend(handles, labels, loc='center right', title="Digits")
    fig_proj.suptitle('Evolution of MNIST Data Projection on a Self-Organizing Map', fontsize=24)
    fig_proj.subplots_adjust(right=0.92)
    proj_save_path = os.path.join(save_dir, "som_projection_evolution.png")
    fig_proj.savefig(proj_save_path, dpi=150)
    print(f"\nProjection evolution plot saved to: {proj_save_path}")

    # --- Finalize and save the WEIGHTS plot ---
    fig_weights.suptitle("Evolution of SOM's Learned Weight Prototypes", fontsize=24)
    weights_save_path = os.path.join(save_dir, "som_weights_evolution.png")
    fig_weights.savefig(weights_save_path, dpi=150)
    print(f"Weights evolution plot saved to: {weights_save_path}")

    plt.show() # Show both figures

# ==============================================================================
# 4. Script Execution
# ==============================================================================

if __name__ == "__main__":
    run_evolution_analysis()