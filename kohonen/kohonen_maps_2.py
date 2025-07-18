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
    Carte Auto-Organisatrice (SOM) optimisée pour GPU avec PyTorch
    """
    def __init__(self, grid_size=(20, 20), input_dim=784, learning_rate=0.5,
                 sigma=None, decay_function='exponential', device='cuda'):
        
        self.grid_height, self.grid_width = grid_size # Dimensions de la grille
        self.input_dim = input_dim # Dimension des vecteurs d'entrée
        self.initial_learning_rate = learning_rate # Taux d'apprentissage initial
        self.learning_rate = learning_rate # Taux d'apprentissage courant

        # Rayon de voisinage initial
        if sigma is None: 
            self.initial_sigma = max(self.grid_height, self.grid_width) / 2.0
        else: 
            self.initial_sigma = float(sigma)
        self.sigma = self.initial_sigma # Rayon courant

        self.decay_function = decay_function # Type de décroissance

        # Vérif GPU/CPU
        if device == 'cuda' and not torch.cuda.is_available():
            print("WARNING: CUDA not available, falling back to CPU. This will be slow.")
            device = 'cpu'
        self.device = torch.device(device)
        print(f"Using device: {self.device}")

        # Poids aléatoires pour chaque neurone de la grille
        self.weights = torch.rand(self.grid_height, self.grid_width, self.input_dim, 
                                device=self.device, dtype=torch.float32)
        
        self._create_coordinate_grid()

    def _create_coordinate_grid(self):
        # Grille des coordonnées (y,x) de chaque neurone
        y_coords, x_coords = torch.meshgrid(
            torch.arange(self.grid_height, device=self.device), 
            torch.arange(self.grid_width, device=self.device), 
            indexing='ij'
        )
        # Stack pour avoir (grid_h, grid_w, 2) avec [y, x] pour chaque position
        self.neuron_coords = torch.stack([y_coords, x_coords], dim=-1).float()

    def _find_bmu(self, input_vector):
        # Distance euclidienne² entre input et tous les poids
        distances_sq = torch.sum((self.weights - input_vector) ** 2, dim=2)
        # Index du neurone le plus proche
        bmu_idx = torch.argmin(distances_sq)
        # Conversion index linéaire -> coordonnées (y,x)
        return bmu_idx // self.grid_width, bmu_idx % self.grid_width

    def _update_weights(self, input_vector, bmu_y, bmu_x):
        bmu_coord = torch.tensor([bmu_y, bmu_x], device=self.device, dtype=torch.float32)
        # Distance² sur la grille entre chaque neurone et le BMU
        distances_sq_to_bmu = torch.sum((self.neuron_coords - bmu_coord) ** 2, dim=2)
        
        # Fonction gaussienne pour l'influence de voisinage
        neighborhood_influence = torch.exp(-distances_sq_to_bmu / (2 * self.sigma ** 2))
        influence = neighborhood_influence.unsqueeze(2) # Ajoute dim pour broadcasting
        
        # Mise à jour: poids += influence * lr * (input - poids)
        self.weights += influence * self.learning_rate * (input_vector - self.weights)

    def _decay_parameters(self, iteration, total_iterations):
        # Décroissance du learning rate et sigma au fil du temps
        if self.decay_function == 'exponential':
            time_constant = total_iterations / math.log(self.initial_sigma)
            self.learning_rate = self.initial_learning_rate * math.exp(-iteration / total_iterations)
            self.sigma = self.initial_sigma * math.exp(-iteration / time_constant)
        elif self.decay_function == 'linear':
            fraction_done = iteration / total_iterations
            self.learning_rate = self.initial_learning_rate * (1 - fraction_done)
            self.sigma = self.initial_sigma * (1 - fraction_done)

    def train(self, data, num_iterations):
        # Conversion numpy -> tensor si nécessaire
        if isinstance(data, np.ndarray): 
            data = torch.from_numpy(data).float()
        data = data.to(self.device)
        
        n_samples = data.shape[0]
        
        # Boucle d'entraînement
        for iteration in tqdm(range(num_iterations), desc=f"Training ({num_iterations} iters)"):
            # Échantillon aléatoire
            input_vector = data[torch.randint(0, n_samples, (1,)).item()]
            # Décroissance des paramètres
            self._decay_parameters(iteration, num_iterations)
            # Trouve BMU et met à jour
            bmu_y, bmu_x = self._find_bmu(input_vector)
            self._update_weights(input_vector, bmu_y, bmu_x)

    def get_weights(self):
        # Retourne les poids sur CPU en numpy
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
    
    canvas = np.ones((grid_h * 28, grid_w * 28))
    
    for i in range(grid_h):
        for j in range(grid_w):
            neuron_image = weights[i, j].reshape(28, 28)
            canvas[i*28:(i+1)*28, j*28:(j+1)*28] = neuron_image
            
    ax.imshow(canvas, cmap='gray')
    ax.set_title(f'{iteration_count:,} Iterations')
    ax.axis('off')

# ==============================================================================
# 3. Main Analysis Driver
# ==============================================================================

def run_evolution_analysis(output_folder_name="som_evolution_combined"):
    """
    Trains a SOM in stages and, for each stage, plots both the data projection
    and the grid of learned neuron weights.
    """
    # Get the directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Create the full path for the output directory inside the script's directory
    save_dir = os.path.join(script_dir, output_folder_name)
    
    # Now, create this directory
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
        
        plot_projection(som, X_subset, y_subset, axes_proj[i], n_iter)
        plot_weight_grid(som, axes_weights[i], n_iter)

    handles, labels = axes_proj[0].get_legend_handles_labels()
    fig_proj.legend(handles, labels, loc='center right', title="Digits")
    fig_proj.suptitle('Evolution of MNIST Data Projection on a Self-Organizing Map', fontsize=24)
    fig_proj.subplots_adjust(right=0.92)
    proj_save_path = os.path.join(save_dir, "som_projection_evolution.png")
    fig_proj.savefig(proj_save_path, dpi=150)
    print(f"\nProjection evolution plot saved to: {proj_save_path}")

    fig_weights.suptitle("Evolution of SOM's Learned Weight Prototypes", fontsize=24)
    weights_save_path = os.path.join(save_dir, "som_weights_evolution.png")
    fig_weights.savefig(weights_save_path, dpi=150)
    print(f"Weights evolution plot saved to: {weights_save_path}")

    plt.show()

# ==============================================================================
# 4. Script Execution
# ==============================================================================

if __name__ == "__main__":
    run_evolution_analysis()