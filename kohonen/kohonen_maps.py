# som_cuda.py
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import math

class SOMCuda:
    def __init__(self, grid_size=(20, 20), input_dim=784, learning_rate=0.5, 
                 sigma=None, decay_function='exponential', device='cuda'):
        """
        Self-Organizing Map with CUDA optimization
        
        Parameters:
        -----------
        grid_size : tuple
            Size of the 2D grid (height, width)
        input_dim : int
            Dimension of input vectors
        learning_rate : float
            Initial learning rate
        sigma : float
            Initial neighborhood radius (default: max(grid_size)/2)
        decay_function : str
            Type of decay ('exponential' or 'linear')
        device : str
            Device to use ('cuda' or 'cpu')
        """
        self.grid_height, self.grid_width = grid_size
        self.input_dim = input_dim
        self.initial_learning_rate = learning_rate
        self.learning_rate = learning_rate
        
        # Set sigma to half the map radius if not specified
        if sigma is None:
            self.initial_sigma = max(self.grid_height, self.grid_width) / 2
        else:
            self.initial_sigma = sigma
        self.sigma = self.initial_sigma
        
        self.decay_function = decay_function
        
        # Device setup with fallback
        if device == 'cuda' and not torch.cuda.is_available():
            print("CUDA not available, falling back to CPU")
            device = 'cpu'
        self.device = torch.device(device)
        print(f"Using device: {self.device}")
        
        # Initialize weights randomly
        self.weights = torch.randn(
            self.grid_height, self.grid_width, self.input_dim,
            device=self.device, dtype=torch.float32
        ) * 0.1
        
        # Create coordinate grid for distance calculations
        self._create_coordinate_grid()
        
    def _create_coordinate_grid(self):
        """Create grid of neuron coordinates for distance calculations"""
        y_coords, x_coords = torch.meshgrid(
            torch.arange(self.grid_height, device=self.device),
            torch.arange(self.grid_width, device=self.device),
            indexing='ij'
        )
        self.neuron_coords = torch.stack([y_coords, x_coords], dim=-1).float()
        
    def _find_bmu(self, input_vector):
        """Find Best Matching Unit for input vector"""
        # Calculate distances between input and all neurons
        distances = torch.sum((self.weights - input_vector) ** 2, dim=2)
        
        # Find BMU coordinates
        bmu_idx = torch.argmin(distances)
        bmu_y = bmu_idx // self.grid_width
        bmu_x = bmu_idx % self.grid_width
        
        return bmu_y, bmu_x
    
    def _calculate_neighborhood(self, bmu_y, bmu_x, sigma):
        """Calculate neighborhood function (Gaussian)"""
        bmu_coord = torch.tensor([bmu_y, bmu_x], device=self.device, dtype=torch.float32)
        
        # Calculate distances from BMU to all neurons
        distances_sq = torch.sum((self.neuron_coords - bmu_coord) ** 2, dim=2)
        
        # Gaussian neighborhood function
        neighborhood = torch.exp(-distances_sq / (2 * sigma ** 2))
        
        return neighborhood
    
    def _update_weights(self, input_vector, bmu_y, bmu_x, learning_rate, sigma):
        """Update weights using SOM learning rule"""
        # Calculate neighborhood influence
        neighborhood = self._calculate_neighborhood(bmu_y, bmu_x, sigma)
        
        # Expand dimensions for broadcasting
        neighborhood = neighborhood.unsqueeze(2)  # Shape: (height, width, 1)
        
        # Calculate weight updates
        weight_diff = input_vector - self.weights
        weight_update = learning_rate * neighborhood * weight_diff
        
        # Update weights
        self.weights += weight_update
    
    def _decay_parameters(self, iteration, total_iterations):
        """Decay learning rate and sigma over time"""
        if self.decay_function == 'exponential':
            time_constant = total_iterations / math.log(self.initial_sigma)
            self.learning_rate = self.initial_learning_rate * math.exp(-iteration / time_constant)
            self.sigma = self.initial_sigma * math.exp(-iteration / time_constant)
        elif self.decay_function == 'linear':
            self.learning_rate = self.initial_learning_rate * (1 - iteration / total_iterations)
            self.sigma = self.initial_sigma * (1 - iteration / total_iterations)
    
    def train(self, data, num_iterations=1000, batch_size=None):
        """
        Train the SOM
        
        Parameters:
        -----------
        data : numpy.ndarray or torch.Tensor
            Training data of shape (n_samples, input_dim)
        num_iterations : int
            Number of training iterations
        batch_size : int
            If None, use online learning (one sample at a time)
        """
        # Convert data to torch tensor if needed
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data).float()
        data = data.to(self.device)
        
        print(f"Training SOM for {num_iterations} iterations...")
        
        for iteration in tqdm(range(num_iterations)):
            # Decay parameters
            self._decay_parameters(iteration, num_iterations)
            
            if batch_size is None:
                # Online learning - one sample at a time
                sample_idx = torch.randint(0, data.shape[0], (1,))
                input_vector = data[sample_idx].squeeze()
            else:
                # Mini-batch learning
                batch_indices = torch.randint(0, data.shape[0], (batch_size,))
                batch_data = data[batch_indices]
                
                for input_vector in batch_data:
                    bmu_y, bmu_x = self._find_bmu(input_vector)
                    self._update_weights(input_vector, bmu_y, bmu_x, 
                                       self.learning_rate, self.sigma)
                continue
            
            # Find BMU and update weights
            bmu_y, bmu_x = self._find_bmu(input_vector)
            self._update_weights(input_vector, bmu_y, bmu_x, 
                               self.learning_rate, self.sigma)
    
    def get_bmu_coordinates(self, data):
        """Get BMU coordinates for all data points"""
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data).float()
        data = data.to(self.device)
        
        bmu_coords = []
        for sample in data:
            bmu_y, bmu_x = self._find_bmu(sample)
            bmu_coords.append([bmu_y.cpu().item(), bmu_x.cpu().item()])
        
        return np.array(bmu_coords)
    
    def calculate_u_matrix(self):
        """Calculate U-matrix (unified distance matrix) for visualization"""
        u_matrix = torch.zeros(self.grid_height, self.grid_width, device=self.device)
        
        for i in range(self.grid_height):
            for j in range(self.grid_width):
                neighbors = []
                
                # Get all valid neighbors (8-connectivity)
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        if di == 0 and dj == 0:
                            continue
                        ni, nj = i + di, j + dj
                        if 0 <= ni < self.grid_height and 0 <= nj < self.grid_width:
                            neighbors.append(self.weights[ni, nj])
                
                if neighbors:
                    neighbors = torch.stack(neighbors)
                    distances = torch.norm(self.weights[i, j] - neighbors, dim=1)
                    u_matrix[i, j] = torch.mean(distances)
        
        return u_matrix.cpu().numpy()
    
    def get_weights(self):
        """Get weight vectors as numpy array"""
        return self.weights.cpu().numpy()


# som_plots.py
import numpy as np
import matplotlib.pyplot as plt
import os

def plot_som_results(som, X_train, y_train, save_dir="som_results"):
    """
    Create comprehensive SOM visualization plots using scatter points for grid visualizations.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Get BMU coordinates for training data
    print("Calculating BMU coordinates...")
    bmu_coords = som.get_bmu_coordinates(X_train)
    
    # --- Create coordinate grid for scatter plots ---
    grid_height, grid_width = som.grid_height, som.grid_width
    # Note: meshgrid for scatter plot (x, y) vs imshow (row, col)
    x_coords, y_coords = np.meshgrid(np.arange(grid_width), np.arange(grid_height))
    x_flat = x_coords.flatten()
    y_flat = y_coords.flatten()
    
    # 1. Plot data distribution on SOM grid (This plot already uses scatter, no change needed)
    plt.figure(figsize=(12, 10))
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    plt.subplot(2, 2, 1)
    for digit in range(10):
        mask = y_train == digit
        if np.any(mask):
            coords = bmu_coords[mask]
            # Add small jitter for better visualization
            jitter_x = coords[:, 1] + np.random.rand(len(coords)) * 0.5 - 0.25
            jitter_y = coords[:, 0] + np.random.rand(len(coords)) * 0.5 - 0.25
            plt.scatter(jitter_x, jitter_y, 
                       c=[colors[digit]], label=f'Digit {digit}', 
                       alpha=0.6, s=20)
    
    plt.title('Data Distribution on SOM Grid')
    plt.xlabel('SOM X coordinate')
    plt.ylabel('SOM Y coordinate')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.gca().invert_yaxis() # Match orientation with other plots

    # 2. U-Matrix visualization using scatter points
    plt.subplot(2, 2, 2)
    u_matrix = som.calculate_u_matrix()
    u_flat = u_matrix.flatten()
    scatter = plt.scatter(x_flat, y_flat, c=u_flat, cmap='viridis', marker='s', s=150)
    plt.title('U-Matrix (Distance Map)')
    plt.colorbar(scatter, fraction=0.046, pad=0.04)
    plt.xlim(-1, grid_width)
    plt.ylim(-1, grid_height)
    plt.gca().invert_yaxis()

    # 3. Hit map (frequency of BMU activation) using scatter points
    plt.subplot(2, 2, 3)
    hit_map = np.zeros((grid_height, grid_width))
    for coord in bmu_coords:
        hit_map[int(coord[0]), int(coord[1])] += 1
    hit_flat = hit_map.flatten()
    # Use size to represent hit count for better visual impact
    size = 10 + (hit_flat / hit_flat.max()) * 250 if hit_flat.max() > 0 else 10
    scatter = plt.scatter(x_flat, y_flat, s=size, c=hit_flat, cmap='Reds', marker='o')
    plt.title('Hit Map (Activation Frequency)')
    plt.colorbar(scatter, fraction=0.046, pad=0.04)
    plt.xlim(-1, grid_width)
    plt.ylim(-1, grid_height)
    plt.gca().invert_yaxis()

    # 4. Weight vector visualization (This remains the same)
    plt.subplot(2, 2, 4)
    weights = som.get_weights()
    prototype_grid = np.zeros((4*28, 4*28))
    for i in range(4):
        for j in range(4):
            y_idx = i * grid_height // 4
            x_idx = j * grid_width // 4
            prototype = weights[y_idx, x_idx].reshape(28, 28)
            start_y, end_y = i*28, (i+1)*28
            start_x, end_x = j*28, (j+1)*28
            prototype_grid[start_y:end_y, start_x:end_x] = prototype
    plt.imshow(prototype_grid, cmap='gray')
    plt.title('Weight Prototypes (4x4 sample)')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/som_overview.png", dpi=150, bbox_inches='tight')
    plt.show()
    
    # 5. Detailed prototype visualization (This remains the same)
    plt.figure(figsize=(15, 12))
    weights = som.get_weights()
    for i in range(min(grid_height, 8)):
        for j in range(min(grid_width, 10)):
            if i < 8 and j < 10:
                plt.subplot(8, 10, i*10 + j + 1)
                prototype = weights[i*grid_height//8, j*grid_width//10]
                plt.imshow(prototype.reshape(28, 28), cmap='gray')
                plt.axis('off')
                plt.title(f'({i},{j})', fontsize=8)
    plt.suptitle('SOM Weight Prototypes', fontsize=16)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/som_prototypes.png", dpi=150, bbox_inches='tight')
    plt.show()
    
    # 6. Class distribution per neuron using scatter points
    plt.figure(figsize=(12, 8))
    class_map = np.zeros((grid_height, grid_width, 10))
    for coord, label in zip(bmu_coords, y_train):
        class_map[int(coord[0]), int(coord[1]), int(label)] += 1
    
    # Dominant class for each neuron
    dominant_class = np.argmax(class_map, axis=2)
    purity = np.max(class_map, axis=2) / (np.sum(class_map, axis=2) + 1e-8)
    
    # Plot dominant class using scatter
    plt.subplot(1, 2, 1)
    dominant_class_flat = dominant_class.flatten()
    scatter = plt.scatter(x_flat, y_flat, c=dominant_class_flat, cmap='tab10', vmin=-0.5, vmax=9.5, marker='s', s=100)
    cbar = plt.colorbar(scatter, ticks=np.arange(10), fraction=0.046, pad=0.04)
    plt.title('Dominant Class per Neuron')
    plt.xlim(-1, grid_width)
    plt.ylim(-1, grid_height)
    plt.gca().invert_yaxis()
    
    # Plot purity using scatter
    plt.subplot(1, 2, 2)
    purity_flat = purity.flatten()
    scatter = plt.scatter(x_flat, y_flat, c=purity_flat, cmap='viridis', vmin=0, vmax=1, marker='s', s=100)
    plt.colorbar(scatter, fraction=0.046, pad=0.04)
    plt.title('Class Purity per Neuron')
    plt.xlim(-1, grid_width)
    plt.ylim(-1, grid_height)
    plt.gca().invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/som_class_analysis.png", dpi=150, bbox_inches='tight')
    plt.show()


# Example usage
def run_som_analysis():
    """Run complete SOM analysis on MNIST"""
    from sklearn.datasets import fetch_openml
    
    print("Loading MNIST dataset...")
    mnist = fetch_openml('mnist_784', version=1)
    X = mnist.data.to_numpy() / 255.0  # Normalize
    y = mnist.target.astype(int).to_numpy()
    
    # Use subset for faster training (optional)
    n_samples = 10000
    indices = np.random.choice(len(X), n_samples, replace=False)
    X_subset = X[indices]
    y_subset = y[indices]
    
    # Create and train SOM
    som = SOMCuda(
        grid_size=(20, 20),
        input_dim=784,
        learning_rate=0.5,
        sigma=10.0,
        device='cuda'
    )
    
    som.train(X_subset, num_iterations=1000)
    
    # Generate plots
    plot_som_results(som, X_subset, y_subset)
    
    return som, X_subset, y_subset


if __name__ == "__main__":
    som, X, y = run_som_analysis()