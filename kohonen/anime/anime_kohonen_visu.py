# anime_som_analysis.py
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import math
import os
from PIL import Image
import zipfile
import io

# ==============================================================================
# 0. Configuration
# ==============================================================================
IMG_SIZE = 64  # Resize images to 64x64
CHANNELS = 3   # Images are in color (RGB)
INPUT_DIM = IMG_SIZE * IMG_SIZE * CHANNELS
MAX_SAMPLES = 5000 # Use a subset of images for faster training. Set to None for all.
GRID_SIZE = (20, 20) # The dimensions of the SOM grid

# ==============================================================================
# 1. Self-Organizing Map (SOM) Class (No changes needed, it's generic)
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
        if sigma is None: self.initial_sigma = max(self.grid_height, self.grid_width) / 2.0
        else: self.initial_sigma = float(sigma)
        self.sigma = self.initial_sigma
        self.decay_function = decay_function
        if device == 'cuda' and not torch.cuda.is_available():
            print("WARNING: CUDA not available, falling back to CPU. This will be slow.")
            device = 'cpu'
        self.device = torch.device(device)
        print(f"Using device: {self.device}")
        self.weights = torch.rand(self.grid_height, self.grid_width, self.input_dim, device=self.device, dtype=torch.float32)
        self._create_coordinate_grid()

    def _create_coordinate_grid(self):
        y_coords, x_coords = torch.meshgrid(torch.arange(self.grid_height, device=self.device), torch.arange(self.grid_width, device=self.device), indexing='ij')
        self.neuron_coords = torch.stack([y_coords, x_coords], dim=-1).float()

    def _find_bmu(self, input_vector):
        distances_sq = torch.sum((self.weights - input_vector) ** 2, dim=2)
        bmu_idx = torch.argmin(distances_sq)
        return bmu_idx // self.grid_width, bmu_idx % self.grid_width

    def _update_weights(self, input_vector, bmu_y, bmu_x):
        bmu_coord = torch.tensor([bmu_y, bmu_x], device=self.device, dtype=torch.float32)
        distances_sq_to_bmu = torch.sum((self.neuron_coords - bmu_coord) ** 2, dim=2)
        neighborhood_influence = torch.exp(-distances_sq_to_bmu / (2 * self.sigma ** 2))
        influence = neighborhood_influence.unsqueeze(2)
        self.weights += influence * self.learning_rate * (input_vector - self.weights)

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
        if isinstance(data, np.ndarray): data = torch.from_numpy(data).float()
        data = data.to(self.device)
        n_samples = data.shape[0]
        for iteration in tqdm(range(num_iterations), desc=f"Training ({num_iterations} iters)"):
            input_vector = data[torch.randint(0, n_samples, (1,)).item()]
            self._decay_parameters(iteration, num_iterations)
            bmu_y, bmu_x = self._find_bmu(input_vector)
            self._update_weights(input_vector, bmu_y, bmu_x)

    def get_weights(self):
        return self.weights.cpu().numpy()

# ==============================================================================
# 2. Data Loading and Visualization Functions (Modified for Anime Faces)
# ==============================================================================
def load_anime_faces_from_zip(zip_filepath, target_size=(IMG_SIZE, IMG_SIZE), max_samples=None):
    """Loads and preprocesses anime face images directly from a zip archive."""
    print(f"Loading images from '{zip_filepath}'...")
    if not os.path.exists(zip_filepath): raise FileNotFoundError(f"Zip file not found: {zip_filepath}")

    all_images = []
    with zipfile.ZipFile(zip_filepath, 'r') as zip_ref:
        image_files = [f for f in zip_ref.namelist() if f.startswith('images/') and f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if max_samples:
            print(f"Loading a random subset of {max_samples} images.")
            np.random.shuffle(image_files)
            image_files = image_files[:max_samples]

        for filename in tqdm(image_files, desc="Processing Images"):
            with zip_ref.open(filename) as file:
                try:
                    with Image.open(io.BytesIO(file.read())) as img:
                        img = img.convert('RGB').resize(target_size, Image.Resampling.LANCZOS)
                        all_images.append(np.array(img))
                except Exception as e:
                    print(f"Warning: Could not process {filename}. Error: {e}")

    if not all_images: raise ValueError("No images loaded. Check zip file.")
    print(f"Successfully loaded {len(all_images)} images.")
    return np.array(all_images)

def plot_weight_grid(som, ax, iteration_count, img_shape=(IMG_SIZE, IMG_SIZE, CHANNELS)):
    """Plots the SOM's weights as a grid of prototype anime faces."""
    print(f"  Plotting weight grid for {iteration_count:,} iterations...")
    weights = som.get_weights()
    grid_h, grid_w, _ = weights.shape
    img_h, img_w, channels = img_shape
    
    # Create a canvas to hold the grid of images
    canvas = np.ones((grid_h * img_h, grid_w * img_w, channels))
    
    for i in range(grid_h):
        for j in range(grid_w):
            # Reshape the neuron's weight vector back into a color image
            neuron_image = weights[i, j].reshape(img_h, img_w, channels)
            # Place the image on the canvas
            canvas[i*img_h:(i+1)*img_h, j*img_w:(j+1)*img_w, :] = neuron_image
            
    # Clip values to be in the valid [0, 1] range for displaying
    canvas = np.clip(canvas, 0, 1)
    ax.imshow(canvas)
    ax.set_title(f'{iteration_count:,} Iterations')
    ax.axis('off')

# The plot_projection function has been removed as it requires true labels,
# which this dataset does not have.

# ==============================================================================
# 3. Main Analysis Driver (Modified for Anime Faces)
# ==============================================================================
def run_anime_som_analysis(output_folder_name="anime_som_evolution"):
    """
    Trains a SOM on the Anime Face dataset in stages and plots the evolution
    of the learned neuron weight grid.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(script_dir, output_folder_name)
    zip_path = os.path.join(script_dir, "archive.zip")
    os.makedirs(save_dir, exist_ok=True)

    # Load, normalize, and flatten the dataset
    X_orig = load_anime_faces_from_zip(zip_path, max_samples=MAX_SAMPLES)
    X_normalized = X_orig / 255.0
    X_flat = X_normalized.reshape(X_normalized.shape[0], -1)
    
    # Define the training checkpoints
    checkpoints = [100, 1000, 5000, 10000, 25000, 50000]
    
    # Set up the plot for the weight grid evolution
    fig_weights, axes_weights = plt.subplots(2, 3, figsize=(15, 10), constrained_layout=True)
    axes_weights = axes_weights.flatten()
    
    for i, n_iter in enumerate(checkpoints):
        print(f"\n--- STAGE {i+1}/{len(checkpoints)}: Training a new SOM for {n_iter} iterations ---")
        
        som = SOMCuda(
            grid_size=GRID_SIZE,
            input_dim=INPUT_DIM,
            learning_rate=0.6,
            sigma=max(GRID_SIZE) / 2,
            device='cuda'
        )
        
        som.train(X_flat, num_iterations=n_iter)
        
        # Plot the current state of the weight grid
        plot_weight_grid(som, axes_weights[i], n_iter)

    fig_weights.suptitle("Evolution of SOM's Learned Anime Face Prototypes", fontsize=24)
    weights_save_path = os.path.join(save_dir, "som_anime_weights_evolution.png")
    fig_weights.savefig(weights_save_path, dpi=150)
    print(f"\nWeights evolution plot saved to: {weights_save_path}")
    
    plt.show()

# ==============================================================================
# 4. Script Execution
# ==============================================================================
if __name__ == "__main__":
    run_anime_som_analysis()