# anime_som_compression.py
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
IMG_SIZE = 64
CHANNELS = 3
INPUT_DIM = IMG_SIZE * IMG_SIZE * CHANNELS
GRID_SIZE = (20, 20)  # A 20x20 codebook gives 400 prototype faces
NUM_TRAINING_ITERATIONS = 100000
MAX_IMAGES_TO_LOAD = 6000 # Use a subset of the dataset to manage memory
TRAIN_SET_SIZE = 5000     # Number of images to train the SOM on
TEST_SET_SIZE = 100       # Number of images to test compression on

# ==============================================================================
# 1. Self-Organizing Map (SOM) Class with Save/Load Functionality
# ==============================================================================
class SOMCuda:
    """
    SOM class with added save() and load() methods for the weights (codebook).
    """
    def __init__(self, grid_size=(20, 20), input_dim=784, learning_rate=0.5,
                 sigma=None, decay_function='linear', device='cuda'):
        self.grid_height, self.grid_width = grid_size
        self.input_dim = input_dim
        self.initial_learning_rate = learning_rate
        self.learning_rate = learning_rate
        if sigma is None: self.initial_sigma = max(self.grid_height, self.grid_width) / 2.0
        else: self.initial_sigma = float(sigma)
        self.sigma = self.initial_sigma
        self.decay_function = decay_function
        if device == 'cuda' and not torch.cuda.is_available():
            print("WARNING: CUDA not available, falling back to CPU.")
            device = 'cpu'
        self.device = torch.device(device)
        self.weights = torch.rand(self.grid_height, self.grid_width, self.input_dim, device=self.device, dtype=torch.float32)
        self._create_coordinate_grid()

    def save(self, filepath):
        torch.save(self.weights, filepath)
        print(f"SOM codebook saved to {filepath}")

    def load(self, filepath):
        self.weights = torch.load(filepath, map_location=self.device)
        print(f"SOM codebook loaded from {filepath}")
        self.grid_height, self.grid_width, self.input_dim = self.weights.shape
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
        fraction_done = iteration / total_iterations
        self.learning_rate = self.initial_learning_rate * (1 - fraction_done)
        self.sigma = self.initial_sigma * (1 - fraction_done)

    def train(self, data, num_iterations):
        if isinstance(data, np.ndarray): data = torch.from_numpy(data).float().to(self.device)
        n_samples = data.shape[0]
        for iteration in tqdm(range(num_iterations), desc=f"Training ({num_iterations} iters)"):
            input_vector = data[torch.randint(0, n_samples, (1,)).item()]
            self._decay_parameters(iteration, num_iterations)
            bmu_y, bmu_x = self._find_bmu(input_vector)
            self._update_weights(input_vector, bmu_y, bmu_x)

    def get_bmu_coordinates(self, data):
        if isinstance(data, np.ndarray): data = torch.from_numpy(data).float().to(self.device)
        bmu_coords = []
        for sample in tqdm(data, desc="Finding BMUs (Compressing)"):
            bmu_y, bmu_x = self._find_bmu(sample)
            bmu_coords.append([bmu_y.cpu().item(), bmu_x.cpu().item()])
        # Use uint8 for indices if grid size is <= 256 (16x16)
        return np.array(bmu_coords, dtype=np.uint16)

    def get_weights(self):
        return self.weights.cpu().numpy()

# ==============================================================================
# 2. Data Loading Function (Adapted for Anime Faces)
# ==============================================================================
def load_anime_faces_from_zip(zip_filepath, target_size=(IMG_SIZE, IMG_SIZE), max_samples=None):
    if not os.path.exists(zip_filepath): raise FileNotFoundError(f"Zip file not found: {zip_filepath}")
    all_images = []
    with zipfile.ZipFile(zip_filepath, 'r') as zip_ref:
        image_files = [f for f in zip_ref.namelist() if f.startswith('images/') and f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if max_samples:
            np.random.shuffle(image_files)
            image_files = image_files[:max_samples]
        for filename in tqdm(image_files, desc="Loading Images"):
            with zip_ref.open(filename) as file:
                try:
                    with Image.open(io.BytesIO(file.read())) as img:
                        img = img.convert('RGB').resize(target_size, Image.Resampling.LANCZOS)
                        all_images.append(np.array(img))
                except Exception: pass # Skip corrupted files
    return np.array(all_images)

# ==============================================================================
# 3. Main Compression & Decompression Pipeline
# ==============================================================================
def run_compression_pipeline():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_filename = "anime_som_codebook.pt"
    model_path = os.path.join(script_dir, model_filename)
    zip_path = os.path.join(script_dir, "archive.zip")

    # --- Step 1: Load data and split into train/test sets ---
    print("--- Loading and Preparing Data ---")
    X_full = load_anime_faces_from_zip(zip_path, max_samples=MAX_IMAGES_TO_LOAD)
    X_full_normalized = X_full / 255.0
    
    # Split the data
    X_train = X_full_normalized[:TRAIN_SET_SIZE]
    X_test = X_full_normalized[TRAIN_SET_SIZE:TRAIN_SET_SIZE + TEST_SET_SIZE]

    # Flatten the data for the SOM
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)

    # --- Step 2: Train the Model (or load if it exists) ---
    print("\n--- Step 2: Training Phase ---")
    if not os.path.exists(model_path):
        print("No pre-trained model found. Training a new SOM codebook...")
        som_trainer = SOMCuda(grid_size=GRID_SIZE, input_dim=INPUT_DIM, device='cuda', learning_rate=0.6)
        som_trainer.train(X_train_flat, num_iterations=NUM_TRAINING_ITERATIONS)
        som_trainer.save(model_path)
    else:
        print(f"Found pre-trained model at {model_path}")

    # --- Step 3: Load the Model and Codebook ---
    print("\n--- Step 3: Loading Codebook for Compression ---")
    som_compressor = SOMCuda(grid_size=GRID_SIZE, input_dim=INPUT_DIM, device='cuda')
    som_compressor.load(model_path)
    codebook = som_compressor.get_weights()

    # --- Step 4: Compress the Test Data ---
    print("\n--- Step 4: Compression Phase ---")
    compressed_indices = som_compressor.get_bmu_coordinates(X_test_flat)

    # --- Step 5: Decompress the Data ---
    print("\n--- Step 5: Decompression Phase ---")
    # Decompress by looking up the indices in the codebook
    decompressed_data_flat = codebook[compressed_indices[:, 0], compressed_indices[:, 1]]

    # --- Step 6: Analysis and Visualization ---
    print("\n--- Step 6: Analysis ---")
    original_size = X_test_flat.nbytes
    codebook_size = codebook.nbytes
    indices_size = compressed_indices.nbytes
    total_compressed_size = codebook_size + indices_size
    
    print(f"Original test data size:      {original_size / 1e6:.3f} MB")
    print(f"Compressed indices size:    {indices_size / 1e6:.3f} MB")
    print(f"Codebook (model) size:      {codebook_size / 1e6:.3f} MB")
    print(f"Total compressed size:      {total_compressed_size / 1e6:.3f} MB")
    print(f"Compression Ratio:          {original_size / total_compressed_size:.2f}x")
    
    mse = np.mean((X_test_flat - decompressed_data_flat)**2)
    print(f"Reconstruction MSE:         {mse:.6f}")
    
    # Visualize the results
    fig, axes = plt.subplots(2, 10, figsize=(20, 5))
    for i in range(10):
        # Original image from the un-flattened test set
        axes[0, i].imshow(X_test[i])
        axes[0, i].set_title("Original")
        axes[0, i].axis('off')
        
        # Decompressed image, reshaped from flat vector
        decompressed_image = decompressed_data_flat[i].reshape(IMG_SIZE, IMG_SIZE, CHANNELS)
        axes[1, i].imshow(np.clip(decompressed_image, 0, 1)) # Clip for valid display range
        axes[1, i].set_title("Decompressed")
        axes[1, i].axis('off')
        
    fig.suptitle(f"SOM Compression | {GRID_SIZE[0]}x{GRID_SIZE[1]} Codebook | Reconstruction MSE: {mse:.4f}", fontsize=16)
    
    plot_filename = "anime_som_compression_comparison.png"
    plot_path = os.path.join(script_dir, plot_filename)
    fig.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\nComparison plot saved to: {plot_path}")
    
    plt.show()

if __name__ == "__main__":
    run_compression_pipeline()