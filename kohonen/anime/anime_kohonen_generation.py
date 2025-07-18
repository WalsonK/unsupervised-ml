# anime_som_train_and_generate.py
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
GRID_SIZE = (20, 20)
NUM_TRAINING_ITERATIONS = 100000
MAX_IMAGES_FOR_TRAINING = 5000 # Use a subset to train faster
MODEL_FILENAME = "anime_som_codebook.pt"
NUM_TO_GENERATE = 50

# ==============================================================================
# 1. Self-Organizing Map (SOM) Class (Full version with training)
# ==============================================================================
class SOMCuda:
    """
    Full SOM class with training, saving, and loading functionality.
    """
    def __init__(self, grid_size=GRID_SIZE, input_dim=INPUT_DIM, learning_rate=0.5,
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

    def get_weights(self):
        if self.weights is not None:
            return self.weights.cpu().numpy()
        return None

# ==============================================================================
# 2. Helper Functions (Data Loading and Generation)
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

def generate_samples_from_som(som, num_samples):
    print(f"\nGenerating {num_samples} new samples from the SOM's latent space...")
    codebook = som.get_weights()
    if codebook is None: raise ValueError("SOM model has not been loaded.")
    rand_y = np.random.uniform(0, som.grid_height - 1, num_samples)
    rand_x = np.random.uniform(0, som.grid_width - 1, num_samples)
    y0 = np.floor(rand_y).astype(int)
    x0 = np.floor(rand_x).astype(int)
    y1 = y0 + 1
    x1 = x0 + 1
    w_tl, w_tr, w_bl, w_br = codebook[y0, x0], codebook[y0, x1], codebook[y1, x0], codebook[y1, x1]
    wy, wx = (rand_y - y0).reshape(-1, 1), (rand_x - x0).reshape(-1, 1)
    top_interp = (1 - wx) * w_tl + wx * w_tr
    bottom_interp = (1 - wx) * w_bl + wx * w_br
    return (1 - wy) * top_interp + wy * bottom_interp

# ==============================================================================
# 3. Main Training and Generation Pipeline
# ==============================================================================
def run_train_and_generate_pipeline():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, MODEL_FILENAME)
    zip_path = os.path.join(script_dir, "archive.zip")

    # --- Step 1: Train the Model (or load if it exists) ---
    print("--- Step 1: Model Training/Loading ---")
    if not os.path.exists(model_path):
        print("No pre-trained model found. Training a new SOM codebook...")
        
        # Load, normalize, and flatten the dataset
        X_orig = load_anime_faces_from_zip(zip_path, max_samples=MAX_IMAGES_FOR_TRAINING)
        X_normalized = X_orig / 255.0
        X_flat = X_normalized.reshape(X_normalized.shape[0], -1)
        
        # Instantiate and train the SOM
        som_trainer = SOMCuda(learning_rate=0.6)
        som_trainer.train(X_flat, num_iterations=NUM_TRAINING_ITERATIONS)
        
        # Save the trained model for future use
        som_trainer.save(model_path)
    else:
        print(f"Found pre-trained model at {model_path}")

    # --- Step 2: Load the model and generate new faces ---
    print("\n--- Step 2: Generation Phase ---")
    som_generator = SOMCuda(device='cuda')
    som_generator.load(model_path)
    generated_faces = generate_samples_from_som(som_generator, num_samples=NUM_TO_GENERATE)

    # --- Step 3: Visualize and Save the Generated Data ---
    print("Visualizing generated anime faces...")
    rows, cols = 5, 10
    fig, axes = plt.subplots(rows, cols, figsize=(20, 10))

    for i, ax in enumerate(axes.flat):
        if i < NUM_TO_GENERATE:
            img = generated_faces[i].reshape(IMG_SIZE, IMG_SIZE, CHANNELS)
            img = np.clip(img, 0, 1) # Clip values for valid display
            ax.imshow(img)
        ax.axis('off')

    fig.suptitle(f"Fresh Anime Faces Generated from SOM Latent Space ({NUM_TO_GENERATE} Samples)", fontsize=20)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    plot_filename = "som_generated_anime_faces.png"
    plot_path = os.path.join(script_dir, plot_filename)
    fig.savefig(plot_path, dpi=150)
    print(f"\nGenerated images plot saved to: {plot_path}")

    plt.show()

if __name__ == "__main__":
    run_train_and_generate_pipeline()