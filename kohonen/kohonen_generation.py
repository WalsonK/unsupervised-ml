# som_generation.py

import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import math
import os

# ==============================================================================
# 1. Self-Organizing Map (SOM) Class (needed for loading the model)
# ==============================================================================

class SOMCuda:
    """
    Minimal SOM class needed to load a pre-trained model and access its weights.
    """
    def __init__(self, grid_size=(20, 20), input_dim=784, device='cuda'):
        self.grid_height, self.grid_width = grid_size
        self.input_dim = input_dim
        
        if device == 'cuda' and not torch.cuda.is_available():
            print("WARNING: CUDA not available, falling back to CPU.")
            device = 'cpu'
        self.device = torch.device(device)
        
        # We don't initialize weights randomly, as we will load them.
        self.weights = None

    def load(self, filepath):
        """Loads the SOM's weights (the codebook) from a file."""
        self.weights = torch.load(filepath, map_location=self.device)
        print(f"SOM codebook loaded from {filepath}")
        self.grid_height, self.grid_width, self.input_dim = self.weights.shape

    def get_weights(self):
        """Returns the weights as a NumPy array on the CPU."""
        if self.weights is not None:
            return self.weights.cpu().numpy()
        return None

# ==============================================================================
# 2. Data Generation Function
# ==============================================================================

def generate_samples_from_som(som, num_samples=50):
    """
    Generates new data samples from a trained SOM using bilinear interpolation.

    Args:
        som (SOMCuda): The trained SOM instance containing the codebook.
        num_samples (int): The number of new images to generate.

    Returns:
        np.ndarray: An array of newly generated image vectors.
    """
    print(f"Generating {num_samples} new samples from the SOM's latent space...")
    codebook = som.get_weights()
    if codebook is None:
        raise ValueError("SOM model has not been loaded. Cannot generate samples.")

    # --- Generate random floating-point coordinates in the latent space ---
    # We subtract 1 to ensure we don't go out of bounds when finding the bottom-right corner.
    rand_y = np.random.uniform(0, som.grid_height - 1, num_samples)
    rand_x = np.random.uniform(0, som.grid_width - 1, num_samples)

    # --- Find the four surrounding integer grid points (corners) ---
    y0 = np.floor(rand_y).astype(int)
    x0 = np.floor(rand_x).astype(int)
    y1 = y0 + 1
    x1 = x0 + 1

    # --- Get the weight vectors (prototypes) of the four corner neurons ---
    w_tl = codebook[y0, x0]  # Top-Left
    w_tr = codebook[y0, x1]  # Top-Right
    w_bl = codebook[y1, x0]  # Bottom-Left
    w_br = codebook[y1, x1]  # Bottom-Right

    # --- Calculate interpolation weights ---
    # These are the fractional distances from the top-left corner.
    wy = (rand_y - y0).reshape(-1, 1)
    wx = (rand_x - x0).reshape(-1, 1)

    # --- Perform Bilinear Interpolation ---
    # 1. Interpolate along the x-axis for the top two corners.
    top_interp = (1 - wx) * w_tl + wx * w_tr
    # 2. Interpolate along the x-axis for the bottom two corners.
    bottom_interp = (1 - wx) * w_bl + wx * w_br
    # 3. Interpolate along the y-axis between the two results.
    generated_vectors = (1 - wy) * top_interp + wy * bottom_interp

    return generated_vectors

# ==============================================================================
# 3. Main Generation Pipeline
# ==============================================================================

def run_generation_pipeline():
    # --- Parameters ---
    model_filename = "som_codebook.pt"
    num_to_generate = 50  # Let's generate 50 new images
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, model_filename)

    # --- Step 1: Check for and Load the Pre-trained Model ---
    if not os.path.exists(model_path):
        print(f"ERROR: Model file not found at '{model_path}'")
        print("Please run the compression script first to train and save the model.")
        return

    som_generator = SOMCuda(device='cuda')
    som_generator.load(model_path)

    # --- Step 2: Generate Fresh Data ---
    generated_images = generate_samples_from_som(som_generator, num_samples=num_to_generate)

    # --- Step 3: Visualize and Save the Generated Data ---
    print("Visualizing generated samples...")
    rows, cols = 5, 10
    fig, axes = plt.subplots(rows, cols, figsize=(15, 8))
    
    for i, ax in enumerate(axes.flat):
        if i < num_to_generate:
            ax.imshow(generated_images[i].reshape(28, 28), cmap='gray')
        ax.axis('off')
        
    fig.suptitle(f"Fresh Data Generated from SOM Latent Space ({num_to_generate} Samples)", fontsize=20)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save the figure
    plot_filename = "som_generated_samples.png"
    plot_path = os.path.join(script_dir, plot_filename)
    fig.savefig(plot_path, dpi=150)
    print(f"\nGenerated images plot saved to: {plot_path}")
    
    plt.show()

if __name__ == "__main__":
    run_generation_pipeline()