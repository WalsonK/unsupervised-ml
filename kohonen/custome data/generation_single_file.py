# som_stroke_generation.py
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import math
import os
# ==============================================================================
# 1. New Data Preprocessing: Resampling Strokes to a Fixed Length
# ==============================================================================
def resample_stroke(stroke, num_points=100):
    """
    Resamples a stroke sequence to a fixed number of points.
    
    Args:
        stroke (np.ndarray): The raw (dx, dy, pen_state) stroke data.
        num_points (int): The desired number of points in the output.
    Returns:
        np.ndarray: A (num_points, 2) array of resampled (x, y) coordinates.
    """
    # 1. Convert to absolute coordinates
    points = np.cumsum(stroke[:, :2], axis=0)
    
    # 2. Calculate the distance between each point
    deltas = np.sqrt(np.sum(np.diff(points, axis=0)**2, axis=1))
    cumulative_dist = np.concatenate([[0], np.cumsum(deltas)])
    
    total_dist = cumulative_dist[-1]
    
    # Avoid division by zero for single-point drawings
    if total_dist == 0:
        return np.tile(points[0], (num_points, 1))
        
    # 3. Create a new, evenly spaced set of distances
    new_distances = np.linspace(0, total_dist, num_points)
    
    # 4. Interpolate x and y coordinates at these new distances
    new_x = np.interp(new_distances, cumulative_dist, points[:, 0])
    new_y = np.interp(new_distances, cumulative_dist, points[:, 1])
    
    return np.stack([new_x, new_y], axis=1)

def load_single_quickdraw_file(filepath, num_points=100, max_samples=50000):
    """
    Loads raw stroke data from a single .npz file and preprocesses it into fixed-length resampled paths.
    
    Args:
        filepath (str): Path to the specific .npz file
        num_points (int): Number of points to resample each stroke to
        max_samples (int): Maximum number of samples to load from the file
    
    Returns:
        np.ndarray: Processed stroke data
    """
    all_paths = []
    
    # Check if file exists
    if not os.path.exists(filepath):
        print(f"ERROR: The file '{filepath}' was not found.")
        return None
    
    # Extract class name from filename
    filename = os.path.basename(filepath)
    if filename.startswith('sketchrnn_') and filename.endswith('.full.npz'):
        class_name = filename.replace('sketchrnn_','').replace('.full.npz','')
    else:
        class_name = filename.replace('.npz', '').replace('.full', '')
    
    print(f"Loading and resampling '{class_name}' strokes to {num_points}-point paths...")
    
    try:
        # Load the data
        data = np.load(filepath, encoding='latin1', allow_pickle=True)
        strokes = data['train'][:max_samples]
        
        print(f"Processing {len(strokes)} strokes from {class_name}...")
        
        for stroke in tqdm(strokes, desc=f"  Processing {class_name.capitalize()}"):
            resampled = resample_stroke(stroke, num_points=num_points)
            all_paths.append(resampled.flatten()) # Flatten to a 1D vector
            
    except Exception as e:
        print(f"ERROR loading file '{filepath}': {e}")
        return None
    
    X = np.array(all_paths, dtype=np.float32)
    
    # Normalize the data to be centered around zero
    X -= np.mean(X, axis=0)
    
    print(f"Dataset loaded and processed: {X.shape[0]} samples, each with {X.shape[1]} dimensions.")
    return X, class_name

# ==============================================================================
# 2. Self-Organizing Map (SOM) Class (No changes needed)
# ==============================================================================
class SOMCuda:
    def __init__(self, grid_size, input_dim, learning_rate=0.5, sigma=None, device='cuda', **kwargs):
        self.grid_height,self.grid_width=grid_size;self.input_dim=input_dim;self.initial_learning_rate=learning_rate;self.learning_rate=learning_rate
        if sigma is None: self.initial_sigma=max(self.grid_height,self.grid_width)/2.0
        else: self.initial_sigma=float(sigma)
        self.sigma=self.initial_sigma;
        if device=='cuda' and not torch.cuda.is_available(): device='cpu'
        self.device=torch.device(device); self.weights=torch.rand(self.grid_height,self.grid_width,self.input_dim,device=self.device,dtype=torch.float32);self._create_coordinate_grid()
    def save(self, filepath): torch.save(self.weights, filepath); print(f"SOM codebook saved to {filepath}")
    def load(self, filepath): self.weights=torch.load(filepath,map_location=self.device);print(f"SOM codebook loaded from {filepath}");self.grid_height,self.grid_width,self.input_dim=self.weights.shape;self._create_coordinate_grid()
    def _create_coordinate_grid(self): y,x=torch.meshgrid(torch.arange(self.grid_height,device=self.device),torch.arange(self.grid_width,device=self.device),indexing='ij');self.neuron_coords=torch.stack([y,x],dim=-1).float()
    def _find_bmu(self,v): d=torch.sum((self.weights-v)**2,dim=2);idx=torch.argmin(d);return idx//self.grid_width,idx%self.grid_width
    def _update_weights(self,v,bmu_y,bmu_x): bmu=torch.tensor([bmu_y,bmu_x],device=self.device,dtype=torch.float32);d_sq=torch.sum((self.neuron_coords-bmu)**2,dim=2);influence=torch.exp(-d_sq/(2*self.sigma**2)).unsqueeze(2);self.weights+=influence*self.learning_rate*(v-self.weights)
    def _decay_parameters(self,i,total):frac=i/total;self.learning_rate=self.initial_learning_rate*(1-frac);self.sigma=self.initial_sigma*(1-frac)
    def train(self,data,num_iterations):
        if isinstance(data,np.ndarray):data=torch.from_numpy(data).float().to(self.device)
        for i in tqdm(range(num_iterations),desc=f"Training ({num_iterations} iters)"): v=data[torch.randint(0,data.shape[0],(1,)).item()];self._decay_parameters(i,num_iterations);bmu_y,bmu_x=self._find_bmu(v);self._update_weights(v,bmu_y,bmu_x)
    def get_weights(self): return self.weights.cpu().numpy()

# ==============================================================================
# 3. New Visualization and Generation Functions
# ==============================================================================
def plot_resampled_stroke(ax, path_vector):
    """Takes a flattened path vector and plots it as a line drawing."""
    # Reshape the vector back into (num_points, 2)
    points = path_vector.reshape(-1, 2)
    ax.plot(points[:, 0], points[:, 1], color='black', linewidth=1.5)
    ax.axis('off'); ax.set_aspect('equal')

def generate_samples_from_som(som, num_samples=50):
    """Generates new path vectors from a trained SOM using bilinear interpolation."""
    print(f"Generating {num_samples} new path samples from the SOM's latent space...")
    codebook = som.get_weights()
    if codebook is None: raise ValueError("SOM model has not been loaded.")
    rand_y = np.random.uniform(0, som.grid_height-1, num_samples); rand_x = np.random.uniform(0, som.grid_width-1, num_samples)
    y0 = np.floor(rand_y).astype(int); x0 = np.floor(rand_x).astype(int); y1 = y0+1; x1 = x0+1
    w_tl = codebook[y0, x0]; w_tr = codebook[y0, x1]; w_bl = codebook[y1, x0]; w_br = codebook[y1, x1]
    wy = (rand_y - y0).reshape(-1, 1); wx = (rand_x - x0).reshape(-1, 1)
    top = (1 - wx) * w_tl + wx * w_tr; bottom = (1 - wx) * w_bl + wx * w_br
    return (1 - wy) * top + wy * bottom

# ==============================================================================
# 4. Main Generation Pipeline
# ==============================================================================
def run_stroke_generation_pipeline(npz_filepath):
    """
    Run the stroke generation pipeline for a single .npz file.
    
    Args:
        npz_filepath (str): Path to the specific .npz file to process
    """
    # --- Parameters ---
    num_resample_points = 100
    input_dim = num_resample_points * 2  # 100 points * 2 coordinates (x, y)
    grid_size = (30, 30)
    num_training_iterations = 150000
    num_to_generate = 60
    
    # --- Setup Paths ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Create model filename based on the input file
    base_filename = os.path.basename(npz_filepath).replace('.npz', '').replace('.full', '')
    model_filename = f"som_{base_filename}_model.pt"
    model_path = os.path.join(script_dir, model_filename)
    
    output_folder_name = f"som_{base_filename}_generation"
    save_dir = os.path.join(script_dir, output_folder_name)
    os.makedirs(save_dir, exist_ok=True)
    
    # --- Step 1: Train the Model (or load if it exists) ---
    print("--- Step 1: Training Phase ---")
    if not os.path.exists(model_path):
        print(f"No pre-trained model found for {base_filename}. Training a new SOM...")
        result = load_single_quickdraw_file(npz_filepath, num_points=num_resample_points, max_samples=5000)
        if result is None: 
            return
        
        X_train, class_name = result
        
        som_trainer = SOMCuda(grid_size=grid_size, input_dim=input_dim, device='cuda')
        som_trainer.train(X_train, num_iterations=num_training_iterations)
        som_trainer.save(model_path)
    else:
        print(f"Found pre-trained model at {model_path}")
        # Still need to get class name for visualization
        result = load_single_quickdraw_file(npz_filepath, num_points=num_resample_points, max_samples=1)
        if result is None: 
            class_name = "unknown"
        else:
            _, class_name = result
    
    # --- Step 2: Load Model and Generate Data ---
    print("\n--- Step 2: Generation Phase ---")
    som_generator = SOMCuda(grid_size=grid_size, input_dim=input_dim, device='cuda')
    som_generator.load(model_path)
    generated_paths = generate_samples_from_som(som_generator, num_samples=num_to_generate)
    
    # --- Step 3: Visualize and Save the Generated Data ---
    print("Visualizing generated stroke paths...")
    rows, cols = 6, 10
    fig, axes = plt.subplots(rows, cols, figsize=(15, 9))
    
    for i, ax in enumerate(axes.flat):
        if i < num_to_generate:
            plot_resampled_stroke(ax, generated_paths[i])
        else:
            ax.axis('off')
    
    fig.suptitle(f"Generated {class_name.capitalize()} Drawings from SOM ({num_to_generate} Samples)", fontsize=20)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plot_path = os.path.join(save_dir, f"som_{class_name}_generated_strokes.png")
    fig.savefig(plot_path, dpi=150)
    print(f"\nGenerated images plot saved to: {plot_path}")
    plt.show()

if __name__ == "__main__":
    # Specify the path to your single .npz file here
    # Example usage:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Option 1: Specify the file directly
    npz_file = os.path.join(script_dir, "quickdraw_data", "sketchrnn_crab.full.npz")
    
    # Option 2: Let user specify via command line argument
    import sys
    if len(sys.argv) > 1:
        npz_file = sys.argv[1]
    else:
        # Default file if no argument provided
        npz_file = os.path.join(script_dir, "quickdraw_data", "sketchrnn_crab.full.npz")
    
    print(f"Processing file: {npz_file}")
    run_stroke_generation_pipeline(npz_file)