# som_quickdraw_from_strokes.py

import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import math
import os
import cv2  # New dependency for rendering strokes

# ==============================================================================
# 1. Stroke-to-Bitmap Conversion and Data Loading
# ==============================================================================

def strokes_to_bitmap(strokes, canvas_size=256, output_size=28):
    """
    Renders a single QuickDraw stroke sequence into a 28x28 bitmap image.

    Args:
        strokes (np.ndarray): An array of strokes, where each row is (dx, dy, pen_state).
        canvas_size (int): The size of the intermediate canvas to draw on.
        output_size (int): The final size of the output image.

    Returns:
        np.ndarray: A 28x28 grayscale image as a NumPy array.
    """
    # Calculate absolute coordinates
    x = np.cumsum(strokes[:, 0])
    y = np.cumsum(strokes[:, 1])
    
    # Get the bounding box of the drawing
    x_min, x_max = np.min(x), np.max(x)
    y_min, y_max = np.min(y), np.max(y)
    
    # Handle drawings that are just a single point
    if x_max - x_min == 0 and y_max - y_min == 0:
        return np.zeros((output_size, output_size), dtype=np.float32)

    # Create a blank canvas
    canvas = np.zeros((canvas_size, canvas_size), dtype=np.uint8)
    
    # Determine the scaling factor and padding to center the drawing
    scale = (canvas_size * 0.8) / max(x_max - x_min, y_max - y_min)
    padding_x = (canvas_size - (x_max - x_min) * scale) / 2
    padding_y = (canvas_size - (y_max - y_min) * scale) / 2
    
    # Draw the strokes
    current_x, current_y = 0, 0
    for stroke in strokes:
        dx, dy, pen_down = stroke
        start_x = int((current_x - x_min) * scale + padding_x)
        start_y = int((current_y - y_min) * scale + padding_y)
        
        current_x += dx
        current_y += dy
        
        end_x = int((current_x - x_min) * scale + padding_x)
        end_y = int((current_y - y_min) * scale + padding_y)
        
        if pen_down == 0: # Pen is down
            cv2.line(canvas, (start_x, start_y), (end_x, end_y), 255, thickness=4)
            
    # Resize the canvas to the final output size
    bitmap = cv2.resize(canvas, (output_size, output_size))
    return bitmap

def load_quickdraw_data_from_strokes(path, max_samples_per_class=2000, max_classes=10):
    """
    Loads data from a directory of QuickDraw sketchrnn .npz files and renders them to bitmaps.
    """
    all_bitmaps = []
    all_labels = []
    class_names = []

    try:
        npz_files = [f for f in os.listdir(path) if f.startswith('sketchrnn_') and f.endswith('.full.npz')]
    except FileNotFoundError:
        print(f"ERROR: The directory '{path}' was not found."); return None, None, None
        
    npz_files = sorted(npz_files)[:max_classes]
    
    if not npz_files: print(f"ERROR: No 'sketchrnn_...' files found in '{path}'."); return None, None, None

    print(f"Loading and rendering {len(npz_files)} classes from stroke data...")

    for label_index, filename in enumerate(npz_files):
        class_name = filename.replace('sketchrnn_', '').replace('.full.npz', '')
        class_names.append(class_name.capitalize())
        print(f"  Processing class: {class_name.capitalize()}")
        
        filepath = os.path.join(path, filename)
        
        # Load the stroke data, allowing pickles and using 'latin1' encoding for compatibility
        data = np.load(filepath, encoding='latin1', allow_pickle=True)
        strokes = data['train'][:max_samples_per_class] # Use the 'train' set
        
        class_bitmaps = []
        for stroke_set in tqdm(strokes, desc=f"  Rendering {class_name}"):
            bitmap = strokes_to_bitmap(stroke_set)
            class_bitmaps.append(bitmap.flatten()) # Flatten to 784-dim vector
        
        all_bitmaps.append(np.array(class_bitmaps, dtype=np.float32) / 255.0)
        all_labels.append(np.full(len(class_bitmaps), label_index))

    X = np.concatenate(all_bitmaps, axis=0)
    y = np.concatenate(all_labels, axis=0)
    
    permutation = np.random.permutation(X.shape[0])
    X, y = X[permutation], y[permutation]

    print(f"Dataset loaded successfully: {X.shape[0]} samples, {len(class_names)} classes.")
    return X, y, class_names

# ==============================================================================
# 2. Self-Organizing Map (SOM) Class (Identical to before)
# ==============================================================================

class SOMCuda: # No changes needed here
    def __init__(self, grid_size=(20, 20), input_dim=784, learning_rate=0.5, 
                 sigma=None, device='cuda', **kwargs):
        self.grid_height, self.grid_width = grid_size; self.input_dim = input_dim
        self.initial_learning_rate = learning_rate; self.learning_rate = learning_rate
        if sigma is None: self.initial_sigma = max(self.grid_height, self.grid_width)/2.0
        else: self.initial_sigma = float(sigma)
        self.sigma = self.initial_sigma
        if device == 'cuda' and not torch.cuda.is_available(): device = 'cpu'
        self.device = torch.device(device); print(f"Using device: {self.device}")
        self.weights = torch.rand(self.grid_height, self.grid_width, self.input_dim, device=self.device, dtype=torch.float32)
        self._create_coordinate_grid()
    def _create_coordinate_grid(self):
        y,x=torch.meshgrid(torch.arange(self.grid_height,device=self.device),torch.arange(self.grid_width,device=self.device),indexing='ij');self.neuron_coords=torch.stack([y,x],dim=-1).float()
    def _find_bmu(self,v): d=torch.sum((self.weights-v)**2,dim=2);idx=torch.argmin(d);return idx//self.grid_width,idx%self.grid_width
    def _update_weights(self,v,bmu_y,bmu_x):
        bmu=torch.tensor([bmu_y,bmu_x],device=self.device,dtype=torch.float32);d_sq=torch.sum((self.neuron_coords-bmu)**2,dim=2)
        influence=torch.exp(-d_sq/(2*self.sigma**2)).unsqueeze(2);self.weights+=influence*self.learning_rate*(v-self.weights)
    def _decay_parameters(self,i,total):frac=i/total;self.learning_rate=self.initial_learning_rate*(1-frac);self.sigma=self.initial_sigma*(1-frac)
    def train(self,data,num_iterations):
        if isinstance(data,np.ndarray):data=torch.from_numpy(data).float().to(self.device)
        for i in tqdm(range(num_iterations),desc=f"Training ({num_iterations} iters)"):
            v=data[torch.randint(0,data.shape[0],(1,)).item()];self._decay_parameters(i,num_iterations)
            bmu_y,bmu_x=self._find_bmu(v);self._update_weights(v,bmu_y,bmu_x)
    def get_bmu_coordinates(self,data):
        if isinstance(data,np.ndarray):data=torch.from_numpy(data).float().to(self.device)
        coords=[self._find_bmu(v) for v in data];return np.array([[c[0].cpu().item(),c[1].cpu().item()] for c in coords])
    def get_weights(self): return self.weights.cpu().numpy()

# ==============================================================================
# 3. Visualization and Main Driver (Minor changes for QuickDraw)
# ==============================================================================

def plot_projection(som,X,y,class_names,ax,n_iter): # Same as before
    print(f"  Plotting projection for {n_iter:,} iters..."); bmu_coords=som.get_bmu_coordinates(X); colors=plt.cm.tab10(np.linspace(0,1,len(class_names)))
    for i,name in enumerate(class_names):
        bmus=bmu_coords[y==i]
        if len(bmus)>0: jx=bmus[:,1]+np.random.rand(len(bmus))*0.8-0.4; jy=bmus[:,0]+np.random.rand(len(bmus))*0.8-0.4; ax.scatter(jx,jy,c=[colors[i]],label=name,s=5,alpha=0.7)
    ax.set_title(f'{n_iter:,} Iterations');ax.set_aspect('equal',adjustable='box');ax.set_xlim(-1,som.grid_width);ax.set_ylim(-1,som.grid_height);ax.invert_yaxis()

def plot_weight_grid(som,ax,n_iter): # Same as before
    print(f"  Plotting weight grid for {n_iter:,} iters..."); w=som.get_weights();h,wd,_=w.shape; canvas=np.ones((h*28,wd*28))
    for i in range(h):
        for j in range(wd): canvas[i*28:(i+1)*28,j*28:(j+1)*28]=w[i,j].reshape(28,28)
    ax.imshow(canvas,cmap='gray');ax.set_title(f'{n_iter:,} Iterations');ax.axis('off')

def run_quickdraw_evolution_analysis(output_folder_name="som_quickdraw_from_strokes"):
    # Get the absolute path to the directory containing this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Create the full, absolute path to the data directory
    data_folder_name = 'quickdraw_data'
    quickdraw_path = os.path.join(script_dir, data_folder_name)

    # Also create the output directory path relative to the script
    save_dir = os.path.join(script_dir, 'quickdraw_output')
    os.makedirs(save_dir, exist_ok=True)

    X, y, class_names = load_quickdraw_data_from_strokes(quickdraw_path, max_samples_per_class=2000, max_classes=10)
    if X is None: return

    grid_size=(25,25); checkpoints=[1000,5000,10000,20000,40000,60000,80000,100000]
    fig_proj,axes_proj=plt.subplots(2,4,figsize=(20,10),constrained_layout=True); axes_proj=axes_proj.flatten()
    fig_weights,axes_weights=plt.subplots(2,4,figsize=(16,16),constrained_layout=True); axes_weights=axes_weights.flatten()
    
    plot_indices=np.random.choice(len(X),2000,replace=False); X_plot,y_plot=X[plot_indices],y[plot_indices]
    
    for i, n_iter in enumerate(checkpoints):
        print(f"\n--- STAGE {i+1}/{len(checkpoints)}: Training SOM for {n_iter} iterations ---")
        som = SOMCuda(grid_size=grid_size, input_dim=X.shape[1], sigma=max(grid_size)/2, device='cuda')
        som.train(X, num_iterations=n_iter)
        plot_projection(som, X_plot, y_plot, class_names, axes_proj[i], n_iter)
        plot_weight_grid(som, axes_weights[i], n_iter)

    h,l=axes_proj[0].get_legend_handles_labels(); fig_proj.legend(h,l,loc='center right',title="Classes");fig_proj.suptitle('Evolution of QuickDraw (from Strokes) Data Projection',fontsize=24)
    fig_proj.subplots_adjust(right=0.90); proj_save_path=os.path.join(save_dir,"projection_evolution.png"); fig_proj.savefig(proj_save_path,dpi=150)
    print(f"\nProjection plot saved to: {proj_save_path}")

    fig_weights.suptitle("Evolution of Learned QuickDraw (from Strokes) Prototypes",fontsize=24)
    weights_save_path=os.path.join(save_dir,"weights_evolution.png"); fig_weights.savefig(weights_save_path,dpi=150)
    print(f"Weights plot saved to: {weights_save_path}")
    plt.show()

if __name__ == "__main__":
    run_quickdraw_evolution_analysis()