# ultra_simple_diffusion_periodic_anime_numpy_load_with_loss_plot.py
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import zipfile
import io
from PIL import Image
import numpy as np

# ==============================================================================
# 0. Configuration & Global Diffusion Math
# ==============================================================================
# --- Hyperparameters ---
IMG_SIZE = 64
CHANNELS = 3
FLAT_IMG_SIZE = IMG_SIZE * IMG_SIZE * CHANNELS
BATCH_SIZE = 128
TIMESTEPS = 1000
EPOCHS = 200
MAX_IMAGES_FOR_TRAINING = 100000 # Use a subset to train faster
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- Directory Setup ---
script_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(script_dir, "anime_diffusion_outputs_numpy_load")
os.makedirs(output_dir, exist_ok=True)
print(f"Using device: {DEVICE}")
print(f"Outputs will be saved to: {output_dir}")

# --- Global Diffusion Variables ---
betas = torch.linspace(0.0001, 0.02, TIMESTEPS, device=DEVICE)
alphas = 1.0 - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
posterior_variance = betas * (1.0 - torch.cat([torch.tensor([1.0], device=DEVICE), alphas_cumprod[:-1]])) / (1.0 - alphas_cumprod)

# ==============================================================================
# 1. The Model (Larger MLP for RGB)
# ==============================================================================
class AnimeSimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.time_emb = nn.Embedding(TIMESTEPS, 256)
        self.main_mlp = nn.Sequential(
            nn.Linear(FLAT_IMG_SIZE + 256, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, FLAT_IMG_SIZE)
        )
    
    def forward(self, x, t):
        t_emb = self.time_emb(t)
        x = torch.cat([x, t_emb], dim=-1)
        return self.main_mlp(x)

# ==============================================================================
# 2. Helper Functions
# ==============================================================================
def load_anime_faces_from_zip(zip_filepath, target_size=(IMG_SIZE, IMG_SIZE), max_samples=None):
    """
    Loads images from a zip file into a single NumPy array in memory.
    """
    if not os.path.exists(zip_filepath):
        raise FileNotFoundError(f"Zip file not found: {zip_filepath}")
        
    all_images = []
    with zipfile.ZipFile(zip_filepath, 'r') as zip_ref:
        image_files = [f for f in zip_ref.namelist() if f.lower().endswith(('.png', '.jpg', '.jpeg')) and not f.startswith('__MACOSX/')]
        
        if max_samples and max_samples < len(image_files):
            print(f"Randomly selecting {max_samples} images for training.")
            np.random.shuffle(image_files)
            image_files = image_files[:max_samples]
            
        for filename in tqdm(image_files, desc="Loading Images into Memory"):
            with zip_ref.open(filename) as file:
                try:
                    with Image.open(io.BytesIO(file.read())) as img:
                        img = img.convert('RGB').resize(target_size, Image.Resampling.LANCZOS)
                        all_images.append(np.array(img))
                except Exception:
                    pass
                    
    return np.array(all_images)

def sample_and_save_images(model, epoch, output_dir):
    """Generates and saves a grid of anime images."""
    print(f"\n--- Sampling anime images at Epoch {epoch} ---")
    model.eval()
    with torch.no_grad():
        x = torch.randn((16, FLAT_IMG_SIZE), device=DEVICE)
        for i in tqdm(reversed(range(TIMESTEPS)), desc="Sampling", leave=False):
            t = torch.full((16,), i, device=DEVICE, dtype=torch.long)
            predicted_noise = model(x, t)
            alpha_t = alphas[t].view(-1, 1)
            sqrt_one_minus_alpha_cumprod_t = sqrt_one_minus_alphas_cumprod[t].view(-1, 1)
            x = (1 / torch.sqrt(alpha_t)) * (x - ((1 - alpha_t) / sqrt_one_minus_alpha_cumprod_t) * predicted_noise)
            if i > 0:
                variance = posterior_variance[t].view(-1, 1)
                x += torch.sqrt(variance) * torch.randn_like(x)
    
    samples = (x + 1) / 2
    samples = torch.clamp(samples, 0, 1)
    samples = samples.reshape(-1, 3, IMG_SIZE, IMG_SIZE)
    grid_img = torchvision.utils.make_grid(samples.cpu(), nrow=4)
    
    plt.figure(figsize=(10, 10))
    plt.imshow(grid_img.permute(1, 2, 0))
    plt.title(f"Generated Anime Faces - Epoch {epoch}")
    plt.axis('off')
    filepath = os.path.join(output_dir, f"anime_samples_epoch_{epoch}.png")
    plt.savefig(filepath, dpi=150)
    plt.close()
    model.train()
    print(f"Saved anime sample grid to {filepath}")

# --- NEW FUNCTION TO PLOT LOSS ---
def plot_and_save_loss(losses, epoch, output_dir):
    """Helper function to plot and save the training loss."""
    print(f"--- Plotting loss at Epoch {epoch} ---")
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(losses) + 1), losses, marker='o', linestyle='-')
    plt.title(f"Training Loss per Epoch (up to Epoch {epoch})")
    plt.xlabel("Epoch")
    plt.ylabel("Average MSE Loss")
    plt.grid(True)
    filepath = os.path.join(output_dir, f"loss_plot_epoch_{epoch}.png")
    plt.savefig(filepath, dpi=150)
    plt.close()
    print(f"Saved loss plot to {filepath}")

# ==============================================================================
# 3. Data Loading (Using the SOM script's logic)
# ==============================================================================
ZIP_FILE_PATH = os.path.join(script_dir, "archive.zip")
X_orig = load_anime_faces_from_zip(ZIP_FILE_PATH, max_samples=MAX_IMAGES_FOR_TRAINING)
X_normalized = (X_orig / 127.5) - 1.0
X_flat = X_normalized.reshape(X_normalized.shape[0], -1)
data_tensor = torch.from_numpy(X_flat).float().to(DEVICE)
print(f"Data tensor created with shape: {data_tensor.shape} on device: {data_tensor.device}")

# ==============================================================================
# 4. Training Loop
# ==============================================================================
model = AnimeSimpleMLP().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
loss_fn = nn.MSELoss()
all_epoch_losses = []
print("--- Starting Anime Diffusion Training (loading all data to memory) ---")

for epoch in range(1, EPOCHS + 1):
    epoch_loss = 0.0
    indices = torch.randperm(data_tensor.shape[0])
    shuffled_data = data_tensor[indices]
    
    progress_bar = tqdm(range(0, len(shuffled_data), BATCH_SIZE), desc=f"Epoch {epoch}/{EPOCHS}")
    
    for i in progress_bar:
        batch = shuffled_data[i:i+BATCH_SIZE]
        optimizer.zero_grad()
        t = torch.randint(0, TIMESTEPS, (batch.shape[0],), device=DEVICE).long()
        
        sqrt_alphas_cumprod_t = sqrt_alphas_cumprod[t].view(-1, 1)
        sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod[t].view(-1, 1)
        noise = torch.randn_like(batch)
        x_noisy = sqrt_alphas_cumprod_t * batch + sqrt_one_minus_alphas_cumprod_t * noise
        
        predicted_noise = model(x_noisy, t)
        loss = loss_fn(noise, predicted_noise)
        
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_epoch_loss = epoch_loss / (len(shuffled_data) / BATCH_SIZE)
    all_epoch_losses.append(avg_epoch_loss)
    
    # --- MODIFIED PERIODIC SAVING BLOCK ---
    if epoch % 20 == 0 or epoch == EPOCHS:
        sample_and_save_images(model, epoch, output_dir)
        # Call the new plotting function here
        plot_and_save_loss(all_epoch_losses, epoch, output_dir)
        
        model_path = os.path.join(output_dir, f"anime_model_epoch_{epoch}.pt")
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")

print("--- Anime Diffusion Training Complete ---")
final_model_path = os.path.join(output_dir, "anime_model_final.pt")
torch.save(model.state_dict(), final_model_path)
print(f"Final anime model saved to {final_model_path}")