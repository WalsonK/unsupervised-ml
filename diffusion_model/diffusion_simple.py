# ultra_simple_diffusion_periodic.py
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

# ==============================================================================
# 0. Configuration & Global Diffusion Math
# ==============================================================================
# --- Hyperparameters ---
IMG_SIZE = 28
FLAT_IMG_SIZE = IMG_SIZE * IMG_SIZE
BATCH_SIZE = 256
TIMESTEPS = 1000
EPOCHS = 100
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- Directory Setup ---
script_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(script_dir, "ultra_simple_outputs_periodic")
os.makedirs(output_dir, exist_ok=True)

print(f"Using device: {DEVICE}")
print(f"Outputs will be saved to: {output_dir}")

# --- Global Diffusion Variables (No Utilities Class) ---
betas = torch.linspace(0.0001, 0.02, TIMESTEPS, device=DEVICE)
alphas = 1.0 - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
posterior_variance = betas * (1.0 - torch.cat([torch.tensor([1.0], device=DEVICE), alphas_cumprod[:-1]])) / (1.0 - alphas_cumprod)

# ==============================================================================
# 1. The Model (Ultra-Simple MLP)
# ==============================================================================
class UltraSimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.time_emb = nn.Embedding(TIMESTEPS, 128)
        self.main_mlp = nn.Sequential(
            nn.Linear(FLAT_IMG_SIZE + 128, 256),
            nn.ReLU(),
            nn.Linear(256, FLAT_IMG_SIZE)
        )

    def forward(self, x, t):
        t_emb = self.time_emb(t)
        x = torch.cat([x, t_emb], dim=-1)
        return self.main_mlp(x)

# ==============================================================================
# 2. Helper Functions for Visualization
# ==============================================================================
def sample_and_save_images(model, epoch, output_dir):
    """Helper function to generate and save a grid of images."""
    print(f"\n--- Sampling images at Epoch {epoch} ---")
    model.eval()
    with torch.no_grad():
        x = torch.randn((64, FLAT_IMG_SIZE), device=DEVICE)
        for i in tqdm(reversed(range(TIMESTEPS)), desc="Sampling", leave=False):
            t = torch.full((64,), i, device=DEVICE, dtype=torch.long)
            predicted_noise = model(x, t)
            alpha_t = alphas[t].view(-1, 1)
            sqrt_one_minus_alpha_cumprod_t = sqrt_one_minus_alphas_cumprod[t].view(-1, 1)
            x = (1 / torch.sqrt(alpha_t)) * (x - ((1 - alpha_t) / sqrt_one_minus_alpha_cumprod_t) * predicted_noise)
            if i > 0:
                variance = posterior_variance[t].view(-1, 1)
                x += torch.sqrt(variance) * torch.randn_like(x)

    samples = (x + 1) / 2
    samples = torch.clamp(samples, 0, 1)
    samples = samples.reshape(-1, 1, IMG_SIZE, IMG_SIZE)
    grid_img = torchvision.utils.make_grid(samples.cpu(), nrow=8)
    plt.figure(figsize=(8, 8))
    plt.imshow(grid_img.permute(1, 2, 0))
    plt.title(f"Generated Digits - Epoch {epoch}")
    plt.axis('off')
    filepath = os.path.join(output_dir, f"samples_epoch_{epoch}.png")
    plt.savefig(filepath, dpi=150)
    plt.close()
    model.train()
    print(f"Saved sample grid to {filepath}")

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
# 3. Data Loading
# ==============================================================================
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: (x * 2) - 1),
    transforms.Lambda(lambda x: x.flatten())
])
dataset = torchvision.datasets.MNIST(root="./mnist_data", train=True, download=True, transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# ==============================================================================
# 4. Training Loop
# ==============================================================================
model = UltraSimpleMLP().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
loss_fn = nn.MSELoss()

# List to store the average loss of each epoch
all_epoch_losses = []

print("--- Starting Training ---")
for epoch in range(1, EPOCHS + 1):
    epoch_loss = 0.0
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}/{EPOCHS}")
    for batch, _ in progress_bar:
        optimizer.zero_grad()
        batch = batch.to(DEVICE)
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

    # Calculate and store the average loss for the epoch
    avg_epoch_loss = epoch_loss / len(dataloader)
    all_epoch_losses.append(avg_epoch_loss)

    # --- Periodic Sampling, Plotting, and Saving ---
    if epoch % 10 == 0:
        sample_and_save_images(model, epoch, output_dir)
        plot_and_save_loss(all_epoch_losses, epoch, output_dir)
        
        model_path = os.path.join(output_dir, f"model_epoch_{epoch}.pt")
        torch.save(model.state_dict(), model_path)

print("--- Training Complete ---")
final_model_path = os.path.join(output_dir, "model_final.pt")
torch.save(model.state_dict(), final_model_path)
print(f"Final model saved to {final_model_path}")