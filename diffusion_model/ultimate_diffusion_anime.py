# ultimate_anime_diffusion_save_every_2_epochs.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import zipfile
import io
from PIL import Image

# ==============================================================================
# 0. Configuration
# ==============================================================================
# --- Hyperparameters ---
IMG_SIZE = 64
CHANNELS = 3
TIMESTEPS = 1000
BATCH_SIZE = 64
LEARNING_RATE = 2e-5
EPOCHS = 100
MAX_IMAGES_FOR_TRAINING = 100000
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- Directory Setup ---
script_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(script_dir, "ultimate_anime_diffusion")
os.makedirs(output_dir, exist_ok=True)
print(f"Using device: {DEVICE}")
print(f"Outputs will be saved to: {output_dir}")

# ==============================================================================
# 1. Diffusion Process Utils
# ==============================================================================
def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)

class DiffusionUtils:
    def __init__(self, timesteps=TIMESTEPS, beta_schedule='cosine', device=DEVICE):
        self.timesteps = timesteps
        self.device = device
        self.betas = cosine_beta_schedule(timesteps).to(device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
    
    def q_sample(self, x_start, t, noise=None):
        if noise is None: noise = torch.randn_like(x_start)
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].reshape(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].reshape(-1, 1, 1, 1)
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    @torch.no_grad()
    def p_sample_step(self, model, x, t):
        betas_t = self.betas[t].reshape(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].reshape(-1, 1, 1, 1)
        sqrt_recip_alphas_t = self.sqrt_recip_alphas[t].reshape(-1, 1, 1, 1)
        predicted_noise = model(x, t)
        model_mean = sqrt_recip_alphas_t * (x - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t)
        if t[0] == 0:
            return model_mean
        else:
            posterior_variance_t = self.posterior_variance[t].reshape(-1, 1, 1, 1)
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance_t) * noise

# ==============================================================================
# 2. U-Net Architecture for Denoising
# ==============================================================================
class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = np.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, dropout=0.1):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)
        self.block1 = nn.Sequential(nn.GroupNorm(8, in_channels), nn.SiLU(), nn.Conv2d(in_channels, out_channels, 3, padding=1))
        self.block2 = nn.Sequential(nn.GroupNorm(8, out_channels), nn.SiLU(), nn.Dropout(dropout), nn.Conv2d(out_channels, out_channels, 3, padding=1))
        self.shortcut = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
    def forward(self, x, time_emb):
        h = self.block1(x)
        h += self.time_mlp(time_emb)[:, :, None, None]
        h = self.block2(h)
        return h + self.shortcut(x)

class Attention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.group_norm = nn.GroupNorm(8, channels)
        self.to_qkv = nn.Conv2d(channels, channels * 3, 1, bias=False)
        self.to_out = nn.Conv2d(channels, channels, 1)
    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(self.group_norm(x)).chunk(3, dim=1)
        q, k, v = map(lambda t: t.reshape(b, c, h * w).transpose(1, 2), qkv)
        sim = torch.bmm(q, k.transpose(1, 2)) * (c ** -0.5)
        attn = sim.softmax(dim=-1)
        out = torch.bmm(attn, v).transpose(1, 2).reshape(b, c, h, w)
        return self.to_out(out) + x

class UNet(nn.Module):
    def __init__(self, in_channels=CHANNELS, model_channels=64, out_channels=CHANNELS, num_res_blocks=2, 
                 attention_resolutions=(16, 8), channel_mult=(1, 2, 2, 4), dropout=0.1):
        super().__init__()
        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(TimeEmbedding(model_channels), nn.Linear(model_channels, time_embed_dim), nn.SiLU(), nn.Linear(time_embed_dim, time_embed_dim))
        
        self.input_blocks = nn.ModuleList([nn.Conv2d(in_channels, model_channels, 3, padding=1)])
        input_block_channels = [model_channels]
        ch = model_channels
        ds = 1
        
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [ResidualBlock(ch, mult * model_channels, time_embed_dim, dropout)]
                ch = mult * model_channels
                if ds in attention_resolutions: layers.append(Attention(ch))
                self.input_blocks.append(nn.Sequential(*layers))
                input_block_channels.append(ch)
            if level != len(channel_mult) - 1:
                self.input_blocks.append(nn.Conv2d(ch, ch, 3, stride=2, padding=1))
                input_block_channels.append(ch)
                ds *= 2
        
        self.middle_block = nn.Sequential(ResidualBlock(ch, ch, time_embed_dim, dropout), Attention(ch), ResidualBlock(ch, ch, time_embed_dim, dropout))
        
        self.output_blocks = nn.ModuleList([])
        for level, mult in reversed(list(enumerate(channel_mult))):
            for i in range(num_res_blocks + 1):
                layers = [ResidualBlock(ch + input_block_channels.pop(), mult * model_channels, time_embed_dim, dropout)]
                ch = mult * model_channels
                if ds in attention_resolutions: layers.append(Attention(ch))
                if level and i == num_res_blocks:
                    layers.append(nn.ConvTranspose2d(ch, ch, 4, stride=2, padding=1))
                    ds //= 2
                self.output_blocks.append(nn.Sequential(*layers))
        
        self.out = nn.Sequential(nn.GroupNorm(8, ch), nn.SiLU(), nn.Conv2d(ch, out_channels, 3, padding=1))
    
    def forward(self, x, timesteps):
        time_emb = self.time_embed(timesteps)
        h = self.input_blocks[0](x)
        hs = [h]
        
        for module in self.input_blocks[1:]:
            if isinstance(module, nn.Sequential):
                for layer in module:
                    if isinstance(layer, ResidualBlock):
                        h = layer(h, time_emb)
                    else:
                        h = layer(h)
            else:
                h = module(h)
            hs.append(h)
        
        for layer in self.middle_block:
            if isinstance(layer, ResidualBlock):
                h = layer(h, time_emb)
            else:
                h = layer(h)
        
        for module in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            for layer in module:
                if isinstance(layer, ResidualBlock):
                    h = layer(h, time_emb)
                else:
                    h = layer(h)
        
        return self.out(h)

# ==============================================================================
# 3. Helper Functions (Data Loading, Plotting)
# ==============================================================================
def load_anime_faces_from_zip(zip_filepath, target_size=(IMG_SIZE, IMG_SIZE), max_samples=None):
    if not os.path.exists(zip_filepath): raise FileNotFoundError(f"Zip file not found: {zip_filepath}")
    all_images = []
    with zipfile.ZipFile(zip_filepath, 'r') as zf:
        image_files = [f for f in zf.namelist() if f.lower().endswith(('.png', '.jpg', '.jpeg')) and not f.startswith('__MACOSX/')]
        if max_samples and max_samples < len(image_files):
            print(f"Randomly selecting {max_samples} images.")
            np.random.shuffle(image_files)
            image_files = image_files[:max_samples]
        for filename in tqdm(image_files, desc="Loading Images into Memory"):
            with zf.open(filename) as file:
                try:
                    img = Image.open(io.BytesIO(file.read())).convert('RGB').resize(target_size, Image.Resampling.LANCZOS)
                    all_images.append(np.array(img))
                except Exception: pass
    return np.array(all_images)

def plot_and_save_loss(losses, output_path):
    plt.figure(figsize=(10, 5))
    plt.plot(losses, label='MSE Loss')
    plt.title("Training Loss Evolution")
    plt.xlabel("Epoch")
    plt.ylabel("Average Loss")
    plt.grid(True)
    plt.legend()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Loss plot saved to {output_path}")

@torch.no_grad()
def sample_and_save_images(model, diffusion_utils, epoch, output_dir):
    model.eval()
    print(f"\n--- Sampling images at Epoch {epoch} ---")
    x = torch.randn(16, CHANNELS, IMG_SIZE, IMG_SIZE, device=DEVICE)
    for i in tqdm(reversed(range(0, diffusion_utils.timesteps)), desc='Sampling', leave=False):
        t = torch.full((16,), i, device=DEVICE, dtype=torch.long)
        x = diffusion_utils.p_sample_step(model, x, t)
    
    samples = (x.clamp(-1, 1) + 1) / 2
    grid = torchvision.utils.make_grid(samples, nrow=4)
    
    plt.figure(figsize=(8, 8))
    plt.imshow(grid.permute(1, 2, 0).cpu())
    plt.axis('off')
    plt.title(f"Generated Anime Faces - Epoch {epoch}")
    filepath = os.path.join(output_dir, f"samples_epoch_{epoch}.png")
    plt.savefig(filepath, dpi=150)
    plt.close()
    model.train()
    print(f"Sample grid saved to {filepath}")

# ==============================================================================
# 4. Training Pipeline
# ==============================================================================
def train():
    zip_path = os.path.join(script_dir, "archive.zip")
    images_np = load_anime_faces_from_zip(zip_path, max_samples=MAX_IMAGES_FOR_TRAINING)
    images_np = (images_np / 127.5) - 1.0
    images_np = images_np.transpose(0, 3, 1, 2)
    data_tensor = torch.from_numpy(images_np).float()
    print(f"Data tensor created with shape: {data_tensor.shape}")

    diffusion_utils = DiffusionUtils(device=DEVICE)
    model = UNet().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    all_epoch_losses = []
    print("--- Starting Training ---")

    for epoch in range(1, EPOCHS + 1):
        epoch_loss = 0.0
        shuffled_data = data_tensor[torch.randperm(data_tensor.size(0))]
        
        progress_bar = tqdm(range(0, len(shuffled_data), BATCH_SIZE), desc=f"Epoch {epoch}/{EPOCHS}")
        for i in progress_bar:
            optimizer.zero_grad()
            
            batch = shuffled_data[i:i+BATCH_SIZE].to(DEVICE)
            t = torch.randint(0, diffusion_utils.timesteps, (batch.shape[0],), device=DEVICE).long()
            noise = torch.randn_like(batch)
            
            x_noisy = diffusion_utils.q_sample(batch, t, noise)
            predicted_noise = model(x_noisy, t)
            
            loss = F.mse_loss(noise, predicted_noise)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = epoch_loss / (len(shuffled_data) / BATCH_SIZE)
        all_epoch_losses.append(avg_loss)
        print(f'Epoch {epoch} completed. Average loss: {avg_loss:.4f}')
        
        # ==========================================================================
        # --- THIS IS THE MODIFIED SAVING BLOCK ---
        # ==========================================================================
        # Save samples, loss plot, and model checkpoint every 2 epochs
        if epoch % 2 == 0 or epoch == EPOCHS:
            sample_and_save_images(model, diffusion_utils, epoch, output_dir)
            plot_and_save_loss(all_epoch_losses, os.path.join(output_dir, "loss_evolution.png"))
            torch.save(model.state_dict(), os.path.join(output_dir, f"model_epoch_{epoch}.pt"))

    print("--- Training Complete ---")
    torch.save(model.state_dict(), os.path.join(output_dir, "model_final.pt"))

# ==============================================================================
# 5. Main Execution
# ==============================================================================
if __name__ == "__main__":
    train()