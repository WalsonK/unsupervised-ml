# diffusion_mnist.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

# ==============================================================================
# 1. Diffusion Process Utils
# ==============================================================================

def linear_beta_schedule(timesteps, start=0.0001, end=0.02):
    """Linear noise schedule"""
    return torch.linspace(start, end, timesteps)

def cosine_beta_schedule(timesteps, s=0.008):
    """Cosine noise schedule (often works better)"""
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)

class DiffusionUtils:
    def __init__(self, timesteps=1000, beta_schedule='cosine', device='cuda'):
        self.timesteps = timesteps
        self.device = device
        
        # Define beta schedule
        if beta_schedule == 'linear':
            self.betas = linear_beta_schedule(timesteps).to(device)
        else:
            self.betas = cosine_beta_schedule(timesteps).to(device)
        
        # Pre-compute useful values
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # For sampling
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        # For denoising
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
    
    def q_sample(self, x_start, t, noise=None):
        """Forward diffusion process: q(x_t | x_0)"""
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].reshape(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].reshape(-1, 1, 1, 1)
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    def p_sample_step(self, model, x, t, t_index):
        """Single denoising step: p(x_{t-1} | x_t)"""
        betas_t = self.betas[t].reshape(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].reshape(-1, 1, 1, 1)
        sqrt_recip_alphas_t = self.sqrt_recip_alphas[t].reshape(-1, 1, 1, 1)
        
        # Predict noise
        model_mean = sqrt_recip_alphas_t * (x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t)
        
        if t_index == 0:
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
        
        self.block1 = nn.Sequential(
            nn.GroupNorm(8, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, 3, padding=1)
        )
        
        self.block2 = nn.Sequential(
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv2d(out_channels, out_channels, 3, padding=1)
        )
        
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x, time_emb):
        h = self.block1(x)
        h += self.time_mlp(time_emb)[:, :, None, None]
        h = self.block2(h)
        return h + self.shortcut(x)

class Attention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.group_norm = nn.GroupNorm(8, channels)
        self.to_qkv = nn.Conv2d(channels, channels * 3, 1)
        self.to_out = nn.Conv2d(channels, channels, 1)
    
    def forward(self, x):
        b, c, h, w = x.shape
        x = self.group_norm(x)
        
        qkv = self.to_qkv(x)
        q, k, v = qkv.chunk(3, dim=1)
        
        q = q.view(b, c, h * w).transpose(1, 2)
        k = k.view(b, c, h * w).transpose(1, 2)
        v = v.view(b, c, h * w).transpose(1, 2)
        
        attention = torch.softmax(torch.bmm(q, k.transpose(1, 2)) / (c ** 0.5), dim=-1)
        out = torch.bmm(attention, v).transpose(1, 2).view(b, c, h, w)
        
        return self.to_out(out) + x

class UNet(nn.Module):
    def __init__(self, in_channels=1, model_channels=64, out_channels=1, num_res_blocks=2, 
                 attention_resolutions=[8], channel_mult=[1, 2, 4], dropout=0.1, num_timesteps=1000):
        super().__init__()
        
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.channel_mult = channel_mult
        self.dropout = dropout
        
        # Time embedding
        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            TimeEmbedding(model_channels),
            nn.Linear(model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim)
        )
        
        # Input projection
        self.input_blocks = nn.ModuleList([
            nn.Conv2d(in_channels, model_channels, 3, padding=1)
        ])
        
        # Downsampling
        input_block_channels = [model_channels]
        ch = model_channels
        ds = 1
        
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [ResidualBlock(ch, mult * model_channels, time_embed_dim, dropout)]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    layers.append(Attention(ch))
                self.input_blocks.append(nn.Sequential(*layers))
                input_block_channels.append(ch)
            
            if level != len(channel_mult) - 1:
                self.input_blocks.append(nn.Conv2d(ch, ch, 3, stride=2, padding=1))
                input_block_channels.append(ch)
                ds *= 2
        
        # Middle
        self.middle_block = nn.Sequential(
            ResidualBlock(ch, ch, time_embed_dim, dropout),
            Attention(ch),
            ResidualBlock(ch, ch, time_embed_dim, dropout)
        )
        
        # Upsampling
        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                layers = [ResidualBlock(ch + input_block_channels.pop(), 
                                      mult * model_channels, time_embed_dim, dropout)]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    layers.append(Attention(ch))
                if level and i == num_res_blocks:
                    layers.append(nn.ConvTranspose2d(ch, ch, 4, stride=2, padding=1))
                    ds //= 2
                self.output_blocks.append(nn.Sequential(*layers))
        
        # Output
        self.out = nn.Sequential(
            nn.GroupNorm(8, ch),
            nn.SiLU(),
            nn.Conv2d(ch, out_channels, 3, padding=1)
        )
    
    def forward(self, x, timesteps):
        # Time embedding
        time_emb = self.time_embed(timesteps)
        
        # Input
        h = self.input_blocks[0](x)
        hs = [h]
        
        # Downsampling
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
        
        # Middle
        for layer in self.middle_block:
            if isinstance(layer, ResidualBlock):
                h = layer(h, time_emb)
            else:
                h = layer(h)
        
        # Upsampling
        for module in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            for layer in module:
                if isinstance(layer, ResidualBlock):
                    h = layer(h, time_emb)
                else:
                    h = layer(h)
        
        return self.out(h)

# ==============================================================================
# 3. Training and Sampling
# ==============================================================================

class DiffusionModel:
    def __init__(self, model, diffusion_utils, device='cuda'):
        self.model = model
        self.diffusion_utils = diffusion_utils
        self.device = device
        
    def training_step(self, batch):
        """Single training step"""
        x_start = batch.to(self.device)
        batch_size = x_start.shape[0]
        
        # Sample random timesteps
        t = torch.randint(0, self.diffusion_utils.timesteps, (batch_size,), device=self.device).long()
        
        # Sample noise
        noise = torch.randn_like(x_start)
        
        # Forward diffusion
        x_noisy = self.diffusion_utils.q_sample(x_start, t, noise)
        
        # Predict noise
        predicted_noise = self.model(x_noisy, t)
        
        # Loss
        loss = F.mse_loss(noise, predicted_noise)
        return loss
    
    @torch.no_grad()
    def sample(self, num_samples=16, img_size=(1, 28, 28)):
        """Generate samples using DDPM"""
        self.model.eval()
        
        # Start from pure noise
        x = torch.randn(num_samples, *img_size, device=self.device)
        
        # Reverse diffusion
        for i in tqdm(reversed(range(0, self.diffusion_utils.timesteps)), desc='Sampling'):
            t = torch.full((num_samples,), i, device=self.device, dtype=torch.long)
            x = self.diffusion_utils.p_sample_step(self.model, x, t, i)
        
        self.model.train()
        return x

# ==============================================================================
# 4. Data Loading and Training Pipeline
# ==============================================================================

def get_mnist_dataloader(batch_size=128, img_size=28):
    """Load MNIST dataset"""
    # Get script directory for data path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, 'mnist_data')
    
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
    ])
    
    dataset = torchvision.datasets.MNIST(
        root=data_dir, train=True, download=True, transform=transform
    )
    
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
    )
    
    return dataloader

def train_diffusion_model():
    # Parameters
    timesteps = 1000
    batch_size = 128
    learning_rate = 2e-4
    num_epochs = 100
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Setup directories at script level
    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(script_dir, 'diffusion_outputs')
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"Using device: {device}")
    print(f"Outputs will be saved to: {save_dir}")
    
    # Initialize model and diffusion utils
    diffusion_utils = DiffusionUtils(timesteps=timesteps, device=device)
    model = UNet(in_channels=1, out_channels=1, num_timesteps=timesteps).to(device)
    diffusion_model = DiffusionModel(model, diffusion_utils, device)
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    # Data
    dataloader = get_mnist_dataloader(batch_size)
    
    # Training loop
    model.train()
    step = 0
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{num_epochs}')
        
        for batch_idx, (data, _) in enumerate(progress_bar):
            optimizer.zero_grad()
            
            loss = diffusion_model.training_step(data)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            
            epoch_loss += loss.item()
            step += 1
            
            # Update progress bar
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            # Sample every 1000 steps
            if step % 1000 == 0:
                print(f"\nGenerating samples at step {step}...")
                samples = diffusion_model.sample(num_samples=16)
                
                # Denormalize samples to [0, 1]
                samples = (samples + 1) / 2
                samples = torch.clamp(samples, 0, 1)
                
                # Plot
                fig, axes = plt.subplots(4, 4, figsize=(8, 8))
                for i, ax in enumerate(axes.flat):
                    ax.imshow(samples[i].cpu().squeeze(), cmap='gray')
                    ax.axis('off')
                
                plt.suptitle(f'Generated Samples (Step {step})')
                plt.tight_layout()
                plt.savefig(os.path.join(save_dir, f'samples_step_{step}.png'), dpi=150, bbox_inches='tight')
                plt.show()
        
        avg_loss = epoch_loss / len(dataloader)
        print(f'Epoch {epoch+1} completed. Average loss: {avg_loss:.4f}')
        
        # Save model every 10 epochs
        if (epoch + 1) % 10 == 0:
            model_path = os.path.join(save_dir, f'diffusion_model_epoch_{epoch+1}.pt')
            torch.save(model.state_dict(), model_path)
            print(f'Model saved at epoch {epoch+1}')
    
    # Final model save
    final_model_path = os.path.join(save_dir, 'diffusion_model_final.pt')
    torch.save(model.state_dict(), final_model_path)
    print('Training completed!')

def generate_samples_from_saved_model(model_path=None, num_samples=64):
    """Generate samples from a saved model"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(script_dir, 'diffusion_outputs')
    
    # Use default path if none provided
    if model_path is None:
        model_path = os.path.join(save_dir, 'diffusion_model_final.pt')
    
    # Load model
    diffusion_utils = DiffusionUtils(timesteps=1000, device=device)
    model = UNet(in_channels=1, out_channels=1, num_timesteps=1000).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    diffusion_model = DiffusionModel(model, diffusion_utils, device)
    
    # Generate samples
    print(f"Generating {num_samples} samples...")
    samples = diffusion_model.sample(num_samples=num_samples)
    
    # Denormalize
    samples = (samples + 1) / 2
    samples = torch.clamp(samples, 0, 1)
    
    # Plot
    grid_size = int(np.sqrt(num_samples))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))
    
    for i, ax in enumerate(axes.flat):
        if i < num_samples:
            ax.imshow(samples[i].cpu().squeeze(), cmap='gray')
        ax.axis('off')
    
    plt.suptitle(f'Generated MNIST Digits ({num_samples} samples)', fontsize=16)
    plt.tight_layout()
    output_path = os.path.join(save_dir, 'final_generated_samples.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Generated samples saved to: {output_path}")
    plt.show()

# ==============================================================================
# 5. Main Execution
# ==============================================================================

if __name__ == "__main__":
    import sys
    
    # Get script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    if len(sys.argv) > 1 and sys.argv[1] == 'generate':
        # Generate samples from saved model
        model_path = os.path.join(script_dir, 'diffusion_outputs', 'diffusion_model_final.pt')
        if os.path.exists(model_path):
            generate_samples_from_saved_model(model_path, num_samples=64)
        else:
            print(f"Model not found at {model_path}. Train the model first.")
    else:
        # Train the model
        train_diffusion_model()