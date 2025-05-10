import sys
sys.path.append("./src/modules")
sys.path.append("./")
import os
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from .modules.lam.modules.blocks import SpatioTemporalTransformer, SpatioTransformer
from torch import Tensor

class LatentActionModel(nn.Module):
    """
    Latent Action Model (LAM) for learning latent representations of actions in videos.
    
    This model can operate in two modes:
    1. Global Actions Mode: Uses a single action token per timestep
    2. Distributed Actions Mode: Uses multiple action tokens (num_patches) per timestep
    
    The model implements a VAE-like architecture with a transformer-based encoder and decoder.
    """

    def __init__(
            self,
            in_dim: int = 8,              # Input dimension of patches
            model_dim: int = 256,         # Model's internal dimension
            latent_dim: int = 16,         # Latent action dimension
            num_patches: int = 256,       # Number of patches for distributed actions
            enc_blocks: int = 2,          # Number of encoder transformer blocks
            dec_blocks: int = 2,          # Number of decoder transformer blocks
            num_heads: int = 8,           # Number of attention heads
            beta: float = 0.01,           # KL divergence weight
            dropout: float = 0.2,         # Dropout probability
            global_actions: bool = True  # If True, use global action tokens; if False, use distributed actions
    ) -> None:
        """
        Initialize the Latent Action Model.
        
        Args:
            in_dim: Input dimension of each patch
            model_dim: Internal representation dimension
            latent_dim: Dimension of the latent action space
            num_patches: Number of patches for distributed action representation
            enc_blocks: Number of transformer blocks in the encoder
            dec_blocks: Number of transformer blocks in the decoder
            num_heads: Number of attention heads in transformer blocks
            beta: Weight for KL divergence term in the loss
            dropout: Dropout probability
            global_actions: If True, use global action tokens; if False, use distributed actions
        """
        super(LatentActionModel, self).__init__()
        
        # Store configuration parameters
        self.model_dim = model_dim
        self.latent_dim = latent_dim
        self.num_patches = num_patches
        self.beta = beta
        self.global_actions = global_actions
        
        # Initialize action prompt based on implementation mode
        if self.global_actions:
            # Global actions: one action token per timestep
            self.action_prompt = nn.Parameter(torch.empty(1, 1, in_dim))
        else:
            # Distributed actions: multiple action tokens per timestep
            self.action_prompt = nn.Parameter(torch.empty(1, 1, self.num_patches, in_dim))
        
        # Initialize action prompt with uniform distribution
        nn.init.uniform_(self.action_prompt, a=-1, b=1)
        
        # Spatio-temporal transformer encoder
        self.encoder = SpatioTemporalTransformer(
            in_dim=in_dim,            # Input dimension
            model_dim=model_dim,      # Model dimension
            out_dim=model_dim,        # Output dimension
            num_blocks=enc_blocks,    # Number of transformer blocks
            num_heads=num_heads,      # Number of attention heads
            dropout=dropout           # Dropout probability
        )
        
        # VAE projection layer for mean and log variance
        self.fc = nn.Linear(model_dim, latent_dim * 2)
        
        # Input and action projection layers
        self.input_up = nn.Linear(in_dim, model_dim)       # Project input tokens to model dimension
        self.action_up = nn.Linear(latent_dim, model_dim)  # Project latent actions to model dimension
        
        # Spatial transformer decoder
        self.decoder = SpatioTransformer(
            in_dim=model_dim,         # Input dimension
            model_dim=model_dim,      # Model dimension
            out_dim=in_dim,           # Output dimension (same as original input)
            num_blocks=dec_blocks,    # Number of transformer blocks
            num_heads=num_heads,      # Number of attention heads
            dropout=dropout           # Dropout probability
        )
        
        # Storage for latent action records during inference
        self.mu_record = None

    def encode(self, tokens: Tensor) -> Dict:
        """
        Encode input tokens into latent action space.
        
        Args:
            tokens: Input tokens of shape (B, T, WxH, in_dim)
                   B: batch size, T: time steps, WxH: spatial patches
                   
        Returns:
            Dictionary containing:
                - tokens: Original input tokens
                - z_rep: Latent action representation
                - z_mu: Mean of the latent distribution
                - z_var: Log variance of the latent distribution
        """
        # Extract batch size and time steps
        B, T = tokens.shape[:2]
        
        if self.global_actions:
            # === Global Actions Mode ===
            
            # Expand action prompt to batch size and time steps
            action_pad = self.action_prompt.expand(B, T, -1)
            
            # Add action prompt as new token at position 0
            # Shape: (B, T, 1+WxH, in_dim)
            padded_tokens = torch.cat([action_pad.unsqueeze(2), tokens], dim=2)
            
            # Encode with spatio-temporal transformer
            # Shape: (B, T, 1+WxH, model_dim)
            z = self.encoder(padded_tokens)
            
            # Extract encoded action tokens for future frames
            # Shape: (B, T-1, model_dim)
            z = z[:, 1:, 0]
            
            # Reshape for VAE projection
            # Shape: (B*(T-1), model_dim)
            z = z.reshape(B * (T - 1), self.model_dim)
            
            # Project to get mean and log variance
            moments = self.fc(z)
            z_mu, z_var = torch.chunk(moments, 2, dim=1)
            
            # Reparameterization trick
            if not self.training:
                z_rep = z_mu
            else:
                z_rep = z_mu + torch.randn_like(z_var) * torch.exp(0.5 * z_var)
            
            # Reshape back to include time dimension and single action token
            # Shape: (B, T-1, 1, latent_dim)
            z_rep = z_rep.reshape(B, T - 1, 1, self.latent_dim)
            
        else:
            # === Distributed Actions Mode ===
            
            # Expand action prompt to batch size and time steps
            action_pad = self.action_prompt.expand(B, T, -1, -1)
            
            # Add action prompt tokens at the beginning
            # Shape: (B, T, num_patches+WxH, in_dim)
            padded_tokens = torch.cat([action_pad, tokens], dim=2)
            
            # Encode with spatio-temporal transformer
            # Shape: (B, T, num_patches+WxH, model_dim)
            z = self.encoder(padded_tokens)
            
            # Extract encoded action tokens for future frames
            # Shape: (B, T-1, num_patches, model_dim)
            z = z[:, 1:, :self.num_patches]
            
            # Reshape for VAE projection
            # Shape: (B*(T-1)*num_patches, model_dim)
            z = z.reshape(B * (T - 1) * self.num_patches, self.model_dim)
            
            # Project to get mean and log variance
            moments = self.fc(z)
            z_mu, z_var = torch.chunk(moments, 2, dim=1)
            
            # Reparameterization trick
            if not self.training:
                z_rep = z_mu
            else:
                z_rep = z_mu + torch.randn_like(z_var) * torch.exp(0.5 * z_var)
            
            # Reshape back to include time dimension and action patches
            # Shape: (B, T-1, num_patches, latent_dim)
            z_rep = z_rep.reshape(B, T - 1, self.num_patches, self.latent_dim)
        
        # Store latent action means during inference for later analysis
        if not self.training:
            if self.mu_record is None:
                self.mu_record = z_mu
            else:
                self.mu_record = torch.cat([self.mu_record, z_mu], dim=0)
        
        return {
            "tokens": tokens,
            "z_rep": z_rep,
            "z_mu": z_mu,
            "z_var": z_var
        }

    def forward(self, batch: Dict) -> Dict:
        """
        Forward pass through the Latent Action Model.
        
        Args:
            batch: Dictionary containing:
                - tokens: Input tokens of shape (B, T, WxH, in_dim)
                
        Returns:
            Dictionary containing original encode outputs plus:
                - recon: Reconstructed tokens
                - kl_loss: KL divergence loss
                - mse_loss: Mean squared error loss
                - loss: Combined loss
        """
        # Encode input tokens to latent action space
        outputs = self.encode(batch["tokens"].squeeze(0))
        
        # Project input tokens to model dimension
        # Shape: (B, T-1, WxH, model_dim)
        video_tokens = self.input_up(outputs["tokens"][:, :-1])
        
        # Project latent actions to model dimension
        # Shape: (B, T-1, num_patches, model_dim) or (B, T-1, 1, model_dim)
        action_tokens = self.action_up(outputs["z_rep"])
        
        # Combine video tokens and action tokens
        # Shape: (B, T-1, WxH, model_dim)
        video_action_tokens = video_tokens + action_tokens
        
        # Decode to reconstruct future frames
        # Shape: (B, T-1, WxH, in_dim)
        token_recon = self.decoder(video_action_tokens)
        
        # Ground truth future frames
        # Shape: (B, T-1, WxH, in_dim)
        gt_future_frames = outputs["tokens"][:, 1:]
        
        # Calculate losses
        # MSE reconstruction loss
        mse_loss = ((gt_future_frames - token_recon) ** 2).mean()
        
        # KL divergence loss
        kl_loss = -0.5 * torch.sum(1 + outputs["z_var"] - outputs["z_mu"] ** 2 - outputs["z_var"].exp(), dim=1).mean()
        
        # Combined loss with beta weighting for KL term
        loss = mse_loss + self.beta * kl_loss

        # Update outputs dictionary with losses and reconstruction
        outputs.update({
            "recon": token_recon,
            "kl_loss": kl_loss,
            "mse_loss": mse_loss,
            "loss": loss
        })
        
        return outputs


if __name__ == "__main__":
    """
    Example usage of the Latent Action Model in both modes.
    """
    # Example usage with global actions
    model_global = LatentActionModel(
        in_dim=8,             # Input dimension of patches
        model_dim=512,        # Model's internal dimension
        latent_dim=32,        # Latent action dimension
        enc_blocks=6,         # Number of encoder transformer blocks
        dec_blocks=6,         # Number of decoder transformer blocks
        num_heads=8,          # Number of attention heads
        dropout=0.1,          # Dropout probability
        global_actions=True   # Use global action tokens
    )
    
    # Example usage with distributed actions
    model_distributed = LatentActionModel(
        in_dim=8,             # Input dimension of patches
        model_dim=256,        # Model's internal dimension
        latent_dim=16,        # Latent action dimension
        num_patches=256,      # Number of patches for distributed actions
        enc_blocks=2,         # Number of encoder transformer blocks
        dec_blocks=2,         # Number of decoder transformer blocks
        num_heads=8,          # Number of attention heads
        beta=0.01,            # KL divergence weight
        dropout=0.2,          # Dropout probability
        global_actions=False  # Use distributed action tokens (default)
    )
    
    # Clean up CUDA memory
    torch.cuda.empty_cache()
    
    # Create sample input tokens
    # Shape: (batch_size, timesteps, patches, feature_dim)
    tokens = torch.randn(3, 20, 256, 8).cuda()
    
    print("======= Testing Global Actions Model =======")
    model_global = model_global.cuda()
    batch = {"tokens": tokens}
    outputs_global = model_global(batch)
    
    print("Output keys:", outputs_global.keys())
    print(f"Reconstruction shape: {outputs_global['recon'].shape}")
    print(f"Loss: {outputs_global['loss']}")
    print(f"MSE Loss: {outputs_global['mse_loss']}")
    print(f"KL Loss: {outputs_global['kl_loss']}")
    
    print("\n======= Testing Distributed Actions Model =======")
    model_distributed = model_distributed.cuda()
    outputs_distributed = model_distributed(batch)
    
    print("Output keys:", outputs_distributed.keys())
    print(f"Reconstruction shape: {outputs_distributed['recon'].shape}")
    print(f"Loss: {outputs_distributed['loss']}")
    print(f"MSE Loss: {outputs_distributed['mse_loss']}")
    print(f"KL Loss: {outputs_distributed['kl_loss']}")