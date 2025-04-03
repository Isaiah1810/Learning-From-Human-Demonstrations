import sys
sys.path.append("./src/modules")
sys.path.append("./")
import os
from typing import Dict


import torch
import torch.nn as nn
import torch.nn.functional as F
from lam.modules.blocks import patchify, unpatchify, SpatioTemporalTransformer, SpatioTransformer
from torch import Tensor

from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from lam.modules.blocks import SpatioTemporalTransformer, SpatioTransformer
from torch import Tensor

class LatentActionModel(nn.Module):

    def __init__(
            self,
            in_dim: int,  
            model_dim: int,
            latent_dim: int,
            enc_blocks: int,
            dec_blocks: int,
            num_heads: int,
            beta: float = 0.01,
            dropout: float = 0.0
    ) -> None:
        super(LatentActionModel, self).__init__()
        self.model_dim = model_dim
        self.latent_dim = latent_dim
        self.beta = beta
        self.action_prompt = nn.Parameter(torch.empty(1, 1, in_dim))
        nn.init.uniform_(self.action_prompt, a=-1, b=1)
        
        self.encoder = SpatioTemporalTransformer(
            in_dim=in_dim,  
            model_dim=model_dim,
            out_dim=model_dim,
            num_blocks=enc_blocks,
            num_heads=num_heads,
            dropout=dropout
        )
        
        self.fc = nn.Linear(model_dim, latent_dim * 2)
        
        # Input projection to model dimension
        self.input_up = nn.Linear(in_dim, model_dim)
        self.action_up = nn.Linear(latent_dim, model_dim)
        
        self.decoder = SpatioTransformer(
            in_dim=model_dim,
            model_dim=model_dim,
            out_dim=in_dim,  # Back to original input dimension
            num_blocks=dec_blocks,
            num_heads=num_heads,
            dropout=dropout
        )
        
        self.mu_record = None

    def encode(self, tokens: Tensor) -> Dict:
        # Tokens shape: (B, T, WxH, in_dim)
        B, T = tokens.shape[:2]
        
        # Add action prompt
        action_pad = self.action_prompt.expand(B, T, -1)
        padded_tokens = torch.cat([action_pad.unsqueeze(2), tokens], dim=2)
        
        print("paadded", padded_tokens.shape)

        # Encode
        z = self.encoder(padded_tokens)  # (B, T, 1+WxH, E)
        
        z = z[:, 1:, 0]  # (B, T-1, E)
        
        # VAE
        z = z.reshape(B * (T - 1), self.model_dim)
        moments = self.fc(z)
        z_mu, z_var = torch.chunk(moments, 2, dim=1)
        
        # Reparameterization
        if not self.training:
            z_rep = z_mu
        else:
            z_rep = z_mu + torch.randn_like(z_var) * torch.exp(0.5 * z_var)
        
        z_rep = z_rep.reshape(B, T - 1, 1, self.latent_dim)
        
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
        # Encode and Latent Action VAE
        outputs = self.encode(batch["tokens"])
        
        # Project input tokens and latent representation
        video_tokens = self.input_up(outputs["tokens"][:, :-1])
        action_tokens = self.action_up(outputs["z_rep"])
        video_action_tokens = video_tokens + action_tokens
        #video_action_tokens = action_tokens
        #del outputs["tokens"]
        
        # Decode
        token_recon = self.decoder(video_action_tokens)
        
        gt_future_frames = outputs["tokens"][:, 1:]

        mse_loss = ((gt_future_frames - token_recon) ** 2).mean()
        kl_loss = -0.5 * torch.sum(1 + outputs["z_var"] - outputs["z_mu"] ** 2 - outputs["z_var"].exp(), dim=1).mean()
        loss = mse_loss + self.beta * kl_loss

        outputs.update(
            {
                "recon": token_recon,
                "kl_loss": kl_loss,
                "mse_loss": mse_loss,
                "loss": loss
            }
        )
        
        return outputs

if __name__ == "__main__":

    model = LatentActionModel(
    in_dim=8,  # Patch dimension     
    model_dim=512,             
    latent_dim=32,                           
    enc_blocks=6,                             
    dec_blocks=6,                            
    num_heads=8,                     
    dropout=0.1        
                        
    )
    
    # dict = model(tokens)

    torch.cuda.empty_cache()

    tokens = torch.randn(3, 20, 256, 8).cuda() # batch, time, sequence, dimension
    model = model.cuda()
    batch = {"tokens": tokens}
    outputs = model(batch)

    print(outputs.keys())

    print(outputs['recon'].shape)
    print(outputs['loss'])