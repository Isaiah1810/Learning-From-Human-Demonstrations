
import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore")
    from .src.modules.OmniTokenizer import OmniTokenizer_VQGAN
    import numpy as np
    import torch
    import sys
    sys.path.append("./src/modules")
    sys.path.append('./src')
    from PIL import Image
    from .src.latent_action import LatentActionModel
    import os
    import yaml



class SequenceTokenizer(torch.nn.Module):
    def __init__(self, vqgan_path=None, latent_action_path=None, config_path='config.yaml'):
        super().__init__()
        
        if vqgan_path is None and latent_action_path is None:
            self.initialized = False
            return
            
        self.initialized = True
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")

            # Load VQGAN
            self.vqgan = OmniTokenizer_VQGAN.load_from_checkpoint(vqgan_path, strict=False, weights_only=False)
            self.vqgan.eval()
            
            # Load config
            with open(config_path, "r") as file:
                self.config = yaml.safe_load(file)
                
            # Load latent action model
            self.latent_action = LatentActionModel(
                in_dim=self.config['model']['in_dim']['value'],    
                model_dim=self.config['model']['model_dim']['value'],             
                latent_dim=self.config['model']['latent_dim']['value'],                           
                enc_blocks=self.config['model']['enc_blocks']['value'],                             
                dec_blocks=self.config['model']['dec_blocks']['value'],                            
                num_heads=self.config['model']['num_heads']['value'],                     
                dropout=self.config['model']['dropout']['value']
            )

            # Load state dict
            state_dict = torch.load(latent_action_path, map_location="cuda:0")
            new_state_dict = {key.replace("module.", ""): value for key, value in state_dict.items()}
            self.latent_action.load_state_dict(new_state_dict)
            
            self.add_module('vqgan', self.vqgan)
            self.add_module('latent_action', self.latent_action)

    def forward(self, sequence, latent_actions=True, reconstructions=False):
        return self.encode(sequence, latent_actions, reconstructions)

    def encode(self, sequence, latent_actions=True, reconstructions=False):
        with torch.no_grad():
            # Ensure sequence is on the right device
            if not isinstance(sequence, torch.Tensor):
                sequence = torch.tensor(sequence)
            
            device = next(self.parameters()).device 
            sequence = sequence.to(device)
            
            gt_embeddings, gt_encodings = self.vqgan.encode(sequence, True, True)

            if not latent_actions and not reconstructions:
                return gt_embeddings

            gt_shape = gt_embeddings.shape

            gt_embeddings = gt_embeddings.reshape(gt_embeddings.shape[0], gt_embeddings.shape[1], -1).permute(0, 2, 1)

            data = gt_embeddings

            data, min_val, max_val = self._normalize(data)
    
            data = data.unsqueeze(0).to(self.device)

            outputs = self.latent_action({'tokens': data})

            actions = outputs['z_rep'].squeeze(2)

            if not reconstructions:
                return gt_embeddings, actions

            recons = outputs['recon']


            recons_norm = self._denormalize(recons, min_val, max_val)

            if recons_norm.dim() == 4: 
                recons_norm = recons_norm.squeeze(0)

            recons_norm = recons_norm.permute(0, 2, 1)

            new_shape = np.array(gt_shape)
            new_shape[0] -= 1
            new_shape = tuple(new_shape)
            recons_norm = recons_norm.reshape(new_shape)
            
            encodings = self.vqgan.codebook.embeddings_to_encodings(recons_norm)

            recons_vids = self.vqgan.decode(encodings, True)
        
            recons_vids *= 2

            if not latent_actions:
                return gt_embeddings, recons_vids

            return gt_embeddings, actions, recons_vids


    def extract_actions(self, gt_embeddings):

        with torch.no_grad():

            data = gt_embeddings

            data, min_val, max_val = self._normalize(data)
    
            data = data.unsqueeze(0).to(self.device)

            outputs = self.latent_action({'tokens': data})

            actions = outputs['z_rep'].squeeze(2)

            return actions

    def _normalize(self, data):
        data_min = data.min(dim=(2), keepdims=True)[0]
        data_max = data.max(dim=(2), keepdims=True)[0]
        data.sub_(data_min).div_(data_max - data_min + 1e-9)
        data.mul_(2).sub_(1)
        
        return data, data_min, data_max
    
    def _denormalize(self, data, min_val, max_val):
        denorm = 0.5*(data + 1)
        denorm = denorm * (max_val[1:] - min_val[1:]) + min_val[1:]
        return denorm