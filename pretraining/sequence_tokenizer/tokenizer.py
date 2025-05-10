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
    def __init__(self, config, device=None):
        super().__init__()
        
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            vqgan_path = config['model']['image_embed_model_path']
            vqgan = OmniTokenizer_VQGAN.load_from_checkpoint(vqgan_path, strict=False, weights_only=False)
            if device is not None:
                self.vqgan = vqgan.to(device)
            else:
                self.vqgan = vqgan
            self.vqgan.eval()
            
            # Load latent action model
            la_config = config['model']['latent_action_params']

            latent_action = LatentActionModel(
                in_dim=la_config['in_dim'],    
                model_dim=la_config['model_dim'],             
                latent_dim=la_config['latent_dim'],                           
                enc_blocks=la_config['enc_blocks'],                             
                dec_blocks=la_config['dec_blocks'],                            
                num_heads=la_config['num_heads'],                     
                dropout=la_config['dropout']
            )

            if device is not None:
                self.latent_action = latent_action.to(device)
            else:
                self.latent_action = latent_action


            latent_action_path = config['model']['latent_action_path']
            state_dict = torch.load(latent_action_path)
            new_state_dict = {key.replace("module.module.", ""): value for key, value in state_dict.items()}
            self.latent_action.load_state_dict(new_state_dict)


            self.add_module('vqgan', self.vqgan)
            self.add_module('latent_action', self.latent_action)

    def forward(self, sequence, latent_actions=True, reconstructions=False):
        return self.encode(sequence, latent_actions, reconstructions)

    def encode(self, sequence, latent_actions=True, reconstructions=False, return_min_max=False):
        with torch.no_grad():
            # Make sure sequence is on the right device
            device = next(self.parameters()).device
            if not isinstance(sequence, torch.Tensor):
                sequence = torch.tensor(sequence, device=device)
            else:
                sequence = sequence.to(device)
            gt_embeddings, gt_encodings = self.vqgan.encode(sequence, True, True)

            if not latent_actions and not reconstructions:
                return gt_embeddings

            gt_shape = gt_embeddings.shape

            gt_embeddings = gt_embeddings.reshape(gt_embeddings.shape[0], gt_embeddings.shape[1], -1).permute(0, 2, 1)

            data = gt_embeddings

            data, min_val, max_val = self._normalize(data)
    
            data = data.unsqueeze(0).unsqueeze(0)

            outputs = self.latent_action({'tokens': data})

            actions = outputs['z_rep'].squeeze(2)

            if not reconstructions:
                if return_min_max:
                    return gt_embeddings, actions, min_val, max_val
                else:
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
                if return_min_max:
                    gt_embeddings, recons_vids, min_val, max_val
                else:
                    return gt_embeddings, recons_vids

            if return_min_max:

                return gt_embeddings, actions, recons_vids, min_val, max_val

            return gt_embeddings, actions, recons_vids


    def extract_actions(self, gt_embeddings):
        with torch.no_grad():
            
            data = gt_embeddings.reshape((gt_embeddings.shape[0], gt_embeddings.shape[1],
                                          gt_embeddings.shape[2] * gt_embeddings.shape[3], 
                                          gt_embeddings.shape[4]))


            data, min_val, max_val = self._normalize(data)
    
            data = data.unsqueeze(0)

            outputs = self.latent_action({'tokens': data})

            actions = outputs['z_rep'].squeeze(2)

            return actions

    def reconstruct(self, embeddings, actions, min_val, max_val, actions_only=False):
        # Get predicted frames from previous frames and actions
        device = next(self.latent_action.parameters()).device
        embeddings = embeddings.to(device)
        actions = actions.to(device)
        recons = self.latent_action.decode_actions(embeddings, actions, actions_only)
        r0 = recons
        recons = recons.reshape((recons.shape[0], recons.shape[1]*recons.shape[2], recons.shape[3]))

        # Get Pixel Space Images from reconstructed embeddings
        recons_norm = self._denormalize(recons, min_val.cpu(), max_val.cpu())
        recons_norm = recons_norm.permute(0, 2, 1)
    
        encodings = self.vqgan.codebook.embeddings_to_encodings(recons_norm).cpu()

        recons_vids = self.vqgan.decode(encodings, True)
    
        recons_vids *= 2        

        return r0, recons_vids

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
