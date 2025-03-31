import numpy as np
import torch
import sys
sys.path.append("./src/modules")
sys.path.append('./src')
from PIL import Image
from accelerate import Accelerator
from latent_action_multipatch import LatentActionModel
import h5py
import os

def load_vqgan_model():
    """Load the VQGAN model."""
    from OmniTokenizer import OmniTokenizer_VQGAN
    return OmniTokenizer_VQGAN.load_from_checkpoint("./models/imagenet_k600.ckpt", strict=False, weights_only=False)
  
class Normalizer():
    def __init__(self):
        pass
    
    def normalize(self, data):
        data_min = data.min(dim=(2), keepdims=True)[0]
        data_max = data.max(dim=(2), keepdims=True)[0]
        data.sub_(data_min).div_(data_max - data_min + 1e-9)
        data.mul_(2).sub_(1)
        
        return data, data_min, data_max
    
    def denormalize(self, data, min_val, max_val):
        denorm = 0.5*(data + 1)
        denorm = denorm * (max_val[1:] - min_val[1:]) + min_val[1:]
        return denorm
    
if __name__ == '__main__':
    model = LatentActionModel(
        in_dim=8,        # Patch dimension      
        model_dim=64,              
        latent_dim=16,                        
        enc_blocks=2,                          
        dec_blocks=2,                         
        num_heads=8,                      
        dropout=0.2                                      
    )
    
    checkpoint_path = './models/checkpoint_epoch_1_multipatch.pth'
    state_dict = torch.load(checkpoint_path, map_location="cuda:0")
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace("module.", "")  # Remove 'module.' prefix
        new_state_dict[new_key] = value
    
    model.load_state_dict(new_state_dict)
    
    vqgan = load_vqgan_model()
    data_dir = 'Images/730/'
    
    from PIL import Image

   # print(os.listdir(data_dir))

    imgs = []

    for path in os.listdir(data_dir):
        img = Image.open(os.path.join(data_dir, path))
        img = img.resize((128, 128))
        inp = torch.tensor(np.array(img).transpose(2, 1, 0).reshape((1, 3, 128, 128)), dtype=torch.float32)
        inp = 2 * (inp / 255) - 1
        imgs.append(inp)

    inp = torch.concatenate(imgs, dim=0)[30:50]
   # print(inp.shape)
    inp = inp.cuda()
    vqgan.eval()
    model.eval()
    
    data, orig_encodings = vqgan.encode(inp, True, True)

    orig_embeddings = data
    
    # Save original shape for restoration later
    original_shape = orig_embeddings.shape

    # Reshape for processing - preserving the shape information
    data = data.reshape(data.shape[0], data.shape[1], -1).permute(0, 2, 1)

    norm = Normalizer()
    
    data, min_val, max_val = norm.normalize(data)
    
    min_val, max_val = min_val.cuda(), max_val.cuda()
    
    model = model.cuda()
    data = data.unsqueeze(0).cuda()
   
    outputs = model({'tokens': data})
    
    print("encoded actions:", outputs['z_rep'].shape)
    
    recons = outputs['recon']
    
    recons_norm = norm.denormalize(recons, min_val, max_val)

    
    # Reshape back to original structure
    # First, make sure recons_norm has the same shape as data before processing
    if recons_norm.dim() == 4:  # If it has an extra dimension from model output
        recons_norm = recons_norm.squeeze(0)
    
    # Reverse the permutation to get back to the original order
    recons_norm = recons_norm.permute(0, 2, 1)
    
    # Reshape back to the original spatial dimensions
    new_shape = np.array(original_shape)
    new_shape[0] -= 1
    new_shape = tuple(new_shape)
    recons_norm = recons_norm.reshape(new_shape)
    
    encodings = vqgan.codebook.embeddings_to_encodings(recons_norm)
    
    # Uncomment for visualization
    recons_vids = vqgan.decode(encodings, True)
    
    recons_vids *= 2

    recons_vids = recons_vids.detach().cpu().numpy()

    images = []

    for i in range(recons_vids.shape[0]):
        img_rec = (((recons_vids[i] + 1)/2)*255).astype(np.uint8).transpose(2, 1, 0)
        img_rec = Image.fromarray(img_rec)
        images.append(img_rec)
        # img_rec.save(f"reconstructions/{i+1}.jpg")

    print(len(images))

    images[0].save('reconstruction.gif', save_all=True, append_images=images[1:], duration=120, loop=0)
    