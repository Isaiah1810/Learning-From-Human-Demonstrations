import numpy as np
import torch
import sys
sys.path.append("./src/modules")
sys.path.append('./src')
from PIL import Image
from accelerate import Accelerator
from latent_action import LatentActionModel
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
    

import cv2
import imageio

def animate_trajectories(orig_trajectory, pred_trajectory, path='./traj_anim.gif', duration=4 / 50, rec_to_pred_t=10,
                         title=None):
    # rec_to_pred_t: the timestep from which prediction transitions from reconstruction to generation
    # prepare images
    font = cv2.FONT_HERSHEY_SIMPLEX
    origin = (5, 15)
    fontScale = 0.4
    color = (255, 255, 255)
    gt_border_color = (255, 0, 0)
    rec_border_color = (0, 0, 255)
    gen_border_color = (0, 255, 0)
    border_size = 2
    thickness = 1
    gt_traj_prep = []
    pred_traj_prep = []
    for i in range(orig_trajectory.shape[0]):
        image = (orig_trajectory[i] * 255).astype(np.uint8).copy()
        image = cv2.putText(image, f'GT:{i}', origin, font, fontScale, color, thickness, cv2.LINE_AA)
        # add border
        image = cv2.copyMakeBorder(image, border_size, border_size, border_size, border_size, cv2.BORDER_CONSTANT,
                                   value=gt_border_color)
        gt_traj_prep.append(image)

        text = f'REC:{i}' if i < rec_to_pred_t else f'PRED:{i}'
        image = (pred_trajectory[i].clip(0, 1) * 255).astype(np.uint8).copy()
        image = cv2.putText(image, text, origin, font, fontScale, color, thickness, cv2.LINE_AA)
        # add border
        border_color = rec_border_color if i < rec_to_pred_t else gen_border_color
        image = cv2.copyMakeBorder(image, border_size, border_size, border_size, border_size, cv2.BORDER_CONSTANT,
                                   value=border_color)
        pred_traj_prep.append(image)

    total_images = []
    for i in range(len(orig_trajectory)):
        white_border = (np.ones((gt_traj_prep[i].shape[0], 4, gt_traj_prep[i].shape[-1])) * 255).astype(np.uint8)
        concat_img = np.concatenate([gt_traj_prep[i],
                                     white_border,
                                     pred_traj_prep[i]], axis=1)
        if title is not None:
            text_color = (0, 0, 0)
            fontScale = 0.25
            thickness = 1
            font = cv2.FONT_HERSHEY_SIMPLEX
            h = 25
            w = concat_img.shape[1]
            text_plate = (np.ones((h, w, 3)) * 255).astype(np.uint8)
            w_orig = orig_trajectory.shape[1] // 2
            origin = (w_orig // 6, h // 2)
            text_plate = cv2.putText(text_plate, title, origin, font, fontScale, text_color, thickness,
                                     cv2.LINE_AA)
            concat_img = np.concatenate([text_plate, concat_img], axis=0)
        # total_images.append((concat_img * 255).astype(np.uint8))
        total_images.append(concat_img)
    imageio.mimsave(path, total_images, duration=duration, loop=0)  # 1/50



if __name__ == '__main__':
    model = LatentActionModel(
        in_dim=8,        # Patch dimension      
        model_dim=64,              
        latent_dim=7,                        
        enc_blocks=2,                          
        dec_blocks=2,                         
        num_heads=8,                      
        dropout=0.2                                      
    )
    
    checkpoint_path = './models/global_7dim_64_latent.pth'
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

    orig_images = []

    for path in os.listdir(data_dir)[::3]:
        img = Image.open(os.path.join(data_dir, path))
        img = img.resize((128, 128))
        inp = torch.tensor(np.array(img).transpose(2, 1, 0).reshape((1, 3, 128, 128)), dtype=torch.float32)
        orig_images.append(inp / 255)
        inp = 2 * (inp / 255) - 1
        imgs.append(inp)

    inp = torch.concatenate(imgs, dim=0)[:20]
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
        img_rec = (((recons_vids[i] + 1)/2)).transpose(2, 1, 0)
        images.append(img_rec)
       # img_rec.save(f"reconstructions/{i+1}.jpg")

    orig_images = np.concatenate(orig_images[1:20]).transpose((0,3,2,1))
    images = np.array(images)

    print(orig_images.shape, images.shape)


    animate_trajectories(orig_images, images, duration=3)

