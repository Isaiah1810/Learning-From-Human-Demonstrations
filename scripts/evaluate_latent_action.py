import piqa 
import numpy as np
import torch
import sys
sys.path.append("./src/modules")
sys.path.append('./src')
from PIL import Image
from accelerate import Accelerator
from latent_action import LatentActionModel
import os
import json

def load_vqgan_model():
    """Load the VQGAN model."""
    from OmniTokenizer import OmniTokenizer_VQGAN
    return OmniTokenizer_VQGAN.load_from_checkpoint("./models/imagenet_k600.ckpt", strict=False, weights_only=False)
  


def process_images(data_dir, vqgan, model):
    orig_images = []
    imgs = []
    for path in os.listdir(data_dir)[:20]:
        img = Image.open(os.path.join(data_dir, path))
        img = img.resize((128, 128))
        inp = torch.tensor(np.array(img).transpose(2, 1, 0).reshape((1, 3, 128, 128)), dtype=torch.float32)
        orig_images.append(inp / 255)
        inp = 2 * (inp / 255) - 1
        imgs.append(inp)

    inp = torch.concatenate(imgs, dim=0)
    inp = inp.cuda()

    data, orig_encodings = vqgan.encode(inp, True, True)

    original_shape = data.shape

    data = data.reshape(data.shape[0], data.shape[1], -1).permute(0, 2, 1)

    
    data, min_val, max_val = model.normalizer.normalize(data)
    
    min_val, max_val = min_val.cuda(), max_val.cuda()
    
    model = model.cuda()
    data = data.unsqueeze(0).cuda()
   
    outputs = model({'tokens': data})


    recons = outputs['recon']
    
    recons_norm = model.normalizer.denormalize(recons, min_val, max_val)

    
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
    
    recons_vids = vqgan.decode(encodings, True)
    
    recons_vids *= 2

    recons_vids = recons_vids.detach().cpu().numpy()

    images = []
    for i in range(recons_vids.shape[0]):
        img_rec = (((recons_vids[i] + 1)/2)).transpose(2, 1, 0)
        images.append(img_rec)
    return images, orig_images

def evaluate_images(images, orig_images):
    tv = piqa.TV()
    tv_val = tv.forward(images - orig_images)

    psnr = piqa.PSNR()
    psnr_val = psnr.forward(images, orig_images)

    ssim = piqa.SSIM()
    ssim_val = ssim.forward(images, orig_images)

    lpips = piqa.LPIPS()
    lpips_val = lpips.forward(images, orig_images)

    gmsd = piqa.GMSD()
    gmsd_val = gmsd.forward(images, orig_images)

    msdi = piqa.MDSI()
    msdi_val = msdi.forward(images, orig_images)

    haarPSI = piqa.HaarPSI()
    haarPSI_val = haarPSI.forward(images, orig_images)

    vsi = piqa.VSI()
    vsi_val = vsi.forward(images, orig_images)

    fsim = piqa.FSIM()
    fsim_val = fsim.forward(images, orig_images)

    fid = piqa.FID()
    fid_val = fid.forward(images, orig_images)

    metrics = {
        'tv': tv_val,
        'psnr': psnr_val,
        'ssim': ssim_val,
        'lpips': lpips_val,
        'gmsd': gmsd_val,
        'msdi': msdi_val,
        'haarPSI': haarPSI_val,
        'vsi': vsi_val,
        'fsim': fsim_val,
        'fid': fid_val
        }
    return metrics

if __name__ == '__main__':
    vqgan = load_vqgan_model()
    vqgan.eval()
    import yaml

    model = LatentActionModel(
        in_dim=8,            
        model_dim=128,              
        latent_dim=7,                        
        enc_blocks=2,                          
        dec_blocks=2,                         
        num_heads=8,                      
        dropout=0.2                                      
    )
    model.eval()

    input_path = '/scratch/iew/sthv2/frames'
    data_dir = '/scratch/iew/sthv2/frames/frames'
    test_json = os.path.join(input_path, 'labels/test.json')
    test_dirs = []
    with open(test_json, 'r') as f:
                    json_file = json.load(f)
                    for item in json_file:
                        if os.path.exists(os.path.join(data_dir, item['id'])):
                            test_dirs.append(os.path.join(data_dir, item['id']))
    dicts = [] 
    for file in test_dirs:
        images, orig_images = process_images(file, vqgan, model)
        metrics = evaluate_images(images, orig_images)
        dicts.append(metrics)
    
    avg_dict = {k: sum(d[k] for d in dicts) / len(dicts) for k in dicts[0]}

    with open('metrics.yaml', 'w') as f:
        yaml.dump(avg_dict, f)

