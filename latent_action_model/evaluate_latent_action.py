import os
import sys
import json
import yaml
import torch
import numpy as np
import piqa
from PIL import Image

# Add source paths
sys.path.append("./src/modules")
sys.path.append("./src")

from src.latent_action import LatentActionModel

def load_vqgan_model(ckpt_path):
    """Load the VQGAN model from checkpoint."""
    from src.modules.OmniTokenizer import OmniTokenizer_VQGAN
    return OmniTokenizer_VQGAN.load_from_checkpoint(
        ckpt_path, strict=False, weights_only=False
    )

def process_images(data_dir, vqgan, model):
    orig_images, imgs = [], []

    for path in os.listdir(data_dir)[:20]:
        img = Image.open(os.path.join(data_dir, path)).resize((128, 128))
        inp = torch.tensor(np.array(img).transpose(2, 1, 0).reshape((1, 3, 128, 128)), dtype=torch.float32)
        orig_images.append(inp / 255)
        inp = 2 * (inp / 255) - 1
        imgs.append(inp)

    inp = torch.cat(imgs, dim=0).cuda()
    data, _ = vqgan.encode(inp, True, True)
    original_shape = data.shape

    data = data.reshape(data.shape[0], data.shape[1], -1).permute(0, 2, 1)
    data, min_val, max_val = model.normalizer.normalize(data)
    data = data.unsqueeze(0).cuda()
    min_val, max_val = min_val.cuda(), max_val.cuda()

    outputs = model({'tokens': data})
    recons = model.normalizer.denormalize(outputs['recon'], min_val, max_val)

    if recons.dim() == 4:
        recons = recons.squeeze(0)
    recons = recons.permute(0, 2, 1).reshape(original_shape)

    encodings = vqgan.codebook.embeddings_to_encodings(recons)
    recons_vids = vqgan.decode(encodings, True) * 2
    recons_vids = recons_vids.detach().cpu().numpy()

    images = [(((recons_vids[i] + 1) / 2)).transpose(2, 1, 0) for i in range(recons_vids.shape[0])]
    return images, orig_images

def evaluate_images(images, orig_images):
    metrics = {}
    metrics['tv'] = piqa.TV()(images - orig_images).item()
    metrics['psnr'] = piqa.PSNR()(images, orig_images).item()
    metrics['ssim'] = piqa.SSIM()(images, orig_images).item()
    metrics['lpips'] = piqa.LPIPS()(images, orig_images).item()
    metrics['gmsd'] = piqa.GMSD()(images, orig_images).item()
    metrics['msdi'] = piqa.MDSI()(images, orig_images).item()
    metrics['haarPSI'] = piqa.HaarPSI()(images, orig_images).item()
    metrics['vsi'] = piqa.VSI()(images, orig_images).item()
    metrics['fsim'] = piqa.FSIM()(images, orig_images).item()
    metrics['fid'] = piqa.FID()(images, orig_images).item()
    return metrics

def main():
    # Load config
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Initialize VQGAN
    vqgan = load_vqgan_model(config['vqgan_ckpt_path'])
    vqgan.eval()

    # Initialize LatentActionModel
    model_cfg = config['model']
    model = LatentActionModel(
        in_dim=model_cfg['in_dim'],
        model_dim=model_cfg['model_dim'],
        latent_dim=model_cfg['latent_dim'],
        enc_blocks=model_cfg['enc_blocks'],
        dec_blocks=model_cfg['dec_blocks'],
        num_heads=model_cfg['num_heads'],
        dropout=model_cfg['dropout']
    )
    model.eval()

    input_path = config['input_path']
    data_dir = config['data_dir']
    test_json = os.path.join(input_path, 'labels/test.json')

    with open(test_json, 'r') as f:
        json_file = json.load(f)
        test_dirs = [
            os.path.join(data_dir, item['id'])
            for item in json_file
            if os.path.exists(os.path.join(data_dir, item['id']))
        ]

    all_metrics = []
    for folder in test_dirs:
        images, orig_images = process_images(folder, vqgan, model)
        metrics = evaluate_images(images, orig_images)
        all_metrics.append(metrics)

    avg_metrics = {k: sum(m[k] for m in all_metrics) / len(all_metrics) for k in all_metrics[0]}

    with open("metrics.yaml", "w") as f:
        yaml.dump(avg_metrics, f)

if __name__ == "__main__":
    main()
