import os
import sys
import numpy as np
import torch
import h5py
import yaml
import imageio
import cv2

from PIL import Image
from src.latent_action import LatentActionModel

# Add local module paths
sys.path.append("./src/modules")
sys.path.append('./src')


def load_vqgan_model():
    """Load the pretrained VQGAN model."""
    from src.modules.OmniTokenizer import OmniTokenizer_VQGAN
    return OmniTokenizer_VQGAN.load_from_checkpoint(
        "./models/imagenet_k600.ckpt", 
        strict=False, 
        weights_only=False
    )


class Normalizer:
    """Normalizer for scaling token data between [-1, 1]."""

    def normalize(self, data):
        data_min = data.min(dim=2, keepdims=True)[0]
        data_max = data.max(dim=2, keepdims=True)[0]
        data = (data - data_min) / (data_max - data_min + 1e-9)
        data = 2 * data - 1
        return data, data_min, data_max

    def denormalize(self, data, min_val, max_val):
        data = 0.5 * (data + 1)
        return data * (max_val[1:] - min_val[1:]) + min_val[1:]


def animate_trajectories(
    orig_trajectory, pred_trajectory, path='./traj_anim.gif', duration=4 / 50, 
    rec_to_pred_t=10, title=None
):
    """Generate a side-by-side GIF animation comparing original and predicted trajectories."""
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.4
    color = (255, 255, 255)
    border_thickness = 1
    border_size = 2
    origin = (5, 15)
    gt_border_color = (255, 0, 0)
    rec_border_color = (0, 0, 255)
    gen_border_color = (0, 255, 0)

    gt_frames, pred_frames = [], []

    for i in range(orig_trajectory.shape[0]):
        # Ground Truth Frame
        gt_img = (orig_trajectory[i] * 255).astype(np.uint8)
        gt_img = cv2.putText(gt_img, f'GT:{i}', origin, font, font_scale, color, border_thickness, cv2.LINE_AA)
        gt_img = cv2.copyMakeBorder(gt_img, border_size, border_size, border_size, border_size, cv2.BORDER_CONSTANT, value=gt_border_color)
        gt_frames.append(gt_img)

        # Predicted Frame
        pred_img = (pred_trajectory[i].clip(0, 1) * 255).astype(np.uint8)
        label = f'REC:{i}' if i < rec_to_pred_t else f'PRED:{i}'
        border_color = rec_border_color if i < rec_to_pred_t else gen_border_color
        pred_img = cv2.putText(pred_img, label, origin, font, font_scale, color, border_thickness, cv2.LINE_AA)
        pred_img = cv2.copyMakeBorder(pred_img, border_size, border_size, border_size, border_size, cv2.BORDER_CONSTANT, value=border_color)
        pred_frames.append(pred_img)

    # Combine Frames and Export
    combined_frames = []
    for gt, pred in zip(gt_frames, pred_frames):
        separator = np.ones((gt.shape[0], 4, gt.shape[2]), dtype=np.uint8) * 255
        combined = np.concatenate([gt, separator, pred], axis=1)

        if title:
            h, w = 25, combined.shape[1]
            header = np.ones((h, w, 3), dtype=np.uint8) * 255
            text_origin = (w // 6, h // 2)
            header = cv2.putText(header, title, text_origin, font, 0.25, (0, 0, 0), 1, cv2.LINE_AA)
            combined = np.concatenate([header, combined], axis=0)

        combined_frames.append(combined)

    imageio.mimsave(path, combined_frames, duration=duration, loop=0)


if __name__ == '__main__':
    # Load configuration
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)

    # Load LatentActionModel
    model = LatentActionModel(
        in_dim=config['model']['in_dim']['value'],
        model_dim=config['model']['model_dim']['value'],
        latent_dim=config['model']['latent_dim']['value'],
        enc_blocks=config['model']['enc_blocks']['value'],
        dec_blocks=config['model']['dec_blocks']['value'],
        num_heads=config['model']['num_heads']['value'],
        dropout=config['model']['dropout']['value'],
        global_patch=config['model']['global_patch']['value']
    )

    # Load model checkpoint
    checkpoint_path = './models/global_7dim_64_latent.pth'
    state_dict = torch.load(checkpoint_path, map_location="cuda:0")
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model = model.cuda().eval()

    # Load VQGAN model
    vqgan = load_vqgan_model()
    vqgan.eval()

    # Load and preprocess images
    data_dir = 'Images/730/'
    output_path = config['recreation']['output_path']['value']
    orig_images = []
    imgs = []

    for path in os.listdir(data_dir)[::3]:
        img = Image.open(os.path.join(data_dir, path)).resize((128, 128))
        tensor_img = torch.tensor(np.array(img).transpose(2, 1, 0).reshape((1, 3, 128, 128)), dtype=torch.float32)
        orig_images.append(tensor_img / 255)
        imgs.append(2 * (tensor_img / 255) - 1)

    # Prepare input batch
    inp = torch.cat(imgs, dim=0)[:20].cuda()

    # VQGAN Encoding
    data, _ = vqgan.encode(inp, return_embedding=True, return_tokens=True)
    orig_embeddings = data
    original_shape = data.shape

    # Reshape and normalize
    data = data.reshape(data.shape[0], data.shape[1], -1).permute(0, 2, 1)
    norm = Normalizer()
    data, min_val, max_val = norm.normalize(data)
    data = data.unsqueeze(0).cuda()
    min_val, max_val = min_val.cuda(), max_val.cuda()

    # Latent Action Model Inference
    outputs = model({'tokens': data})
    print("Encoded actions:", outputs['z_rep'].shape)
    recons = outputs['recon']
    recons_norm = norm.denormalize(recons, min_val, max_val).squeeze(0).permute(0, 2, 1)

    # Reshape to original
    reshaped = recons_norm.reshape(original_shape[0] - 1, original_shape[1], original_shape[2], original_shape[3])
    encodings = vqgan.codebook.embeddings_to_encodings(reshaped)

    # Decode
    decoded = vqgan.decode(encodings, return_embedding=True) * 2
    decoded = decoded.detach().cpu().numpy()

    # Convert to displayable images
    recons_images = [((frame + 1) / 2).transpose(2, 1, 0) for frame in decoded]
    orig_images_np = np.concatenate(orig_images[1:20]).transpose(0, 3, 2, 1)

    # Animate
    animate_trajectories(orig_images_np, np.array(recons_images), duration=3, path=output_path)
