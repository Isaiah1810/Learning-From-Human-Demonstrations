import numpy as np
import torch 
import sys
sys.path.append("./src/modules")
from genie.action import LatentAction
from PIL import Image
from accelerate import Accelerator
import h5py
import os

ENC_BLUEPRINT = (
    ('space-time_attn', {
        'n_rep' : 2,
        'n_embd' : 256,
        'n_head' : 4,
        'd_head' : 8,
    }),
    ('spacetime_downsample', {
        'in_channels' : 256,
        'kernel_size' : 3,
        'time_factor' : 1,
        'space_factor' : 2,
    }),
    ('space-time_attn', {
        'n_rep' : 2,
        'n_embd' : 256,
        'n_head' : 4,
        'd_head' : 8,
    }),
)

DEC_BLUEPRINT = (
    ('space-time_attn', {
        'n_rep' : 2,
        'n_embd' : 256,
        'n_head' : 4,
        'd_head' : 16,
        'has_ext' : True,
        'time_attn_kw'  : {'key_dim' : 8},
    }),
    ('spacetime_upsample', {
        'in_channels' : 256,
        'kernel_size' : 3,
        'time_factor' : 1,
        'space_factor' : 2,
    }),
    ('space-time_attn', {
        'n_rep' : 2,
        'n_embd' : 256,
        'n_head' : 4,
        'd_head' : 16,
        'has_ext' : True,
        'time_attn_kw'  : {'key_dim' : 8},
    }),
)

def load_vqgan_model():
    """Load the VQGAN model."""
    from OmniTokenizer import OmniTokenizer_VQGAN
    return OmniTokenizer_VQGAN.load_from_checkpoint("./imagenet_k600.ckpt", strict=False, weights_only=False)





if __name__ == '__main__':

    model = LatentAction(
        enc_desc = ENC_BLUEPRINT,
        dec_desc = DEC_BLUEPRINT,
        d_codebook=8,      
        inp_channels=1,     
        inp_shape=(32, 32), 
        n_embd=256, 
    )

    checkpoint_path = ''

    model.load_state_dict(torch.load(checkpoint_path))

    data_dir = ''

    with h5py.File(data_dir) as f:
        data = f['tokens']
        data = torch.Tensor(np.array(data))

    print("data original type", data.dtype)
    data_dtype = data.dtype
    data = data.to(dtype=torch.float32).permute(1,0,2,3)

    vqgan = load_vqgan_model()

    # Normalize the data, save the max
    data_max = data.max()
    data = data / data_max

    print("data input shape:", data.shape)

    (act, idxs, enc_video) = model.encode(data)

    print("encoded video shape:", enc_video.shape)

    recons = model.decode(enc_video, act)

    print("reconstructed latent video shape:", recons.shape)

    recons *= data_max
    recons.to(dtype=data_dtype)

    vid = vqgan.decode(recons, True)

    print("reconstructed video shape:", vid.shape)

    img_rec = (((recons.detach().numpy() + 1)/2)*255).astype(np.uint8)[19].transpose(1, 2, 0)
    img = Image.fromarray(img_rec)

    img.save("./reconstructed_image.png")

    