import torch
import numpy as np
import os
import cv2
from PIL import Image
import h5py
import json
import multiprocessing as mp
from torchvision.transforms import ToTensor


def load_vqgan_model():
    """Load the VQGAN model."""
    from OmniTokenizer import OmniTokenizer_VQGAN
    return OmniTokenizer_VQGAN.load_from_checkpoint("./imagenet_k600.ckpt", strict=False, weights_only=False)

def preprocess_frame(path, size):
    """Preprocess a single frame for tokenization."""
    from PIL import Image
    from torchvision.transforms import ToTensor
    img = Image.open(path)
    img = img.resize((size, size))
    img = ToTensor()(img)
    img = img * 2 - 1
    return img.unsqueeze(0)

def tokenize_video(input_path, size, model, device):
    """Tokenize all frames in a video directory."""
    import torch
    import os
    from torchvision.transforms import ToTensor
    from PIL import Image
    
    frames = []
    files = sorted([os.path.join(input_path, file) for file in os.listdir(input_path) 
                   if file.lower().endswith(('.png', '.jpg', '.jpeg'))])
    print(input_path, os.listdir(input_path))
    all_tokens = []
        
    for file_path in files:
        img = Image.open(file_path)
        img = img.resize((size, size))
        img = ToTensor()(img)
        frames.append(img)
        
    if frames:
            remainder = (len(frames) - 1) % 4
            if remainder != 0:
                frames.extend([frames[-1]] * (4 - remainder))
            frame_tensor = torch.stack(frames, dim=2).reshape((1, 3, -1, size, size)).to(device)
            tokens = model.encode(frame_tensor, False)
            all_tokens.append(tokens)
    
    if all_tokens:
        return torch.cat(all_tokens, dim=0)
    return torch.tensor([])

def vqgan_tokenizer(input_path: str, output_path: str, size: int, device: str):
    """Tokenize videos using VQGAN."""
    import os
    import h5py
    import torch
    from tqdm import tqdm
    
    os.makedirs(output_path, exist_ok=True)
    model = load_vqgan_model().to(device)
    model.eval() 
    
    if all(os.path.isdir(os.path.join(input_path, d)) for d in os.listdir(input_path) if not d.startswith('.')):
        dirs = [os.path.join(input_path, d) for d in os.listdir(input_path) if os.path.isdir(os.path.join(input_path, d))]
    else:
        dirs = [input_path]

    with torch.no_grad():  
        for dir_path in tqdm(dirs, desc="Processing directories"):
            dir_name = os.path.basename(dir_path)
            output_file = os.path.join(output_path, f"{dir_name}.h5")            
            tokens = tokenize_video(dir_path, size, model, device)
            print(tokens.shape)
            if len(tokens) > 0:
                with h5py.File(output_file, "w") as f:
                    f.create_dataset('tokens', data=tokens.cpu().numpy(), 
                                    compression="gzip", compression_opts=9)
                print(f"Processed {dir_name}: {tokens.shape}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert videos to latent representations")
    parser.add_argument("path", help="Path to dataset directory")
    parser.add_argument("output_path", help="Path to save output")
    parser.add_argument("tokenizer", choices=["dinov2", "vqgan"], help="Tokenizer type")
    parser.add_argument("size", type=int, help="Image size")
    
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    if not os.path.isdir(args.path):
        print("Error: Directory does not exist")
        exit(1)
    
    if args.tokenizer == "dinov2" and args.size % 14 != 0:
        print("Error: Size must be a multiple of 14 for DINOv2")
        exit(1)
    elif args.tokenizer == "vqgan" and args.size % 8 != 0:
        print("Error: Size must be a multiple of 8 for VQGAN")
        exit(1)
    
    if args.tokenizer == "dinov2":
        raise NotImplementedError
    else:
        vqgan_tokenizer(args.path, os.path.join(args.output_path, "vqgan"), args.size, device)
