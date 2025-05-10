import torch
import numpy as np
import os
import cv2
from PIL import Image
import h5py
import json
import multiprocessing as mp
from torchvision.transforms import ToTensor
import sys
sys.path.append("./src/modules")


def load_vqgan_model():
    """Load the VQGAN model."""
    from latent_action_model.src.modules.OmniTokenizer import OmniTokenizer_VQGAN
    return OmniTokenizer_VQGAN.load_from_checkpoint("./imagenet_k600.ckpt", strict=False, weights_only=False)

def preprocess_frame(path, size):
    """Preprocess a single frame for tokenization."""
    from PIL import Image
    from torchvision.transforms import ToTensor
    img = Image.open(path)
    img = img.resize((size, size))
    img = ToTensor()(img)
    img = img * 2 - 1
    return img

def tokenize_video(input_path, size, model, device, ep_len=100, sample_len=20):
    import torch
    import os
    from torchvision.transforms import ToTensor
    from PIL import Image


    paths = sorted([os.path.join(input_path, file) for file in os.listdir(input_path) 
                   if file.lower().endswith(('.png', '.jpg', '.jpeg'))])

    seqs_per_episode = ep_len - sample_len + 1

    # Pad To Meet Episode Length
    while len(paths) < ep_len:
            paths.append(paths[-1])

    frames = []
    for file_path in paths:
        frames.append(preprocess_frame(file_path, size))


    seq_tokens = []
    
    for i in range(seqs_per_episode):
        seq_frames = torch.stack(frames[i:i+sample_len], dim=0).to(device)
        tokens = model.encode(seq_frames, True, True)[0]
        tokens = tokens.reshape(tokens.shape[0], tokens.shape[1], tokens.shape[3]*tokens.shape[4]).permute(0,2,1)
        seq_tokens.append(tokens)

    return seq_tokens


def vqgan_tokenizer(input_path: str, output_path: str, size: int, device: str, ep_len: int, sample_len: int):
    """Tokenize videos using VQGAN."""
    import os
    import h5py
    import torch
    from tqdm import tqdm
    
    data_dir = os.path.join(input_path, 'frames')

    os.makedirs(output_path, exist_ok=True)
    os.makedirs(os.path.join(output_path, 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_path, 'validation'), exist_ok=True)
    os.makedirs(os.path.join(output_path, 'test'), exist_ok=True)

    model = load_vqgan_model().to(device)
    model.eval() 

    # Get Training Split Episodes
    train_json = os.path.join(input_path, 'labels/train.json')
    train_dirs = []
    with open(train_json, 'r') as f:
                    json_file = json.load(f)
                    for item in json_file:
                        if os.path.exists(os.path.join(data_dir, item['id'])):
                            train_dirs.append(os.path.join(data_dir, item['id']))

    # Get Validation Split Episodes
    val_json = os.path.join(input_path, 'labels/validation.json')
    val_dirs = []
    with open(val_json, 'r') as f:
                    json_file = json.load(f)
                    for item in json_file:
                        if os.path.exists(os.path.join(data_dir, item['id'])):
                            val_dirs.append(os.path.join(data_dir, item['id']))


    # Get Test Split Episodes
    test_json = os.path.join(input_path, 'labels/test.json')
    test_dirs = []
    with open(test_json, 'r') as f:
                    json_file = json.load(f)
                    for item in json_file:
                        if os.path.exists(os.path.join(data_dir, item['id'])):
                            test_dirs.append(os.path.join(data_dir, item['id']))

    train_dirs = [os.path.join(input_path, d) for d in train_dirs if os.path.isdir(os.path.join(input_path, d))]
    val_dirs = [os.path.join(input_path, d) for d in val_dirs if os.path.isdir(os.path.join(input_path, d))]
    test_dirs = [os.path.join(input_path, d) for d in test_dirs if os.path.isdir(os.path.join(input_path, d))]

    with torch.no_grad():
        seq = 0
        for dir_path in tqdm(train_dirs, desc="Tokenizing Training Data"):
            tokens = tokenize_video(dir_path, size, model, device,  ep_len, sample_len)    
            for i in range(len(tokens)):
                with h5py.File(os.path.join(output_path, 'train', f'{seq}.h5'), 'w') as f:
                      seq += 1
                      f.create_dataset('tokens', data=tokens[i].cpu().numpy(), 
                                       compression='gzip', compression_opts=9)


        seq = 0
        for dir_path in tqdm(val_dirs, desc="Tokenizing Validation Data"):
            tokens = tokenize_video(dir_path, size, model, device, ep_len, sample_len)    
            for i in range(len(tokens)):
                with h5py.File(os.path.join(output_path, 'validation', f'{seq}.h5'), 'w') as f:
                      seq += 1
                      f.create_dataset('tokens', data=tokens[i].cpu().numpy(), 
                                       compression='gzip', compression_opts=9)

        seq = 0
        for dir_path in tqdm(test_dirs, desc="Tokenizing Test Data"):
            tokens = tokenize_video(dir_path, size, model, device, ep_len, sample_len)    
            for i in range(len(tokens)):
                with h5py.File(os.path.join(output_path, 'test', f'{seq}.h5'), 'w') as f:
                      seq += 1
                      f.create_dataset('tokens', data=tokens[i].cpu().numpy(), 
                                       compression='gzip', compression_opts=9)



if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert videos to latent representations")
    parser.add_argument("path", help="Path to dataset directory")
    parser.add_argument("output_path", help="Path to save output")
    parser.add_argument("size", type=int, help="Pre-embedded image size")
    
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    if not os.path.isdir(args.path):
        print("Error: Directory does not exist")
        exit(1)

    elif args.tokenizer == "vqgan" and args.size % 8 != 0:
        print("Error: Size must be a multiple of 8 for VQGAN")
        exit(1)
    
    if args.tokenizer == "dinov2":
        raise NotImplementedError
    else:
        vqgan_tokenizer(args.path, os.path.join(args.output_path, "vqgan"), args.size, device, ep_len=100, sample_len=20)
