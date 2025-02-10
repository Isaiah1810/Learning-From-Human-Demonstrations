import torch
import numpy as np
import os
import cv2
import h5py
import json
import multiprocessing as mp
from torch.utils.data import DataLoader


def load_dino_model():
    """Load the DINOv2 model."""
    import dinov2

    backbone_archs = {
        "small": "vits14",
        "base": "vitb14",
        "large": "vitl14",
        "giant": "vitg14",
    }
    model = torch.hub.load("facebookresearch/dinov2", f"dinov2_{backbone_archs['small']}")
    model.eval().cuda()
    return model


def preprocess_frame(frame: np.ndarray, size: int) -> torch.Tensor:
    """Preprocess a frame for DINO input."""
    frame = cv2.resize(frame, (size, size))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return torch.tensor(frame.transpose(2, 0, 1), dtype=torch.float32).unsqueeze(0).to("cuda")


def process_video_dino(video_path, model, size, batch_size=16):
    """Convert video into latent DINO output."""
    cap = cv2.VideoCapture(video_path)
    frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(preprocess_frame(frame, size))
    cap.release()

    frame_batches = [torch.cat(frames[i:i + batch_size], dim=0) for i in range(0, len(frames), batch_size)]
    latents = [model(batch).cpu().detach().numpy() for batch in frame_batches]
    return np.concatenate(latents, axis=0)


def dino_tokenizer(input_path: str, output_path: str, size: int):
    """Extract frames and convert to DINOv2 embeddings."""
    model = load_dino_model()
    os.makedirs(output_path, exist_ok=True)
    
    with h5py.File(os.path.join(output_path, "dino_embeddings.hdf5"), "w") as f:
        for file in os.listdir(input_path):
            file_no = file.rsplit(".", 1)[0]
            latents = process_video_dino(os.path.join(input_path, file), model, size)
            f.create_dataset(file_no, data=latents, compression="gzip")


def load_vqgan_model():
    """Load the VQGAN model."""
    from OmniTokenizer import OmniTokenizer_VQGAN
    model = OmniTokenizer_VQGAN.load_from_checkpoint("./imagenet_ucf_vae.ckpt", strict=False, weights_only=False)
    return model


def preprocess_frame_vqgan(frame: np.ndarray, size: int) -> torch.Tensor:
    """Resize and normalize the frame for VQGAN processing."""
    frame = cv2.resize(frame, (size, size))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return torch.tensor(frame.transpose(2, 0, 1), dtype=torch.float32).unsqueeze(0).to("cuda")


def encode_frames(model, frames):
    """Encode a batch of frames using VQGAN."""
    return model.encode(frames.to("cuda"), False).cpu().detach().numpy()


def process_video_vqgan(video_path, model, size):
    """Process a video into VQGAN embeddings."""
    cap = cv2.VideoCapture(video_path)
    frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(preprocess_frame_vqgan(frame, size))
    cap.release()

    num_frames = len(frames)
    frames.extend([frames[-1]] * (4 - num_frames % 4 + 1))  # Adjust frame count to be divisible by 4
    latents = encode_frames(model, torch.stack(frames, axis=2))
    return latents


def vqgan_tokenizer(input_path: str, output_path: str, size: int):
    """Tokenize videos using VQGAN."""
    os.makedirs(output_path, exist_ok=True)
    
    with h5py.File(os.path.join(output_path, "vqgan_embeddings.hdf5"), "w") as f:
        for file in os.listdir(input_path):
            file_no = file.rsplit(".", 1)[0]
            model = load_vqgan_model().to("cuda")
            latents = process_video_vqgan(os.path.join(input_path, file), model, size)
            f.create_dataset(file_no, data=latents, compression="gzip")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert videos to latent representations")
    parser.add_argument("path", help="Path to dataset directory")
    parser.add_argument("output_path", help="Path to save output")
    parser.add_argument("tokenizer", choices=["dinov2", "vqgan"], help="Tokenizer type")
    parser.add_argument("size", type=int, help="Image size")
    
    args = parser.parse_args()

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
        dino_tokenizer(args.path, os.path.join(args.output_path, "dinov2"), args.size)
    else:
        vqgan_tokenizer(args.path, os.path.join(args.output_path, "vqgan"), args.size)
