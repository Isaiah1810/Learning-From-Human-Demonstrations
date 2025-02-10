import torch
import numpy as np
import os
import cv2
import h5py
from torch.utils.data import DataLoader
import json
import multiprocessing as mp

global size

def load_dino_model():
    """Load the DINOv2 model."""
    import dinov2
    BACKBONE_SIZE = "small"
    backbone_archs = {
        "small": "vits14",
        "base": "vitb14",
        "large": "vitl14",
        "giant": "vitg14",
    }
    backbone_arch = backbone_archs[BACKBONE_SIZE]
    model = torch.hub.load("facebookresearch/dinov2", f"dinov2_{backbone_arch}")
    model.eval().cuda()
    return model

def preprocess_frame(frame: np.ndarray) -> torch.Tensor:
    '''Process frame for DINO input'''
    frame = cv2.resize(frame, (size, size))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    inp = torch.tensor(frame.transpose(2, 0, 1), dtype=torch.float32).unsqueeze(0).to('cuda')
    return inp

def process_video_dino(video_path, output_folder, model, batch_size=16):
    '''Convert video into latent dino output'''
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(preprocess_frame(frame))
    
    cap.release()

    # Process in batches for efficiency
    frame_batches = [torch.cat(frames[i:i+batch_size], dim=0) for i in range(0, len(frames), batch_size)]

    latents = [model(batch).cpu().detach().numpy() for batch in frame_batches]
    
    #np.save(f"{output_folder}.npy", np.concatenate(latents, axis=0))
    return np.concatenate(latents, axis=0)

def dino_tokenizer(path: str, output_path: str):
    """Extract frames and convert to DINOv2 embeddings."""
    model = load_dino_model()
    with h5py.File(os.path.joint(output_path, 'something-somethingv2-dino.hdf5'), 'w') as f:
        for i, file in enumerate(os.listdir(path)):
            file_no = file[:-5]  # Remove file extension
            os.makedirs(output_path, exist_ok=True)
            input_path = os.path.join(path, file)
            latents = process_video_dino(input_path, output_path, model)
            f.create_dataset(file_no, data=latents, compression='gzip')


def load_vqgan_model():
    from OmniTokenizer import OmniTokenizer_VQGAN
    """Load the VQGAN model."""
    vqgan_ckpt = "./imagenet_ucf_vae.ckpt"
    model = OmniTokenizer_VQGAN.load_from_checkpoint(vqgan_ckpt, strict=False, weights_only=False)
    return model

def preprocess_frame_vqgan(frame: np.ndarray, size: int) -> torch.Tensor:
    """Resize and normalize the frame for VQGAN processing."""
    size = int(size)
    frame = cv2.resize(frame, (size, size))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    inp = torch.tensor(frame.transpose(2, 0, 1), dtype=torch.float32).unsqueeze(0).to('cuda')
    return inp

def encode_frames(model, frames):
    """Encode a batch of frames using VQGAN."""
    frames = frames.to('cuda')
    print(frames.shape)
    return model.encode(frames, False).cpu().detach().numpy()

def process_video_vqgan(video_path, output_folder, model, size):
    """Process a video into VQGAN embeddings using multiprocessing."""
    cap = cv2.VideoCapture(video_path)
    frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(preprocess_frame_vqgan(frame, size))


    # The OmniTokenizer VQGan requires vidoes to have temporal dimension 
    # divisble by 4 plus an extra frame 
    num_frames = len(frames)
    frames_to_divisble_by_four = 4 - (num_frames % 4)
    frames.extend([frames[-1] for i in range(frames_to_divisble_by_four + 1)])

    cap.release()
    
    latents = encode_frames(model, torch.stack(frames, axis=2))
    
    return latents

def vqgan_worker(task_queue, size, f):
    """Worker function for multiprocessing."""
    model = load_vqgan_model().to('cuda')
    
    while True:
        video_path, output_folder, file_no = task_queue.get()
        if video_path is None:
            break
        latents = process_video_vqgan(video_path, output_folder, model, size)
        # Maybe need to add MP Lock here
        f.create_dataset(file_no, data=latents, compression='gzip')


def vqgan_tokenizer(path: str, output_path: str, size: int):
    """Multiprocessing-based tokenizer for VQGAN."""
    num_workers = min(mp.cpu_count(), 4)
    task_queue = mp.Queue()

    with h5py.File(os.path.join(output_path, 'something-somethingv2-vqgan.hdf5'), 'w') as f:
        
        # Start worker processes
        # processes = []
        # for _ in range(num_workers):
        #     p = mp.Process(target=vqgan_worker, args=(task_queue, size, f))
        #     p.start()
        #     processes.append(p)
        
        # Assign tasks
        for file in os.listdir(path):
            file_no = file[:-5]
            os.makedirs(output_path, exist_ok=True)
            
            input_path = os.path.join(path, file)
            model = load_vqgan_model().to('cuda')
            latents = process_video_vqgan(input_path, output_path, model, size)
            f.create_dataset(file_no, data=latents, compression='gzip')
            #task_queue.put((input_path, output_path, file_no))


        # Stop workers
        # for _ in range(num_workers):
        #     task_queue.put((None, None))
        
        # task_queue.close()
        # task_queue.join_thread()

        # for p in processes:
        #     p.join()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        prog='Data Tokenizer',
        description='Converts dataset at given directory to latent representations',
        epilog=''
    )
    
    parser.add_argument('path')
    parser.add_argument('output_path')
    parser.add_argument('tokenizer')
    parser.add_argument('size')

    args = parser.parse_args()


    if not os.path.isdir(args.path):
        print("Directory does not exist")
        exit(1)

    if args.size is None:
        print("Invalid image size parameter")

    size = int(args.size)

    match args.tokenizer:
        case 'dinov2':
            if size % 14 != 0:
                print("Size must be multiple of 14 for dinov2 (due to patch size)")
                exit(1)
            dino_tokenizer(args.path, os.path.join(args.output_path, 'dinov2'))
        
        case 'vqgan':
            if size % 8 != 0:
                print("Size for VQGan has to be divisble by 8")
                exit(1)
            vqgan_tokenizer(args.path, os.path.join(args.output_path, 'vqgan'), int(size))
    
        case _:
            print("Invalid tokenizer type")
            exit(1)

    