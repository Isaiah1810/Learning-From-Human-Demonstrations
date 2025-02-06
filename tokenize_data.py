
import torch
import numpy as np
import argparse
import sys
import os
import threading
import cv2
from queue import Queue

global size



def dino_tokenizer(path: str, output_path: str, verbose: bool) -> None:
    '''
    Converts webm video files a listed path to frames compressed with dinov2 



    '''
    import dinov2
    BACKBONE_SIZE = 'large'
    backbone_archs = {
        "small": "vits14",
        "base": "vitb14",
        "large": "vitl14",
        "giant": "vitg14",
    }
    backbone_arch = backbone_archs[BACKBONE_SIZE]
    backbone_name = f"dinov2_{backbone_arch}"

    model = torch.hub.load(repo_or_dir="facebookresearch/dinov2", model=backbone_name)

    model.eval()
    model.cuda()


    def encode_frame(model, frame: np.ndarray) -> np.ndarray:
        '''
        Takes in an image and a dinov2 model, resizes image, and outputs the 
        latent representation of the image 

        model:
            dinov2 model
        frame: 
            numpy array of shape (w, h, c)

        OUTPUT:
            latent: numpy array of shape (batch_size, 1024)
        '''
        frame = cv2.resize(frame, (size, size))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        inp = torch.tensor(np.array(frame).transpose(2, 1, 0).reshape((1, 3, size, size)), dtype=torch.float32).to('cuda')
        latent = model(inp)
        print(latent.shape)
        return latent


    def save_frames(frame_queue: Queue, output_path: str, model) -> None:
            while True:
                frame_id, frame = frame_queue.get()
                if frame is None:
                    break
                tokens = encode_frame(model, frame).cpu().detach().numpy()
                print(os.path.join(output_path, f'{frame_id}.npy'))
                np.save(os.path.join(output_path, f'{frame_id}.npy'),tokens)
                frame_queue.task_done()

    def preprocess_video(path, output_folder, model):
        cap = cv2.VideoCapture(path)
        frame_queue = Queue()
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        num_threads = 4  # Adjust based on your CPU
        threads = []
        for _ in range(num_threads):
            t = threading.Thread(target=save_frames, args=(frame_queue, output_folder, model), daemon=True)
            t.start()
            threads.append(t)

        # Read and enqueue frames
        frame_id = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_queue.put((frame_id, frame))
            frame_id += 1

        cap.release()
        frame_queue.join()

        # Stop worker threads
        for _ in range(num_threads):
            frame_queue.put((None, None))
        for t in threads:
            t.join()

        print(f"Extracted {frame_id} frames to {output_folder}")

    for file in os.listdir(path):
        file_no = file[:-5] # Extract number/remove file extension
        output_dir = os.path.join(output_path, file_no)
        os.makedirs(output_dir, exist_ok=True)
        input_dir = os.path.join(path, file)
        preprocess_video(input_dir, output_dir, model)
    


def vqgan_tokenizer(path, output_path, verbose):
    from OmniTokenizer import OmniTokenizer_VQGAN
    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='Data Tokenizer',
        description='Converts dataset at given directory to latent representations',
        epilog=''
    )
    
    parser.add_argument('path')
    parser.add_argument('output_path')
    parser.add_argument('tokenizer')
    parser.add_argument('size')
    parser.add_argument('-v', '--verbose', action='store_true')

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
            dino_tokenizer(args.path, args.output_path, 
                           args.verbose)
        
        case 'vqgan':
            vqgan_tokenizer(args.path, args.output_path, 
                            args.verbose)
    
        case _:
            print("Invalid tokenizer type")
            exit(1)

    