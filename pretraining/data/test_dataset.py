import torch
import os
from dataset import VideoDataset
import sys
sys.path.append("/scratch/iew/Learning-From-Human-Demonstrations/")

data_dir = '/scratch/iew/sthv2/tokens/vqgan/train'
embed_model_path = './imagenet_k600.ckpt'
la_path = './latent_action.pth'
la_config_path = './config.yaml'


dataset = VideoDataset('./pretraining/configs/dataset_config.yaml')

item = dataset[0]

print("V Shape:", item[0].shape)
print("S Shape:", item[1].shape)
print("A Shape:", item[2].shape)
print(item[3])
print(item[4])

