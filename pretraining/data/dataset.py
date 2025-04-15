import torch 
from torch.utils.data import Dataset
from sequence_tokenizer import SequenceTokenizer
import os
import numpy as np
import h5py
from PIL import Image
import yaml

class VideoDataset(Dataset):
    def __init__(self, config_path, tokenizer=None, sequence_len=20, 
                 is_embedded=True, skip_max=3, width=16, height=16, input_dim=8, 
                 action_dim=7):
        
        with open(config_path, "r") as file:
                self.config = yaml.safe_load(file)

        data_dir = self.config['dataset']['data_dir']
        embed_model_path = self.config['dataset']['embed_model_path']
        la_path = self.config['dataset']['la_path']
        
        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = SequenceTokenizer(embed_model_path, la_path, config_path)
            print("WARNING: Creating New Tokenizer Instance in Dataset is slow for distributed training")

        self.sequence_len = sequence_len
        self.is_embedded = is_embedded
        self.skip_max = skip_max

        self.width = width
        self.height = height
        self.input_dim = input_dim
        self.action_dim = action_dim

        self.files = [os.path.join(data_dir, fname) for fname in os.listdir(data_dir) if fname.endswith('.h5')] 

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):

        skip_V = np.random.randint(0, self.skip_max + 1)
        skip_S = np.random.randint(0, self.skip_max + 1)


        file_path = self.files[idx]
    
        if self.is_embedded:

            with h5py.File(file_path, 'r') as f:
                data = f['tokens']
                data = np.array(data)
                data = torch.Tensor(data)
                norm_data, _, _ = self._normalize(data)

        else:
            imgs = []
            num_frames = min(len(os.listdir(file_path)), self.sequence_len)
            for file in os.listdir(file_path)[:num_frames]:
                img = Image.open(os.path.join(file_path, file))
                img = img.resize((128, 128))
                inp = torch.tensor(np.array(img).transpose(2, 1, 0).reshape((1, 3, 128, 128)), dtype=torch.float32)
                inp = 2 * (inp / 255) - 1
                imgs.append(inp)

            sequence = torch.concatenate(imgs, dim=0)
            data = self.tokenizer.encode(sequence, False, False)
            norm_data, _, _ = self._normalize(data)

        V = norm_data[::skip_V+1]
        S = norm_data[::skip_S+1]

        V_len = V.shape[0]
        S_len = S.shape[0]

        V_pad_len = 0
        S_pad_len = 0

        if V_len >= self.sequence_len:
            V = V[:self.sequence_len]

        else:
            V_pad_len = self.sequence_len - V_len
            padding = torch.zeros(V_pad_len, self.width * self.height, self.input_dim)
            V = torch.cat((V, padding), dim=0)
        
        if S_len >= self.sequence_len:
            S = S[:self.sequence_len]

        else:
            S_pad_len = self.sequence_len - S_len
            padding = torch.zeros(S_pad_len, self.width * self.height, self.input_dim)
            S = torch.cat((S, padding), dim=0)

        A = self.tokenizer.extract_actions(S)
        A = A.reshape((self.sequence_len-1, 1, 1, self.action_dim))

        V = V.reshape((self.sequence_len, self.height, self.width, self.input_dim))

        S = S.reshape((self.sequence_len, self.height, self.width, self.input_dim))

        action_padding = torch.zeros((1, 1, 1, self.action_dim)).to(self.tokenizer.device)
        A = torch.cat((action_padding, A), dim=0)
        A = A.expand(self.sequence_len, self.height, self.width, self.action_dim)

        padding_mask_V = torch.zeros(self.sequence_len, dtype=torch.bool)
        padding_mask_SA = torch.zeros(self.sequence_len, dtype=torch.bool)
        padding_mask_V[:self.sequence_len - V_pad_len] = True
        padding_mask_SA[:self.sequence_len - S_pad_len] = True

        V = V.detach().cpu()
        S = S.detach().cpu()
        A = A.detach().requires_grad_(False).cpu()

        return V, S, A, padding_mask_V, padding_mask_SA


    def _normalize(self, data):
        data_min = data.min(dim=(2), keepdims=True)[0]
        data_max = data.max(dim=(2), keepdims=True)[0]
        data.sub_(data_min).div_(data_max - data_min + 1e-9)
        data.mul_(2).sub_(1)
        
        return data, data_min, data_max
    
    def _denormalize(self, data, min_val, max_val):
        denorm = 0.5*(data + 1)
        denorm = denorm * (max_val[1:] - min_val[1:]) + min_val[1:]
        return denorm