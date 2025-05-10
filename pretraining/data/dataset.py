import torch 
from torch.utils.data import Dataset
import os
import numpy as np
import h5py
from PIL import Image
import yaml


class VideoDataset(Dataset):
    def __init__(self, data_dir, sequence_len=20, 
                 is_embedded=True, skip_max=3, width=16, height=16, input_dim=8, 
                 action_dim=7):
        
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
            print("Requires Embedded Data")
            raise NotImplementedError

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

        V = V.reshape((self.sequence_len, self.height, self.width, self.input_dim))

        S = S.reshape((self.sequence_len, self.height, self.width, self.input_dim))

        padding_mask_V = torch.zeros(self.sequence_len, dtype=torch.bool)
        padding_mask_SA = torch.zeros(self.sequence_len, dtype=torch.bool)
        padding_mask_V[:self.sequence_len - V_pad_len] = True
        padding_mask_SA[:self.sequence_len - S_pad_len] = True

        V = V.detach().cpu()
        S = S.detach().cpu()

        return V, S, torch.tensor(0), padding_mask_V, padding_mask_SA


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
