import torch 
from torch.utils.data import Dataset
from sequence_tokenizer import SequenceTokenizer
import os
import numpy as np
import h5py

class VideoDataset(Dataset):
    def __init__(self, data_dir, embed_model_path, la_path, la_config_path, 
                 sequence_len=20, is_embedded=True, frame_skip=True, skip_max=3, 
                 width=16, height=16, input_dim=8, action_dim=7):
        
        self.tokenizer = SequenceTokenizer(embed_model_path, la_path, la_config_path)
        self.sequence_len = sequence_len
        self.is_embedded = is_embedded
        self.skip_max = skip_max

        self.width = width
        self.height = height
        self.input_dim = input_dim
        self.action_dim = action_dim

        self.files = [os.path.join(data_dir, fname) for fname in os.listdir(data_dir) if fname.endswith('.h5')] 

    def __getlen__(self):
        return len(self.files)
    
    def __getitem__(self, idx):

        skip_V = np.random.randint(0, self.max_skip + 1) if self.training else 0
        skip_S = np.random.randint(0, self.max_skip + 1) if self.training else 0


        file_path = self.files[idx]
    
        if self.is_embedded:

            with h5py.File(file_path, 'r') as f:
                data = f['tokens']
                data = np.array(data)
                data = torch.Tensor(data)
                norm_data, _, _ = self.normalizer.normalize(data)

        else:
            # Embed Sequence
            raise NotImplementedError

        V = norm_data[::skip_V+1]

        S = norm_data[::skip_S+1]

        V_len = V.shape[0]
        S_len = S.shape[0]

        if V_len >= self.sequence_len:
            V = V[:self.sequence_len]

        else:
            V_pad_len = self.sequence_len - V_len
            padding = torch.zeros(V_pad_len, self.width * self.height, self.input_dim)
            V = torch.concatenate((V, padding), dim=0)
        
        if S_len >= self.sequence_len:
            S = S[:self.sequence_len]

        else:
            S_pad_len = self.sequence_len - S_len
            padding = torch.zeros(S_pad_len, self.width * self.height, self.input_dim)
            S = torch.concatenate((S, padding), dim=0)

        A = self.tokenizer.extract_actions(S)

        V = V.reshape((V.shape[0], self.height, self.width, self.input_dim))

        S = S.reshape((S.shape[0], self.height, self.width, self.input_dim))

        padding_mask_V = torch.zeros(self.sequence_len, dtype=torch.bool)
        padding_mask_SA = torch.zeros(self.sequence_len, dtype=torch.bool)
        padding_mask_V[:V_pad_len] = True
        padding_mask_SA[:S_pad_len] = True

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