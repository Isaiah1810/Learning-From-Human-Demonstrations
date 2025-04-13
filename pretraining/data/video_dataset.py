import torch
from torch.utils.data import Dataset

# placeholder dataset
class DummyVideoActionDataset(Dataset):
    def __init__(self, size=1000, input_dim=128, action_dim=8, frames=8, height=16, width=16):
        self.size = size
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.frames = frames
        self.height = height
        self.width = width

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        valid_len_V = torch.randint(1, self.frames + 1, (1,)).item()
        valid_len_SA = torch.randint(1, self.frames + 1, (1,)).item()

        V = torch.zeros(self.frames, self.height, self.width, self.input_dim)
        S = torch.zeros(self.frames, self.height, self.width, self.input_dim)
        A = torch.zeros(self.frames, self.height, self.width, self.action_dim)

        V[:valid_len_V] = torch.randn(valid_len_V, self.height, self.width, self.input_dim)
        S[:valid_len_SA] = torch.randn(valid_len_SA, self.height, self.width, self.input_dim)
        A[:valid_len_SA] = torch.randn(valid_len_SA, self.height, self.width, self.action_dim)

        padding_mask_V = torch.zeros(self.frames, dtype=torch.bool)
        padding_mask_SA = torch.zeros(self.frames, dtype=torch.bool)
        padding_mask_V[:valid_len_V] = True
        padding_mask_SA[:valid_len_SA] = True

        return V, S, A, padding_mask_V, padding_mask_SA
