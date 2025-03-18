import numpy as np
import torch 
import sys
sys.path.append("./src/modules")
from genie.action import LatentAction
from accelerate import Accelerator
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import h5py
import os
import json

class TokenizedSthv2(Dataset):
    def __init__(self, data_dir, split_json=None, dataset_name='tokens'):
        self.data_dir = data_dir
        self.dataset_name = dataset_name

        if split_json == None:
            self.files = os.listdir(data_dir)
           # self.files = sorted(self.files, key=lambda x: int(x.split(".")[0]))
        
        else:
            self.files = []
            with open(split_json, 'r') as f:
                json_file = json.load(f)
                for item in json_file:
                    if os.path.exists(os.path.join(data_dir,f"{item['id']}.h5")):
                         self.files.append(f"{item['id']}.h5")
           # self.files = sorted(self.files, key=lambda x: int(x.split(".")[0]))

        self.h5_files = [os.path.join(data_dir, fname) for fname in self.files]

    def __len__(self):
        return len(self.h5_files)

    def __getitem__(self, idx):
        file_path = self.h5_files[idx]
        with h5py.File(file_path, 'r') as f:
            data = f['tokens']
            data = torch.Tensor(data)
        data = torch.tensor(data, dtype=torch.float32)
        return data


def collate_fn(batch):

    lengths = [item.shape[0] for item in batch]  # Store original time dimensions
    padded_batch = pad_sequence(batch, batch_first=True, padding_value=0)  # Pad along `t`
    
    return padded_batch, torch.tensor(lengths)

ENC_BLUEPRINT = (
    ('space-time_attn', {
        'n_rep' : 2,
        'n_embd' : 256,
        'n_head' : 4,
        'd_head' : 8,
    }),
    ('spacetime_downsample', {
        'in_channels' : 256,
        'kernel_size' : 3,
        'time_factor' : 1,
        'space_factor' : 2,
    }),
    ('space-time_attn', {
        'n_rep' : 2,
        'n_embd' : 256,
        'n_head' : 4,
        'd_head' : 8,
    }),
)

DEC_BLUEPRINT = (
    ('space-time_attn', {
        'n_rep' : 2,
        'n_embd' : 256,
        'n_head' : 4,
        'd_head' : 16,
        'has_ext' : True,
        'time_attn_kw'  : {'key_dim' : 8},
    }),
    ('spacetime_upsample', {
        'in_channels' : 256,
        'kernel_size' : 3,
        'time_factor' : 1,
        'space_factor' : 2,
    }),
    ('space-time_attn', {
        'n_rep' : 2,
        'n_embd' : 256,
        'n_head' : 4,
        'd_head' : 16,
        'has_ext' : True,
        'time_attn_kw'  : {'key_dim' : 8},
    }),
)

def evaluate(model, data, accelerator):
    model.eval()
    total_loss = 0
    num_batches = len(data)

    with torch.no_grad():
        for batch in data:
            loss = model(batch)
            total_loss += loss.item()
    avg_loss = total_loss / num_batches
    return avg_loss


def train(model, train_data, val_data, optimizer, accelerator, writer, n_epochs):
    model.train() 

    for epoch in range(n_epochs):
        loss_avg = 0
        num_batches = len(train_data)
        model.train()

        for batch in train_data:
            optimizer.zero_grad(set_to_none=True) 

            loss = model(batch) 
            accelerator.backward(loss) 
            optimizer.step() 

            loss_avg += loss.item()

        loss_avg /= num_batches 
        writer.add_scalar("Epoch_Loss_Avg/train", loss_avg, epoch) 
        
        val_loss_avg = evaluate(model, val_data, accelerator)
        
        writer.add_scalar("Epoch_Loss_Avg/Validation", val_loss_avg, epoch)
        
        accelerator.print(f"Epoch [{epoch+1}/{n_epochs}], Training Loss: {loss_avg:.4f}, Validation Loss {val_loss_avg}") 

    return loss_avg


import argparse

def get_args():
    parser = argparse.ArgumentParser(description="Train Latent Action Model with Tokenized Data")
    
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the dataset directory")
    parser.add_argument("--labels_dir", type=str, required=True, help="Path to the labels directory")
    parser.add_argument("--output_dir", type=str, required=False, default="./", help="Path to where outputs are saved")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    
    parser.add_argument("--use_mixed_precision", action="store_true", help="Enable mixed precision")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle dataset")

    parser.add_argument("--device", type=str, choices=["cpu", "cuda", "tpu"], default="cuda", help="Device for training")


    return parser.parse_args()



if __name__ == '__main__':
    import argparse

    args = get_args()

    data_dir = args.data_dir
    labels_dir = args.labels_dir
    output_dir = args.output_dir
    num_epochs = args.epochs
    batch_size = args.batch_size
    lr = args.lr
    mixed_precision = args.use_mixed_precision
    shuffle = args.shuffle
    device = args.device

    writer = SummaryWriter()
    accelerator = Accelerator()

    model = LatentAction(
        enc_desc = ENC_BLUEPRINT,
        dec_desc = DEC_BLUEPRINT,
        d_codebook=8,      
        inp_channels=1,     
        inp_shape=(32, 32), 
        n_embd=256, 
    )

    train_data = DataLoader(TokenizedSthv2(data_dir, os.path.join(labels_dir, "train.json")), batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    val_data = DataLoader(TokenizedSthv2(data_dir, os.path.join(labels_dir, "validation.json")),batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    test_data = DataLoader(TokenizedSthv2(data_dir, os.path.join(labels_dir, "test.json")),batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model, optimizer, data = accelerator.prepare(model, optimizer, train_data)

    model.train()

    train_loss_avg = train(model, train_data, val_data, optimizer, accelerator, writer, num_epochs)

    print(f"Final Average Training Loss: {train_loss_avg}")

    test_loss_avg = evaluate(model, test_data, accelerator)

    print(f"Average Test Loss: {test_loss_avg}")

    torch.save(model.state_dict(), os.path.join(output_dir, f'lr{lr}-bs{batch_size}epochs-{num_epochs}.pth'))
