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

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"


class TokenizedSthv2(Dataset):
    def __init__(self, data_dir, dataset_name='tokens'):
        self.data_dir = data_dir
        self.dataset_name = dataset_name

        self.files = []

        if split_json == None:
            self.files = os.listdir(data_dir)

        self.h5_files = [os.path.join(data_dir, fname) for fname in self.files]
        for file_path in self.h5_files:
            with h5py.File(file_path, 'r') as f:
                data = f['tokens']
                data = torch.Tensor(data)
                for i in range(data.shape[0]):
                    self.files.append(file_path)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        with h5py.File(file_path, 'r') as f:
            data = f['tokens']
            data = torch.Tensor(data)
        data = torch.tensor(data, dtype=torch.float32).cuda()
        return data


def collate_fn(batch):

    lengths = [item.shape[0] for item in batch]  
    padded_batch = pad_sequence(batch, batch_first=True, padding_value=0)  
    
    return padded_batch

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

def evaluate(model, data_loader, accelerator):
    model.eval()
    total_loss = 0
    num_batches = len(data_loader)

    with torch.no_grad():
        for batch in data_loader:
            with accelerator.autocast():
                loss = model(batch)
            total_loss += loss.item()
    
    # Gather losses across all processes and average
    avg_loss = accelerator.gather(total_loss)
    avg_loss = avg_loss.mean()
    
    return avg_loss


def train(model, train_dataloader, val_dataloader, optimizer, lr_scheduler, accelerator, writer, n_epochs):

    model, optimizer, train_dataloader, val_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, val_dataloader, lr_scheduler
    )

    for epoch in range(n_epochs):
        model.train()
        total_train_loss = 0
        
        for batch in train_dataloader:
            optimizer.zero_grad()
            
            with accelerator.autocast():
                loss, _ = model(batch)
            
            accelerator.backward(loss)
            
            accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            lr_scheduler.step()
            
            total_train_loss += loss.detach()
        
        avg_train_loss = accelerator.gather(total_train_loss).mean()
        
        model.eval()
        val_loss = evaluate(model, val_dataloader, accelerator)
        
        if accelerator.is_main_process:
            writer.add_scalar("Epoch_Loss_Avg/train", avg_train_loss, epoch)
            writer.add_scalar("Epoch_Loss_Avg/Validation", val_loss, epoch)
            accelerator.print(f"Epoch [{epoch+1}/{n_epochs}], "
                              f"Training Loss: {avg_train_loss:.4f}, "
                              f"Validation Loss: {val_loss:.4f}")
        
        if accelerator.is_main_process:
            accelerator.save_state(os.path.join(output_dir, f"checkpoint_epoch_{epoch}"))

    return avg_train_loss

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

    train_data = DataLoader(TokenizedSthv2(os.path.join(data_dir, 'train')), batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    val_data = DataLoader(TokenizedSthv2(os.path.join(data_dir, 'validation')),batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    test_data = DataLoader(TokenizedSthv2(os.path.join(data_dir, 'test')),batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model, optimizer, data = accelerator.prepare(model, optimizer, train_data)

    model.train()

    train_loss_avg = train(model, train_data, val_data, optimizer, accelerator, writer, num_epochs)

    print(f"Final Average Training Loss: {train_loss_avg}")

    test_loss_avg = evaluate(model, test_data, accelerator)

    print(f"Average Test Loss: {test_loss_avg}")

    torch.save(model.state_dict(), os.path.join(output_dir, f'lr{lr}-bs{batch_size}epochs-{num_epochs}.pth'))
