import numpy as np
import torch 
import sys
sys.path.append("./src/modules")
sys.path.append('./src/')
from latent_action_multipatch import LatentActionModel
from accelerate import Accelerator
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import h5py
import os
import json
from tqdm import tqdm

class TokenizedSthv2(Dataset):
    def __init__(self, data_dir, dataset_name='tokens'):
        self.data_dir = data_dir
        self.dataset_name = dataset_name
        self.normalizer = Normalizer()
        self.files = [os.path.join(data_dir, fname) for fname in os.listdir(data_dir) if fname.endswith('.h5')]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        with h5py.File(file_path, 'r') as f:
            data = f['tokens']
            data = torch.Tensor(np.array(data))
            norm_data, _, _ = self.normalizer.normalize(data)
        return norm_data


class Normalizer():
    def __init__(self):
        pass
    def normalize(self, data):
        data_min = data.min(dim=(2), keepdims=True)[0]
        data_max = data.max(dim=(2), keepdims=True)[0]
        data.sub_(data_min).div_(data_max - data_min + 1e-9)
        data.mul_(2).sub_(1)
    
        return data, data_min, data_max
    
    def denormalize(self, data, min_val, max_val):
        denorm = 0.5*(data + 1)
        denorm = denorm * (max_val - min_val) + min_val
        return denorm
    


def evaluate(model, data_loader, accelerator):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in tqdm(data_loader, desc=f"Evaluating"):
       
            

            with accelerator.autocast():
                outputs = model({'tokens':batch})
            loss = outputs['loss']
            if not loss.isnan():
                total_loss += loss.item()

            else:
                print("loss", loss)
                print(batch.min(), batch.max(), torch.isnan(batch).any())
                
                 
    # Gather losses across all processes and average
    avg_loss = total_loss / (len(data_loader) + 1e-9)
    print(avg_loss)
    
    return avg_loss


def train(model, train_dataloader, val_dataloader, optimizer, accelerator, 
          n_epochs, run):

    model, optimizer, train_dataloader, val_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, val_dataloader
    )
    
    epoch_train_losses = np.zeros(n_epochs)
    epoch_val_losses = np.zeros(n_epochs)

    for epoch in range(n_epochs):
        model.train()
        total_train_loss = 0
           
        for batch in tqdm(train_dataloader, desc=f"Epoch:{epoch}"):
            optimizer.zero_grad()
            
          #  print(type(batch))
          #  print(batch.shape)

            with accelerator.autocast():
                outputs = model({'tokens':batch})
          #  loss, _ = model(batch)
            loss = outputs['loss']
            accelerator.backward(loss)
            
            accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_train_loss += loss.detach()
            
          #  print("Average Batch Loss:", loss.detach())
            
        avg_train_loss = accelerator.gather(total_train_loss) / (len(train_dataloader) + 1e-9)
      
        avg_train_loss = avg_train_loss.sum() / (len(avg_train_loss))

        epoch_train_losses[epoch] = avg_train_loss

        model.eval()
        val_loss = evaluate(model, val_dataloader, accelerator)
        
        epoch_val_losses[epoch] = val_loss
        
        if accelerator.is_main_process:
            wandb.log({
                'validation loss': val_loss, 
                'training_loss': avg_train_loss
                })

            accelerator.print(f"Epoch [{epoch+1}/{n_epochs}], "
                              f"Training Loss: {avg_train_loss:.4f}, "
                              f"Validation Loss: {val_loss:.4f}")
       
        torch.cuda.empty_cache()

        if accelerator.is_main_process:

            checkpoint_dir = os.path.join(run.config.training['output_dir']['value'], 
                                          'checkpoints', run.id)
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pth")

            torch.save(model.state_dict(), checkpoint_path)
            wandb.log_model(checkpoint_dir, f'{run.id}_epoch_{epoch}')

    return avg_train_loss, epoch_train_losses, epoch_val_losses

import argparse

# def get_args():
#     parser = argparse.ArgumentParser(description="Train Latent Action Model with Tokenized Data")
    
#     parser.add_argument("--data_dir", type=str, required=True, help="Path to the dataset directory")
#     parser.add_argument("--labels_dir", type=str, required=True, help="Path to the labels directory")
#     parser.add_argument("--output_dir", type=str, required=False, default="./", help="Path to where outputs are saved")
#     parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
#     parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
#     parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    
#     parser.add_argument("--shuffle", action="store_true", help="Shuffle dataset")


    # return parser.parse_args()



if __name__ == '__main__':
    # import argparse
    import yaml

    # args = get_args()
    torch.backends.cudnn.enabled = False

    # data_dir = args.data_dir
    # labels_dir = args.labels_dir
    # output_dir = args.output_dir
    # num_epochs = args.epochs
    # batch_size = args.batch_size
    # lr = args.lr
    # shuffle = args.shuffle

    accelerator = Accelerator(device_placement=True, mixed_precision='fp16')

    print(f"Using {accelerator.device} GPUs.")
    print(f"Current Device: {torch.cuda.current_device()}")
    print(f"Device Name: {torch.cuda.get_device_name(torch.cuda.current_device())}")


    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)

    if accelerator.is_main_process:
        import wandb
        wandb.login()
        run = wandb.init(
            project='lfhd_latent_action',
            config=config
        )
    else:
        run = None

    model = LatentActionModel(
        in_dim=config['model']['in_dim']['value'],  # Patch dimension     
        model_dim=config['model']['model_dim']['value'],             
        latent_dim=config['model']['latent_dim']['value'],                           
        enc_blocks=config['model']['enc_blocks']['value'],                             
        dec_blocks=config['model']['dec_blocks']['value'],                            
        num_heads=config['model']['num_heads']['value'],                     
        dropout=config['model']['dropout']['value']
    )

    data_dir = config['training']['data_dir']['value']
    batch_size = config['training']['batch_size']['value']
    num_epochs = config['training']['epochs']['value']
    lr = config['training']['learning_rate']['value']
    shuffle = config['training']['shuffle']['value']
    output_dir = config['training']['output_dir']['value']

    train_data = DataLoader(TokenizedSthv2(os.path.join(data_dir, 'train')), batch_size=batch_size, shuffle=shuffle)
    val_data = DataLoader(TokenizedSthv2(os.path.join(data_dir, 'validation')),batch_size=batch_size, shuffle=shuffle)
    test_data = DataLoader(TokenizedSthv2(os.path.join(data_dir, 'test')),batch_size=batch_size, shuffle=shuffle)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model, optimizer, train_data, val_data, test_data = accelerator.prepare(model, optimizer, train_data, val_data, test_data)

    model.train()

    train_loss_avg, train_losses, val_losses = train(model, train_data, val_data, optimizer, accelerator, num_epochs, run)

    

   # torch.save(model.state_dict(), os.path.join(output_dir, f'lr{lr}-bs{batch_size}epochs-{num_epochs}.pth'))
