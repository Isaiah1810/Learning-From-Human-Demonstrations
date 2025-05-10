# === Imports ===
import os
import sys
import yaml
import h5py
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from accelerate import Accelerator
from src.latent_action import LatentActionModel


# === Dataset Classes ===
class TokenizedSthv2(Dataset):
    """
    Dataset for loading tokenized data stored in .h5 files.

    Args:
        data_dir (str): Path to the directory containing .h5 files.
        dataset_name (str): Name of the dataset inside the .h5 file.
        max_skip (int): Maximum skip interval during sampling.
    """
    def __init__(self, data_dir, dataset_name='tokens', max_skip=3):
        self.data_dir = data_dir
        self.dataset_name = dataset_name
        self.normalizer = Normalizer()
        self.files = [os.path.join(data_dir, fname) for fname in os.listdir(data_dir) if fname.endswith('.h5')]
        self.max_skip = max_skip 
        self.training = 'train' in data_dir 

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        skip = np.random.randint(0, self.max_skip + 1) if self.training else 0

        with h5py.File(file_path, 'r') as f:
            data = np.array(f[self.dataset_name])
            if skip > 0:
                data = data[::skip + 1]
            data = torch.Tensor(data)
            norm_data, _, _ = self.normalizer.normalize(data)

        return norm_data


# === Utility Classes ===
class Normalizer():
    """
    Normalizes and denormalizes tensor data to range [-1, 1].
    """
    def normalize(self, data):
        data_min = data.min(dim=2, keepdim=True)[0]
        data_max = data.max(dim=2, keepdim=True)[0]
        data = (data - data_min) / (data_max - data_min + 1e-9)
        data = data * 2 - 1
        return data, data_min, data_max

    def denormalize(self, data, min_val, max_val):
        return 0.5 * (data + 1) * (max_val - min_val) + min_val


# === Data Loader Helper ===
def variable_collate_fn(batch):
    """
    Pads sequences in a batch to the same length.
    """
    return pad_sequence(batch, batch_first=True, padding_value=0.0)


# === Evaluation Function ===
def evaluate(model, data_loader, accelerator):
    """
    Evaluates the model on validation data.

    Args:
        model: PyTorch model.
        data_loader: DataLoader for validation data.
        accelerator: HuggingFace accelerator.

    Returns:
        float: Average loss over the validation set.
    """
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            with accelerator.autocast():
                outputs = model({'tokens': batch})
            loss = outputs['loss']
            if not loss.isnan():
                total_loss += loss.item()

    return total_loss / (len(data_loader) + 1e-9)


# === Training Function ===
def train(model, train_dataloader, val_dataloader, optimizer, accelerator, n_epochs, run):
    """
    Training loop for the model.

    Returns:
        avg_train_loss, train_losses_per_epoch, val_losses_per_epoch
    """
    model, optimizer, train_dataloader, val_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, val_dataloader
    )

    epoch_train_losses = np.zeros(n_epochs)
    epoch_val_losses = np.zeros(n_epochs)

    for epoch in range(n_epochs):
        model.train()
        total_train_loss = 0.0

        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch}"):
            optimizer.zero_grad()

            with accelerator.autocast():
                outputs = model({'tokens': batch})
            loss = outputs['loss']
            accelerator.backward(loss)
            accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_train_loss += loss.detach()

        # Average loss across processes
        avg_train_loss = accelerator.gather(total_train_loss) / (len(train_dataloader) + 1e-9)
        avg_train_loss = avg_train_loss.sum() / len(avg_train_loss)

        epoch_train_losses[epoch] = avg_train_loss
        val_loss = evaluate(model, val_dataloader, accelerator)
        epoch_val_losses[epoch] = val_loss

        if accelerator.is_main_process:
            wandb.log({'validation loss': val_loss, 'training_loss': avg_train_loss})
            checkpoint_dir = os.path.join(run.config.training['output_dir']['value'], 'checkpoints', run.id)
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            wandb.log_model(checkpoint_dir, f'{run.id}_epoch_{epoch}')

        torch.cuda.empty_cache()

    return avg_train_loss, epoch_train_losses, epoch_val_losses


# === Main Script ===
if __name__ == '__main__':
    torch.backends.cudnn.enabled = False

    accelerator = Accelerator(device_placement=True, mixed_precision='fp16')
    print(f"Using {accelerator.device} GPUs.")
    print(f"Current Device: {torch.cuda.current_device()}")
    print(f"Device Name: {torch.cuda.get_device_name(torch.cuda.current_device())}")

    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)

    # Setup Weights & Biases
    if accelerator.is_main_process:
        import wandb
        wandb.login()
        run = wandb.init(
            project=config['training']['wandb_project']['value'],
            config=config
        )
    else:
        run = None

    # Initialize model
    model = LatentActionModel(
        in_dim=config['model']['in_dim']['value'],   
        model_dim=config['model']['model_dim']['value'],             
        latent_dim=config['model']['latent_dim']['value'],                           
        enc_blocks=config['model']['enc_blocks']['value'],                             
        dec_blocks=config['model']['dec_blocks']['value'],                            
        num_heads=config['model']['num_heads']['value'],                     
        dropout=config['model']['dropout']['value'],
        global_patch=config['model']['global_patch']['value']
    )

    # Load datasets
    data_dir = config['training']['data_dir']['value']
    batch_size = config['training']['batch_size']['value']
    num_epochs = config['training']['epochs']['value']
    lr = config['training']['learning_rate']['value']
    shuffle = config['training']['shuffle']['value']

    train_data = DataLoader(
        TokenizedSthv2(os.path.join(data_dir, 'train')),
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=variable_collate_fn
    )

    val_data = DataLoader(
        TokenizedSthv2(os.path.join(data_dir, 'validation')),
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=variable_collate_fn
    )

    test_data = DataLoader(
        TokenizedSthv2(os.path.join(data_dir, 'test')),
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=variable_collate_fn
    )

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Prepare with accelerator
    model, optimizer, train_data, val_data, test_data = accelerator.prepare(
        model, optimizer, train_data, val_data, test_data
    )

    # Start training
    model.train()
    train_loss_avg, train_losses, val_losses = train(
        model, train_data, val_data, optimizer, accelerator, num_epochs, run
    )
