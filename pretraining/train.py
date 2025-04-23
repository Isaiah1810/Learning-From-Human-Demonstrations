import yaml
from pathlib import Path
import argparse
from model.action_predictor import VideoToAction
from model.trainer import VideoActionTrainer
import wandb
import torch.multiprocessing as mp
from data.dataset import VideoDataset
mp.set_start_method('spawn', force=True)

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config.yaml')
    args = parser.parse_args()

    config = load_config(args.config)
    model_cfg = config['model']
    train_cfg = config['train']

    # Prepare results directory
    results_folder = Path(train_cfg['results_folder'])
    results_folder.mkdir(parents=True, exist_ok=True)

    # Detect checkpoint
    running_ckpt_path = results_folder / "current_model.pt"
    resume_ckpt_path = None
    if running_ckpt_path.exists():
        resume_ckpt_path = running_ckpt_path
    elif train_cfg.get('checkpoint') is not None:
        resume_ckpt_path = Path(train_cfg['checkpoint'])

    # Prepare W&B kwargs for trainer
    wandb_id = train_cfg.get('run_id', wandb.util.generate_id())
    wandb_kwargs = {
        "mode": train_cfg.get("wandb_mode", "disabled"),
        "name": results_folder.name,
        "id": wandb_id,
        "resume": "allow",
        "config": config
    }
    config['train']['run_id'] = wandb_id

    data_dir = config['train']['dataset_path']

    # Model
    model = VideoToAction(
        input_dim=model_cfg['input_dim'],
        model_dim=model_cfg['model_dim'],
        action_dim=model_cfg['action_dim'],
        encoder_depth=model_cfg['encoder_depth'],
        decoder_depth=model_cfg['decoder_depth'],
        heads=model_cfg['heads'],
        dim_head=model_cfg['dim_head'],
        ff_mult=model_cfg['ff_mult'],
        attn_dropout=model_cfg['attn_dropout'],
        ff_dropout=model_cfg['ff_dropout'],
        use_rel_pos_spatial=model_cfg['use_rel_pos_spatial'],
        use_rel_pos_temporal=model_cfg['use_rel_pos_temporal'],
        use_peg_spatial_layers_enc=model_cfg['use_peg_spatial_layers_enc'],
        use_peg_temporal_layers_enc=model_cfg['use_peg_temporal_layers_enc'],
        use_peg_spatial_layers_dec=model_cfg['use_peg_spatial_layers_dec'],
        use_peg_temporal_layers_dec=model_cfg['use_peg_temporal_layers_dec'],
        attn_num_null_kv=model_cfg['attn_num_null_kv'],
        loss_type=train_cfg['loss_type'],
        tokenizer_config=config,
        use_tokenizer=True
    )

    dataset = VideoDataset(data_dir)

    # Trainer
    trainer = VideoActionTrainer(
        model=model,
        dataset=dataset,
        batch_size=train_cfg['batch_size'],
        num_train_steps=train_cfg['num_train_steps'],
        results_folder=str(results_folder),
        lr=train_cfg['lr'],
        grad_accum_every=train_cfg['grad_accum_every'],
        max_grad_norm=train_cfg['max_grad_norm'],
        use_ema=train_cfg.get('use_ema', False),
        save_model_every=train_cfg['save_model_every'],
        save_milestone_every=train_cfg['save_milestone_every'],
        milestone_optim=train_cfg.get('milestone_optim', True),
        accelerator_kwargs=dict(log_with="wandb"),
        resume_checkpoint=str(resume_ckpt_path) if resume_ckpt_path else None,
        wandb_kwargs=wandb_kwargs
    )

    trainer.train()


if __name__ == '__main__':
    main()
