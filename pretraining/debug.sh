#!/bin/bash
#SBATCH --job-name=laq_dryrun
#SBATCH --output=jobs/dryrun.%j.out
#SBATCH --error=jobs/dryrun.%j.err
#SBATCH --partition=general
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=160G
#SBATCH --time=00:30:00
#SBATCH --constraint='A100_80GB|A6000|L40S|L40|A100_40GB'
#SBATCH --exclude='babel-14-25,babel-11-5,babel-1-23,babel-13-13'

# optional env for debugging
export NCCL_P2P_DISABLE=1

source /usr/share/Modules/init/bash
module load cuda-12.4

# Activate conda env
export PATH=/home/sroutra2/miniconda3/envs/lq/bin:$PATH

export WANDB_USER_NAME="sroutray"
export WANDB_API_KEY="c26eceb1ceadd0a0f6dd1e5beafe9ce9e431be0b"
wandb login $WANDB_API_KEY
# Disable W&B for dry run
export WANDB_MODE=disabled

# Run with minimal config and dummy dataset
accelerate launch \
  --num_processes=1 \
  --mixed_precision=bf16 \
  --main_process_port=27562 \
  train.py \
  --config configs/config.yaml