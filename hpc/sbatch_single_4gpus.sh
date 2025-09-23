#!/bin/bash -l

#SBATCH -J tiny-lm-single
#SBATCH -A LXP
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH -t 00:10:00
#SBATCH -o logs/%x-%j.out
#SBATCH -q default

set -euo pipefail
mkdir -p logs

# --- Environment (choose one stack) ---
module purge
module load Python

python3 -m pip install --user -r requirements.txt --no-warn-script-location
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

srun torchrun --nnodes=1 --nproc_per_node=4 train_ddp.py \
  --epochs 1 --batch_size 16 --fp16 \
  --outdir runs/$SLURM_JOB_ID
