#!/usr/bin/env bash
#SBATCH -J tiny-lm-ds
#SBATCH -A <project>
#SBATCH -p <gpu-partition>
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
#SBATCH -t 00:15:00
#SBATCH -o logs/%x-%j.out

set -euo pipefail
python3 -m pip install --user -r requirements.txt --no-warn-script-location
python3 -m pip install --user deepspeed --no-warn-script-location

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

srun deepspeed --num_nodes=$SLURM_NNODES --num_gpus=$SLURM_GPUS_PER_NODE \
  train_ddp.py --epochs 1 --batch_size 16 --fp16 --outdir runs/$SLURM_JOB_ID \
  --deepspeed hpc/deepspeed_config.json
