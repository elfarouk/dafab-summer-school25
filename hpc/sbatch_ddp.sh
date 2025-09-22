#!/bin/bash -l
#SBATCH -J tiny-lm-ddp
#SBATCH -A LXP
#SBATCH -p gpu
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
#SBATCH -t 00:15:00
#SBATCH -o logs/%x-%j.out
#SBATCH -q default

mkdir -p logs

module purge
module load Python
python3 -m pip install --user -r requirements.txt --no-warn-script-location

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NCCL_DEBUG=INFO
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1

# Pick up IB interfaces automatically (adjust if your cluster uses ib0, mlx5, etc.)
#export NCCL_SOCKET_IFNAME=$(ip -o link show | awk -F': ' '{print $2}' | grep -E 'ib|mlx|ens|enp' | paste -sd, -)

# Master node address/port (needed on some clusters)
MASTER_ADDR=$(scontrol show hostnames "$SLURM_NODELIST" | head -n1)
export MASTER_ADDR

srun torchrun \
    --nnodes=$SLURM_NNODES \
    --nproc_per_node=$SLURM_GPUS_PER_NODE \
    --rdzv_backend=c10d \
    --rdzv_endpoint=${MASTER_ADDR}:29500 \
    train_ddp.py --epochs 1 --batch_size 16 --fp16 --outdir runs/$SLURM_JOB_ID
