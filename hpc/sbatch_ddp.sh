#!/bin/bash -l

#SBATCH -J tiny-lm-ddp
#SBATCH -A LXP
#SBATCH -p gpu
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH -t 00:15:00
#SBATCH -o logs/%x-%j.out
#SBATCH -q default

mkdir -p logs

module purge
module load Python
source /home/users/fmansouri/scratch-farouk/demo-dafab/dafab-summer-school25/env/bin/activate

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NCCL_DEBUG=INFO
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1

echo "NODELIST="${SLURM_NODELIST}
master_ip=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
master_port=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')
echo "head node is ${master_ip}:${master_port}"

NUM_NODES=$SLURM_JOB_NUM_NODES
export GPUS=$SLURM_JOB_NUM_GPUS
export PORT=$master_port
export MASTER_ADDR=$master_ip
export OMP_NUM_THREADS=16

echo "Number of GPUs per node: $GPUS"
echo "Number of nodes allocated: $NUM_NODES"


NCCL_CROSS_NIC=0 CUDA_VISIBLE_DEVICES=0,1,2,3 srun torchrun \
--nnodes=$SLURM_JOB_NUM_NODES \
--nproc_per_node=$SLURM_GPUS_PER_NODE \
--rdzv_backend=c10d \
--rdzv_endpoint=$MASTER_ADDR:$PORT \
train_ddp.py --epochs 1 --batch_size 16 --fp16 --outdir runs/$SLURM_JOB_ID
