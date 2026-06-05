#!/bin/bash
#SBATCH --job-name=train_diffuser_large_dataset
#SBATCH -A vsi@v100
#SBATCH -p gpu_p13
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=20:00:00
#SBATCH --output=LOG_fm/out/%j.out
#SBATCH --error=LOG_fm/err/%j.err
#SBATCH --hint=nomultithread
#SBATCH --exclusive

module purge
module load pytorch-gpu/py3/2.6.0

export MASTER_ADDR=$(scontrol show hostnames $SLURM_NODELIST | head -n1)
export MASTER_PORT=29500

srun python train_fm_lower_weight_seg.py
 
