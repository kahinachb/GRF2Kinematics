#!/bin/bash
#SBATCH --job-name=train_diffuser_large_dataset
#SBATCH -A vsi@v100
#SBATCH -p gpu_p13
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --time=20:00:00
#SBATCH --output=LOGgrf2kine/out/%j.out
#SBATCH --error=LOGgrf2kine/err/%j.err
#SBATCH --hint=nomultithread
#SBATCH --exclusive

module purge
module load pytorch-gpu/py3/2.6.0

export MASTER_ADDR=$(scontrol show hostnames $SLURM_NODELIST | head -n1)
export MASTER_PORT=29500

srun python train_diffuser_lower_PE_sinTimestepEmbedding.py
 
