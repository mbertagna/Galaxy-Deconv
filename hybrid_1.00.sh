#!/bin/bash
#SBATCH -A e32704  # Account name
#SBATCH -p gengpu  # GPU partition
#SBATCH --gres=gpu:a100:1  # Request 1 A100 GPU
#SBATCH -N 1  # Number of nodes
#SBATCH -n 20  # Number of tasks
#SBATCH -t 16:00:00  # Max runtime
#SBATCH --mem=32G  # Memory allocation
#SBATCH --job-name=train_hybrid  # Job name
#SBATCH --output=train_%j.log  # Log file (SLURM_JOB_ID included)

# Load Python module
module load python/3.10.1

# Ensure real-time logging
export PYTHONUNBUFFERED=1

# Activate virtual environment
if [ -d "Galaxy-Deconv.env" ]; then
    source Galaxy-Deconv.env/bin/activate
else
    echo "ERROR: Virtual environment 'Galaxy-Deconv.env' not found!"
    exit 1
fi

# Run training script and log output
time python train.py --model Unrolled_ADMM --n_iters 2 --n_epochs 20 --loss L1_FPFSCoeffLoss --batch_size 64 --lr 1e-4 --aux_weight 1.00 --loss_coeffs "m20" "m22c" "m22s" "m40" "m42c" "m42s" "m44c" "m44s" 2>&1 | tee "train_output_${SLURM_JOB_ID}.txt"