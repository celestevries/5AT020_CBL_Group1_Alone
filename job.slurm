#!/bin/bash

# -------------- Set job requirements -------------- #
# Duration for which the nodes remain allocated D-HH:MM:SS
#SBATCH -t 1:00:00

# Set the partition you want to submit
#SBATCH -p gpu_mig

# Number of nodes
#SBATCH -N 1

# GPU MIG (9 cpus per node)
#SBATCH --cpus-per-task 9

# Number of GPUS for the job
#SBATCH --gpus=1

# Mail from a job
#SBATCH --mail-type=BEGIN,END
##SBATCH --mail-user=XXXXXX@student.tue.nl

# Define output folder for logs
#SBATCH -o ./logs/slurm-%j.out
 
# -------------- Loading modules -------------- #
module load 2024
module load CUDA/12.6.0
module load cuDNN/9.5.0.50-CUDA-12.6.0

# -------------- Get script's path -------------- #
SCRIPT_DIR="$(pwd)"

# -------------- Code -------------- #
# Load virtual environment
source "$SCRIPT_DIR/.venv/5AT020/bin/activate"

# Execute a Python program.
python "$SCRIPT_DIR/test_slurm.py"
