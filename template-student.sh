#!/bin/bash


#SBATCH --account students
#SBATCH --partition students
#SBATCH --job-name my-job
#SBATCH --gres gpushard:4090-1gb:4
#SBATCH --nodes 1
#SBATCH --tasks-per-node 1
#SBATCH --cpus-per-task 1
#SBATCH --mem 2GB
#SBATCH --time=00:35:00
#SBATCH --output slurm_logs/slurm-%j.log


# Initialize conda & activate environment
source /home/ioankots/miniconda3/etc/profile.d/conda.sh
conda activate torch

# Some common diagnostics
echo "----------------"
echo "Date: $(date)"
echo "Host: $(hostname -f)"
echo "User: $(whoami)"
echo "----------------"
# And some specific to python (and conda)
echo "Python: $(which python)"
echo "----------------"

# Main
srun --nodes 1 --ntasks 1 python project_KI.py &

wait
