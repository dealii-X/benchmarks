#!/bin/bash -l
#SBATCH --job-name=dealii      # Job name
## SBATCH --output=myjob.out    # Output file
## SBATCH --error=myjob.err     # Error file
#SBATCH -J myjob

#SBATCH --time=00:05:00       # Maximum runtime (hh:mm:ss)

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16     # Number of CPU cores per task

#SBATCH --partition=a100       # Partition to run in (can be different)
#SBATCH --gres=gpu:1

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_PLACES=cores
export OMP_PROC_BIND=spread


srun ./benchmark 5


