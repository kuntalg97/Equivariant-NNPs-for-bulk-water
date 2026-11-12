#!/bin/bash

#SBATCH --job-name=wat_AANN
#SBATCH --output=us-%j.out
# >>> Insert SLURM details here (e.g. #SBATCH --account=..., #SBATCH --partition=..., #SBATCH --time=...)
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=48

# >>> Insert module loads for your cluster here (e.g. Python, MPI)
ulimit -l unlimited
conda deactivate
conda activate <PATH_TO_DEEPMD_CONDA_ENV>

PLUMED_KERNEL=<PATH_TO_PLUMED_INSTALL>/lib/libplumedKernel.so

cores=48

mpirun -np ${cores} lmp_mpi -in in.water -var SEED ${RANDOM}
