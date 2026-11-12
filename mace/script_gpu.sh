#!/bin/bash

# >>> Insert SLURM details here (e.g. #SBATCH --account=..., #SBATCH --partition=<GPU partition>, #SBATCH --time=...)
#SBATCH --gres=gpu:1
#SBATCH --job-name=matpes
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

# >>> Insert module loads for your cluster here (e.g. MPI, CUDA, CMake, MKL, GSL)

conda activate <PATH_TO_MACE_CONDA_ENV>

ntasks_per_node=$SLURM_NTASKS_PER_NODE
numnodes=$SLURM_JOB_NUM_NODES
n=$(( ntasks_per_node * numnodes ))

mpirun -np $n <PATH_TO_MACE_LAMMPS_BUILD>/lmp -k on g 1 -sf kk -in water.in
