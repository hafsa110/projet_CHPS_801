#!/bin/bash
#SBATCH --job-name=my_job
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28
#SBATCH --time=00:10:00
#SBATCH --output=output/proc.txt
#SBATCH --reservation=CHPS
#SBATCH --exclusive


nproc
# Execute your C++ code with 12 threads per task
g++ -fopenmp -o proc proc.cpp
./proc