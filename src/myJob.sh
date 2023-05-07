#!/bin/bash
#SBATCH --job-name=my_job
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28
#SBATCH --time=00:10:00
#SBATCH --output=output/farida_100.txt
#SBATCH --reservation=CHPS
#SBATCH --exclusive

# Load any necessary modules
module load opencv/

# Compile your code using make
make clean
make

nproc
# Execute your C++ code with 12 threads per task
./opencv_test.pgr img/farida.jpg