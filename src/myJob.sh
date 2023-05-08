#!/bin/bash
#SBATCH --job-name=job
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28
#SBATCH --time=00:10:00
#SBATCH --output=output/farida_llvm_100.txt
#SBATCH --reservation=CHPS
#SBATCH --exclusive

# Module
module load opencv/
module load gcc/
module load llvm/12.0.1_spack2021_gcc-10.2.0-wje3

# Make
make clean
make

# Execute gcc
#./opencv_test.pgr img/farida.jpg

# llvm
./llvm_opencv_test.pgr img/farida.jpg