#!/bin/sh
#SBATCH -o SOAP-BO
#SBATCH --nodes=1
#SBATCH -A  AL-IP
#SBATCH --ntasks-per-node=36
#SBATCH -p bdwall
#SBATCH --time=24:00:00 

module purge 

#source ~/.bashrc
module load   intel/17.0.4-74uvhji  intel-mpi/2017.3-dfphq6k  intel-mkl/2017.3.196-v7uuj6z    StdEnv
export OMP_NUM_THREADS=36

export OMP_STACKSIZE=8192M 

export PYTHONPATH=/home/gsivaraman/miniconda3/lib/python3.7/site-packages/
#export QUIP_PATH=/lcrc/project/IP-ML/vama/libatoms/QUIP-git/build/linux_x86_64_ifort_icc_openmp/
export QUIP_PATH=/home/gsivaraman/libatoms/QUIP/build/linux_x86_64_ifort_icc_openmp/
ulimit -s unlimited



ulimit -s unlimited
which python 

python -u  BayesOpt_SOAP.py  > BO-SOAP.out 

echo "Clean up and exit!"
rm gap.xml* *idx
