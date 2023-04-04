#!/bin/sh
#SBATCH -N 1
#SBATCH --partition bdwall
#SBATCH --exclusive
#SBATCH --time=00:10:00
#SBATCH --job-name=AL4GAP_init
#SBATCH --output=AL4GAP_init.out
#SBATCH --error=AL4GAP_init.err
#SBATCH --account=AL-IP

CONDA_ENV=/home/ac.vwoo/miniconda3/envs/al4gap
DRIVER=/lcrc/project/AL-IP/vwoo/AL4GAP/driver.py

dbnodes=1
simnodes=3
nodes=$(($dbnodes + $simnodes))
ntasks=8

# command line arguments
echo number of total nodes $nodes #1
echo number of database nodes $dbnodes #2 
echo number of simulation nodes $simnodes #3
echo number of tasks to launch $ntasks #4 
HOST_FILE=$(echo $SLURM_JOB_NODELIST) #5
SLURM_JOB_ID=$(echo $SLURM_JOB_ID) #6 

# set env
source /home/ac.vwoo/miniconda3/etc/profile.d/conda.sh
conda activate $CONDA_ENV
export SMARTSIM_LOG_LEVEL=debug

# Run driver
python $DRIVER $nodes $dbnodes $simnodes $ntasks $HOST_FILE $SLURM_JOB_ID
