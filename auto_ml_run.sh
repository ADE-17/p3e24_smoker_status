#!/bin/bash
#SBATCH --clusters=tinygpu
#SBATCH --partition=rtx2080ti
#SBATCH --gres=gpu:rtx2080ti:1
#SBATCH --time=24:00:00

cd $WORK
# module load python/3.10-anaconda
source miniconda/bin/activate
srun python /home/woody/iwso/iwso092h/kaggle/p3e24_smoker_status/automl_run.py