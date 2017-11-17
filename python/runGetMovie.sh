#!/bin/sh -l
#PBS -m be
#PBS -M jklymak@gmail.com
#PBS -l select=1:ncpus=1
#PBS -l walltime=2:10:00
#PBS -q transfer
#PBS -A ONRDC35552400
#PBS -j oe
#PBS -N getMovie


cd $PBS_O_WORKDIR
echo $PBS_O_WORKDIR
pwd
module load costinit
module load /scr/projects/COST/modules/pkgs/python/2.6.8 
module load /scr/projects/COST/modules/pkgs/numpy/1.6.2 
module load /scr/projects/COST/modules/pkgs/scipy/0.11.0 
module load /scr/projects/COST/modules/pkgs/matplotlib/1.1.1 
setenv PYTHONPATH ~/python/lib/:${PYTHONPATH}
python makeMovie.py
