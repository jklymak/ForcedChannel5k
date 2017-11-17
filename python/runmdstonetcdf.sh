#!/bin/zsh -l
#PBS -m be
#PBS -M jklymak@gmail.com
#PBS -l select=1:ncpus=1
#PBS -l walltime=1:35:30
#PBS -q transfer
#PBS -A ONRDC35552400
#PBS -j oe
#PBS -N mdstonetcdf

cd $PBS_O_WORKDIR
source /u/home/jklymak/.zshrc
echo ${PATH}
free -tg
PRE=CW3dfull01U10
for ITER in $(seq 36000 15 36180)
do
    echo $ITER
    python mdstonetcdf.py $PRE $ITER
#    python mdstonetcdfW.py $PRE $ITER
#    python mdscombine.py $PRE $ITER
done
rsync -av ../results/$PRE/_Model/input/*.nc valdez.seos.uvic.ca:leewaves17/LWCoarse2/$PRE/
