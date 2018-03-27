#!/bin/zsh -l
#PBS -m be
#PBS -M jklymak@gmail.com
#PBS -l select=1:ncpus=1
#PBS -l walltime=00:15:00
#PBS -q transfer
#PBS -A ONRDC35552400
#PBS -j oe
#PBS -N GetEnergyBudget

# run w/ qsub -v PRE="LWRegrid2low01U02",U0="10",N0="10",f0="38" runGetEnergy.sh

cd $PBS_O_WORKDIR
echo $PRE
echo $U0
echo $N0
echo $f0
source /u/home/jklymak/.zshrc
${HOME}/miniconda3/bin/python ../python/GetEnergyBudget.py $PRE 0.${U0} ${N0} ${f0}
