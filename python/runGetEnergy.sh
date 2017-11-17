#!/bin/sh -l
#PBS -m be
#PBS -M jklymak@gmail.com
#PBS -l select=1:ncpus=1
#PBS -l walltime=0:10:00
#PBS -q transfer
#PBS -A ONRDC35552400
#PBS -j oe
#PBS -N GetEnergyBudget.py


cd $PBS_O_WORKDIR
echo $PBS_O_WORKDIR
pwd
#python GetEnergyBudget.py lee3dfull06U10 0.10
#python GetEnergyBudget.py lee3dfilt02U10 0.10
#python GetEnergyBudget.py lee3dfilt04U10 0.10
#python GetEnergyBudget.py lee3dfilt05U10 0.10
#python GetEnergyBudget.py lee3dfilt06U10 0.10
#python GetEnergyBudget.py lee3dlow03U10 0.10
#python GetEnergyBudget.py lee3dlow04U10 0.10
#python GetEnergyBudget.py lee3dlow05U10 0.10
python GetEnergyBudget.py lee3dlow06U10 0.10
#python GetEnergyBudget.py lee3dlow02U10 0.10
#python GetEnergyBudget.py lee3dfilt02U10 0.10
#python GetEnergyBudget.py lee3dfull01U20 0.20
#python GetEnergyBudget.py lee3dfilt01U05 0.05
#python GetEnergyBudget.py lee3dfilt01U20 0.20
