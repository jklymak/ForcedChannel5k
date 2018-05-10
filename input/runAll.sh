#!/bin/bash

# run as: source runAll.sh LW1kmlowU10Amp305K18 10 10 100
todo=ChannelToyHighCd
U0=$2
N0=$3
f0=$4

cd ../results/${todo}/input

MainRun=$(mpirun -np 2 ../build/mitgcmuv >& mit.out)

# qsub -W depend=afterany:$MainRun -N ${todo} transfertoarchive.sh
# qsub -W depend=afterany:$MainRun -v  PRE="${todo}",U0="${U0}",N0="${N0}",f0="${f0}"  runGetEnergy.sh
