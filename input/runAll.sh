#!/bin/bash

# run as: source runAll.sh LW1kmlowU10Amp305K18 10 10 100
todo=$1

MainRun=$(qsub -N ${todo} runModel01.sh )
qsub -W depend=afterany:$MainRun -N ${todo} runModel02.sh

# qsub -W depend=afterany:$MainRun -v  PRE="${todo}",U0="${U0}",N0="${N0}",f0="${f0}"  runGetEnergy.sh
