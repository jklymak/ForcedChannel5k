#!/bin/bash
#PBS -m be
#PBS -M jklymak@gmail.com
#PBS -l select=1:ncpus=1
#PBS -l walltime=00:20:00
#PBS -q transfer
#PBS -A ONRDC35552400
#PBS -j oe

# run as qsub -N LW1kmlowU10Amp305K18 transfertoarchive
TOP=AbHillParam
#TODO=Shelf1km01

TODO=${PBS_JOBNAME}

cd ${WORKDIR}/${TOP}
tar cfv ${TODO}.tar ${TODO}
archive put -C ${TOP} ${TODO}.tar
archive ls -halt ${TOP}
rm ${TODO}.tar
echo "Transfer job ${TOP}/${TODO}.tar ended"

# run from subModelTransfer.sh

# can run as depend:
# qsub -W depend=afterok:7931796.h60s1 transfertoarchive.sh

#  P
