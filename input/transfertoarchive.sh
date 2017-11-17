#!/bin/bash
#PBS -m be
#PBS -M jklymak@gmail.com
#PBS -l select=1:ncpus=1
#PBS -l walltime=12:00:00
#PBS -q transfer
#PBS -A ONRDC35552400
#PBS -j oe

TOP=LeeWave3d
#TODO=Shelf1km01

TODO=${PBS_JOBNAME}

cd ${WORKDIR}/${TOP}
tar cfv ${TODO}_files.tar  ${WORKDIR}/${TOP}/${TODO}
gzip ${TODO}_files.tar
rsh ${ARCHIVE_HOST} mkdir ${ARCHIVE_HOME}/${TOP}
rcp ${TODO}_files.tar.gz  ${ARCHIVE_HOST}:${ARCHIVE_HOME}/${TOP}
rsh ${ARCHIVE_HOST} ls -l --block-size=M ${ARCHIVE_HOME}/${TOP}
echo "Transfer job ${TOP}/${TODO} ended"
rm ${TODO}_files.tar.gz
# run from subModelTransfer.sh

# can run as depend:
# qsub -W depend=afterok:7931796.h60s1 transfertoarchive.sh

#  P
