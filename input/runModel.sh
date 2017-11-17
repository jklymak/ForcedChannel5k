#!/bin/sh -l
#PBS -m be
#PBS -M jklymak@gmail.com
#PBS -l select=2:ncpus=16:mpiprocs=16
#PBS -l walltime=21:10:00
#PBS -q standard
#PBS -A ONRDC35552400
#PBS -j oe
#PBS -N ${JOBNAME}


# This takes 53 minutes  for 10 h, so 200 h = 17.67h
#


cd $PBS_O_WORKDIR
# top=$1  Passed as qsub  -v top=h60h27f20 runModel.sh

PARENT=LWCoarse2

top=${PBS_JOBNAME}
results=${WORKDIR}/${PARENT}/
outdir=$results$top

# These should already be copied
#cp data $outdir/_Model/input
#cp eedata $outdir/_Model/input
cp dataRestart $outdir/_Model/input/data
#cp ../build/mitgcmuv $outdir/_Model/build

#printf "Copying to archive"
#rm -rf ../archive/$top
#cp -r $outdir/_Model/ ../archive/$top
#rm -rf ../archive/$top/indata/*

cd $outdir/_Model/input
pwd

ls -al ../build/mitgcmuv
printf "Starting: $outdir\n"
mpirun -np 32 ../build/mitgcmuv > mit.out
