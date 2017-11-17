#!/bin/sh -l
#PBS -m be
#PBS -M jklymak@gmail.com
#PBS -l select=1:ncpus=16:mpiprocs=16
#PBS -l walltime=00:00:20
#PBS -q debug
#PBS -A ONRDC35552400
#PBS -j oe
#PBS -N lee3dlow01U10

cd $PBS_O_WORKDIR
# top=$1  Passed as qsub  -v top=h60h27f20 runModel.sh

PARENT=LeeWave3d

top=${PBS_JOBNAME}
results=${WORKDIR}/${PARENT}/
outdir=$results$top

# These should already be copied
#cp data $outdir/_Model/input
#cp eedata $outdir/_Model/input
#cp data.* $outdir/_Model/input
#cp ../build/mitgcmuv $outdir/_Model/build

#printf "Copying to archive"
#rm -rf ../archive/$top
#cp -r $outdir/_Model/ ../archive/$top
#rm -rf ../archive/$top/indata/*

cd $outdir/_Model/input
pwd

ls -al ../build/mitgcmuv
printf "Starting: $outdir\n"
# mpirun -np 128 ../build/mitgcmuv > mit.out

# Now archive
TODO=${PBS_JOBNAME}
cd ${WORKDIR}/${PARENT}
rm -f archive_job
cat > archive_job << END
#!/bin/bash
#PBS -m be
#PBS -M jklymak@gmail.com
#PBS -l walltime=24:00:00
#PBS -q transfer
#PBS -A ONRDC35552400
#PBS -l select=1:ncpus=1
#PBS -j oe
#PBS -S /bin/bash

echo "Transfer job ${PARENT}/${TODO} Started"
cd ${WORKDIR}/${PARENT}
tar cfv ${TODO}_files.tar  ${WORKDIR}/${PARENT}/${TODO}
gzip ${TODO}_files.tar
rsh ${ARCHIVE_HOST} mkdir ${ARCHIVE_HOME}/${PARENT}
rcp ${TODO}_files.tar.gz  ${ARCHIVE_HOST}:${ARCHIVE_HOME}/${PARENT}
rsh ${ARCHIVE_HOST} ls -l --block-size=M ${ARCHIVE_HOME}/${PARENT}
echo "Transfer job ${PARENT}/${TODO} ended"
END
#
# Submit the archive job script.
qsub archive_job
# End of batch job


