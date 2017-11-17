#!/bin/bash
ONE=$(qsub -N lee3dlow04U10 transfertoarchive.sh)
echo $ONE
two=$(qsub -W depend=afterany:$ONE -N lee3dlow05U10 transfertoarchive.sh)
echo $two
qsub -W depend=afterany:$two -N lee3dlow06U10 transfertoarchive.sh

