#!/bin/bash

for todo in "Channel5k1000_linquaddrag_01" "Channel5k1000_highlinquadspread_01" "Channel5k1000_lindrag_01" "Channel5k1000_doubledrag_01"
do
  echo "transfering ${todo}"
  qsub -N "${todo}" transfertoarchive.sh
done
