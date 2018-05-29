#!/bin/bash

for todo in "Channel5k1000_01" "Channel5k1000_02" "Channel5k1000_vrough_01" "Channel5k1000_spreadrough_01" "Channel5k1000_halfrough_01"
do
  echo "transfering ${todo}"
  qsub -N "${todo}" transfertoarchive.sh
done
