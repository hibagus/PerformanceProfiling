#!/bin/bash
# Default Value
NUM_GPUS=${1:-1}
#SECOND_DELAY=${3:-1}


# Read NVIDIA-SMI Output

echo "GPU,TX (MiB/s),RX (MiB/s),Total (MiB/s)"
#Loop Until Terminated
for (( ; ; ))
do
  AGGRTX=0
  AGGRRX=0 
  # Read NVIDIA-SMI Output
  for (( GPU = 0; GPU < $NUM_GPUS; GPU=GPU+1))
  do
    readarray -t OUTPUT < <(nvidia-smi dmon -i ${GPU} -c 1 -s t)
    TEMP=($(echo ${OUTPUT[2]}))
    TOTAL=$((${TEMP[1]}+${TEMP[2]}))
    echo "${GPU},${TEMP[1]},${TEMP[2]},${TOTAL}"
    
    AGGRTX=$(($AGGRTX+${TEMP[1]}))
    AGGRRX=$(($AGGRRX+${TEMP[2]}))

  done
  AGGRTOT=$(($AGGRTX+$AGGRRX))
  echo "S,${AGGRTX},${AGGRRX},${AGGRTOT}"
  sleep 1
done

